# Contains the core methods for the computation of the (D)SSIM and its gradients,
# but not in a particularly user-friendly form.

const STD = 1.5f0       # Gaussian standard deviation
const RADIUS = 5i32     # Radius of the Gaussian kernel
const KERNEL_UNNORMALIZED = @SVector [
    exp(-0.5f0 * (x / STD)^2) for x = -RADIUS:RADIUS
]  
# (Also with the prefactor of 1 / (STD * sqrt(2 * Float32(π))), this does not sum to 1.)
const KERNEL = KERNEL_UNNORMALIZED ./ sum(KERNEL_UNNORMALIZED)

# Stabilization constants in the SSIM map formula denominator.
const C1 = 0.01f0 ^ 2
const C2 = 0.03f0 ^ 2
# This assumes intensities lie between 0 and 1.

# Each image in an image batch is handled separately, reusing internal buffers (like shared
# memory). We split each image into non-overlapping tiles. Each tile is handled by a single
# block, but blocks might handle multiple (strided) tiles.
const TILE_WIDTH = 8i32
const TILE_HEIGHT = 32i32
# Obtained by testing many configurations on an RTX 3070.
# Also fits comfortably into standard shared memory (48 KiB) for RGB images.

const MAX_NB_TILES_PER_BLOCK = 1i32
# (So actually every block only deals with one tile per image in a batch.)
# Use a const MAX_ upper bound, so that the compiler can unroll the loop.

const NB_PXS_PER_TILE = TILE_WIDTH * TILE_HEIGHT
const NB_THREADS = NB_PXS_PER_TILE  # Can also be set to something different
# In our implementation each thread corresponds to one or more pixel locations (v, u) in
# every tile for our block. We use v and u instead of y and x to avoid confusion with the
# threadIdx dimensions. Our images will always have size (c, h, w), where c is the number of
# color channels (possibly 1), h is the height and w the width of the images. We also batch
# images together to end up with (c, h, w, b), where b is the batch size. Every thread will
# handle all channels, so that we use 2D kernels, where the x-index then corresponds to the
# vertical (height, v) axis, and the y-index to the horizontal (width, u) axis.

# Each thread has to compute the value in the SSIM map at its locations (v, u). This
# requires the 2D convolution of the input images at such a location, for which we need 
# [v - RADIUS : v + RADIUS, u - RADIUS : u + RADIUS].
# On the thread-block level, this means we need to load in a tile of the input pixels at our
# thread locations, together with a margin of RADIUS at all sides.

# We will compute the separable 2D convolutions via 2 1D convolutions (first u, then v), in
# shared memory. Since each block computes complete tiles in the SSIM map, after the
# convolutions we need a tile of the block size: TILE_WIDTH x TILE_HEIGHT.
const SHMEM_CONV_UV_WIDTH = TILE_WIDTH
const SHMEM_CONV_UV_HEIGHT = TILE_HEIGHT
# This is the block-level output of the v-convolution, which means the input must have size
const SHMEM_CONV_U_WIDTH = SHMEM_CONV_UV_WIDTH
const SHMEM_CONV_U_HEIGHT = SHMEM_CONV_UV_HEIGHT + 2 * RADIUS
# This in turn is the block-level output of the u-convolution, meaning its input, i.e.
# the required size before any convolution is
const SHMEM_PRECONV_WIDTH = SHMEM_CONV_U_WIDTH + 2 * RADIUS
const SHMEM_PRECONV_HEIGHT = SHMEM_CONV_U_HEIGHT
# We need to load in enlarged tiles of this size into shared memory, in order to produce
# the convolution output tile.
const NB_PXS_TO_LOAD_TO_SHMEM = SHMEM_PRECONV_HEIGHT * SHMEM_PRECONV_WIDTH
const NB_CONV_U_PXS = SHMEM_CONV_U_HEIGHT * SHMEM_CONV_U_WIDTH

const MAX_PX_PER_THREAD = cld(NB_PXS_PER_TILE, NB_THREADS)
# For certain choices of TILE_WIDTH, TILE_HEIGHT and NB_THREADS a thread might need to deal
# with multiple pixels in a tile. (Not relevant for the chosen constants: 1)
const MAX_PX_PER_THREAD_I64 = Int(MAX_PX_PER_THREAD)
# Need Int64 (not Int32) to use in SVector size
const MAX_CONV_U_PX_PER_THREAD = cld(SHMEM_CONV_U_WIDTH * SHMEM_CONV_U_HEIGHT, NB_THREADS)
# Similarly, but now for the pixels in a u-convolved tile a thread needs to handle when
# working on the v-convolution. (Relevant for the chosen constants.)



"""
    _ssim_kernel_fwd!(
        N_ssims, 
        N_dssims_dQ, N_dssims_dM, N_dssims_dP, 
        imgs1, imgs2,
        ::Val{nb_channels}, ::Val{single_image_pair},
        ::Val{use_dynamic_shmem},
        ::Val{skip_ssim}, ::Val{skip_backward}
    ) where {nb_channels, single_image_pair, use_dynamic_shmem, skip_ssim, skip_backward}
    -> Nothing

Perform the forward pass of the SSIM, possibly storing relevant values for the backward
pass.

We use a 1D grid of blocks, where each block consists of a 1D arrangement of threads.
Pixels (all color channels) of the full SSIM maps (one for each image in the image batch)
are produced by a single thread, but threads might deal with multiple pixels in a strided
fashion.

The full SSIM map for each image is partitioned into 2D tiles. Each tile is handled by a
single block, though again a block might deal with multiple tiles in a strided fashion.
We iterate over the images in a batch in the supplied order.

# Arguments
- `N_ssims`: A `CuVector` (or similar) of `length` the batch size. It will store for each
    input image the output sum of its SSIM map. To obtain the actual SSIMs, this still needs
    to be divided by the number `N` of pixel components per image 
    (`N = prod(size(imgs1)[1:3])`). We do not yet perform this division inside of this
    kernel for reasons of numerical stability.
    Only relevant when `!skip_ssim`.
    !!! The initial values must be zeros. !!!
- `N_dssims_dQ`, `N_dssims_dM`, `N_dssims_dP`: Intermediate gradients to store for the
    backward pass, multiplied by `N`. See the mathematics section in the documentation for
    more information. For memory coalescing the `size`s are (h, c, w, b), where c is the
    number of color channels (possibly 1) in the input images, h and w are the height and
    width of the images, and b (possibly 1) is the batch size.
    Only relevant when `!skip_backward`.
- `imgs1`: The first input image batch in normalized floating point format, and of `size`
    (c, h, w, b). The eventual backward pass gradients are computed with respect to these
    images.
- `imgs2`: The second input image batch with the same format as the first. This is
    considered a constant for the gradients.
- `nb_channels`: The number of color channels c. Statically known for possible
    optimisation by the compiler (like loop unrolling). The amount of required shared
    memory in the kernel grows linearly with `nb_channels`, so it is probably a bad idea
    to process e.g. hyperspectral images.
- `single_image_pair`: Whether the batch size is 1. Statically known for allowing the
    compiler to elide the batch-loop.
- `use_dynamic_shmem`: Whether to use `CuStaticSharedArray`s (`false`) or
    `CuDynamicSharedArray`s (`true`) for shared memory. In the latter case the `@cuda` call
    also needs to include the amount of requested shared memory. When using dynamic shared
    memory, we can request more than with static memory. On the other hand, it is slightly
    slower (when we do not exceed the static bound).
- `skip_ssim`: Whether to output `N_ssims` (in-place). When we are only interested in the
    gradients and not the SSIMs themselves, this can be skipped.
- `skip_backward`: Whether to fill the gradient maps `N_dssims_dQ`, `N_dssims_dM` and
    `N_dssims_dP`. When we are only interested in the SSIMs themselves, and not the
    gradients, this can be skipped.
"""
function _ssim_kernel_fwd!(
    N_ssims, 
    N_dssims_dQ, N_dssims_dM, N_dssims_dP, 
    imgs1, imgs2,
    nb_channels_val::Val{nb_channels}, ::Val{single_image_pair},
    use_dynamic_shmem_val,
    ::Val{skip_ssim}, ::Val{skip_backward}
) where {nb_channels, single_image_pair, skip_ssim, skip_backward}

    _, img_height, img_width, batch_size = size(imgs1)
    single_image_pair && ( batch_size = 1 )  # batch_size becomes statically known
    T1 = eltype(imgs1); T2 = eltype(imgs2); TK = eltype(KERNEL)
    Tc = promote_type(T1, T2, TK)  # used for convolution
    Ts = eltype(N_ssims)           # used for output ssims 

    shmem_img1, shmem_img2, shmem_conv_u = _get_shmem_fwd(
        T1, T2, Tc, nb_channels_val, use_dynamic_shmem_val)
    # (Needs a function barrier for type inference on shmem_img1 etc.)

    thread_idx_in_block = threadIdx().x  # 1D index
    # blockDim().x == NB_THREADS

    nb_tiles_u = cld(img_width, TILE_WIDTH)
    nb_tiles_v = cld(img_height, TILE_HEIGHT)
    nb_tiles = nb_tiles_u * nb_tiles_v
    tile_idx_to_2D = CartesianIndices((nb_tiles_v, nb_tiles_u))
    # Converts a 1D tile index to 2D

    block_idx = blockIdx().x  # 1D index of our block
    nb_blocks = gridDim().x   # Number of requested blocks

    batch_idx = 1i32  # The image in imgs1 and imgs2 we are currently considering
    # Conceptually within each iteration we only consider 
    #   img1 = imgs1[:, :, :, batch_idx]
    # and similar for img2
    @inbounds while batch_idx <= batch_size
    # (Every memory access in the entire function is inbounds)
        tile_idx = block_idx  # 1D index of the current tile our block is working on
        block_tile_counter = 1i32
        while block_tile_counter <= MAX_NB_TILES_PER_BLOCK
            tile_idx <= nb_tiles || break
            # Determine tiles boundaries: (current) actual output tile + necessary
            # boundaries for convolution
            tile_v, tile_u = Tuple(tile_idx_to_2D[tile_idx])
            conv_uv_tile_start_u0 = (tile_u - 1i32) * TILE_WIDTH    # 0-indexed
            conv_uv_tile_start_v0 = (tile_v - 1i32) * TILE_HEIGHT   # a.k.a. tile_start_v0
            # Cf. SHMEM_CONV_U_WIDTH and _HEIGHT
            conv_u_tile_start_u0 = conv_uv_tile_start_u0
            conv_u_tile_start_v0 = conv_uv_tile_start_v0 - RADIUS
            preconv_tile_start_u0 = conv_u_tile_start_u0 - RADIUS
            preconv_tile_start_v0 = conv_u_tile_start_v0

            # Load tile with boundary into shared memory
            load_idx = thread_idx_in_block  # local 1D index within enlarged tile to load
            load_idx_to_2D = CartesianIndices((SHMEM_PRECONV_HEIGHT, SHMEM_PRECONV_WIDTH))
            while load_idx <= NB_PXS_TO_LOAD_TO_SHMEM
                load_v, load_u = Tuple(load_idx_to_2D[load_idx])
                # local 2D index, within the enlarged tile
                img_u = load_u + preconv_tile_start_u0  # global 2D index,
                img_v = load_v + preconv_tile_start_v0  # within the input images
                ch = 1i32
                if 1i32 <= img_u <= img_width && 1i32 <= img_v <= img_height
                    while ch <= nb_channels
                        shmem_img1[ch, load_v, load_u] = imgs1[ch, img_v, img_u, batch_idx]
                        shmem_img2[ch, load_v, load_u] = imgs2[ch, img_v, img_u, batch_idx]
                        ch += 1i32
                    end
                else
                    while ch <= nb_channels
                        shmem_img1[ch, load_v, load_u] = T1(0)  # zero-padding
                        shmem_img2[ch, load_v, load_u] = T2(0)  # (Could also just write 0)
                        ch += 1i32
                    end
                end 
                load_idx += NB_THREADS
            end
            sync_threads()

            # Convolve (the shared memory versions of) img1, img1 .^ 2, img1 .* img2, img2 
            # and img2 .^ 2 current image tiles with the KERNEL
            @inline function _conv_uv(img_val_fun)
                _conv_u!(
                    shmem_conv_u,
                    img_val_fun,
                    thread_idx_in_block,
                    nb_channels_val
                )
                return _conv_v(
                    shmem_conv_u, 
                    thread_idx_in_block,
                    conv_uv_tile_start_u0, conv_uv_tile_start_v0, 
                    img_width, img_height, nb_channels_val
                )  
            end
            
            thread_pxs_img1_first_moment = _conv_uv(I -> shmem_img1[I])
            # An SMatrix containing for each pixel (v, u) our thread handles and each color
            # channel ch, the local first moment (averaged according to KERNEL) of
            # img1[ch, :, :] at (v, u).
            thread_pxs_img1_second_moment = _conv_uv(
                I -> @inbounds(Tc(shmem_img1[I])) ^ 2)
            # E.g. in case T1 === T2 === N0f8 we want this squaring to happen in 
            # Tc === Float32 precision. After multiplying with the KERNEL values we end up
            # in Tc anyway.
            thread_pxs_imgs_mixed_moment = _conv_uv(
                I -> @inbounds(Tc(shmem_img1[I]) * shmem_img2[I]))
            thread_pxs_img2_first_moment = _conv_uv(
                I -> @inbounds(shmem_img2[I]))
            thread_pxs_img2_second_moment = _conv_uv(
                I -> @inbounds(Tc(shmem_img2[I])) ^ 2)
            # Needs explicit @inbounds as the general @inbounds of the while loop does not
            # propagate through closures.

            # Compute and reduce SSIM map
            pixel_idx_in_tile = thread_idx_in_block  # 1D index within our tile 
            pixel_idx_to_2D = CartesianIndices((TILE_HEIGHT, TILE_WIDTH))
            if !skip_ssim
                thread_sum_ssim_px::Ts = 0
                # sum over all ssim map pixels our thread deals with, for the current image
                # Note: not
                #   thread_sum_ssim_px = Ts(0)
                # as then after
                #   thread_sum_ssim_px += s
                # the type might change to the promotion type with s.
                # Because out of bounds threads do not have an s value to add, their
                # thread_sum_ssim_px type remains of type Ts. Having different types causes
                # a freeze in the block reduction below.
            end
            thread_pixel_counter = 1i32
            while thread_pixel_counter <= MAX_PX_PER_THREAD
                pixel_idx_in_tile <= NB_PXS_PER_TILE || break
                local_v, local_u = Tuple(pixel_idx_to_2D[pixel_idx_in_tile])
                global_v = conv_uv_tile_start_v0 + local_v  # 2D index within img1, img2
                global_u = conv_uv_tile_start_u0 + local_u  # and within the SSIM map
                if global_u <= img_width && global_v <= img_height  
                # (1 <=  is not necessary, as conv_uv_tile_start_v0, _u0 >= 0)
                    ch = 1i32
                    while ch <= nb_channels
                        # Notation cf. maths in documentation
                        m = thread_pxs_img1_first_moment[ch, thread_pixel_counter]
                        m_R = thread_pxs_img2_first_moment[ch, thread_pixel_counter]
                        q = thread_pxs_img1_second_moment[ch, thread_pixel_counter]
                        q_R = thread_pxs_img2_second_moment[ch, thread_pixel_counter]
                        p = thread_pxs_imgs_mixed_moment[ch, thread_pixel_counter]

                        m_sq = m * m
                        v = q - m_sq
                        mm_R = m * m_R
                        c = p - mm_R

                        m_R_sq = m_R * m_R
                        v_R = q_R - m_R_sq

                        x = 2 * mm_R + C1
                        y = 2 * c + C2
                        z = m_sq + m_R_sq + C1
                        t = v + v_R + C2
                        zt_inv = 1 / (z * t)
                        s = x * y * zt_inv

                        if !skip_ssim
                            thread_sum_ssim_px += s
                        end
                        if !skip_backward
                            N_dssims_dQ[global_v, ch, global_u, batch_idx] = -s / t 
                            N_dssims_dM[global_v, ch, global_u, batch_idx] = 
                                2 * zt_inv * ((y - x) * m_R + s * (z - t) * m) 
                            N_dssims_dP[global_v, ch, global_u, batch_idx] = 2 * x * zt_inv
                        end 
                        ch += 1i32
                    end  # channels loop
                end  # if global_ inbounds

                pixel_idx_in_tile += NB_THREADS
                thread_pixel_counter += 1i32
            end  # loop over pixels for this thread

            if !skip_ssim
                tile_ssim_sum = CUDA.reduce_block(
                    +, thread_sum_ssim_px, Ts(0), Val(true))
                # The total SSIM sum is the sum of these tile sums.
                # (Note that also threads outside of the image boundaries need to
                # contribute to the reduction. Since thread_sum_ssim_px is initialized
                # at 0, this is not an issue.)
                if thread_idx_in_block == 1
                    CUDA.@atomic N_ssims[batch_idx] += tile_ssim_sum
                    # If at this point we divide by the number of pixels components per
                    # image, then e.g.
                    #   ssim(img1, img1) = 0.9999624f0 < 1f0
                    # for two full HD zero images, because of accumulating rounding
                    # errors. Therefore, we divide as late as possible: after the
                    # kernel.
                end
                # Slightly faster (due to clever compiler optimizations regarding
                # atomics) than outputting block- or warp-level reduced SSIM maps and
                # using sum afterwards. 
            end
            
            tile_idx += nb_blocks
            block_tile_counter += 1i32
        end  # loop over tiles

        batch_idx += 1i32
    end  # loop over images in the batch

    return
end


"""
    _get_shmem_fwd(
        T1, T2, Tc, 
        ::Val{nb_channels}, 
        ::Val{use_dynamic_shmem}
    ) where {nb_channels, use_dynamic_shmem}

Return the shared memory arrays required for `_ssim_kernel_fwd!`.

# Arguments
- `T1`: The `eltype` of the first image batch.
- `T2`: The `eltype` of the second image batch.
- `Tc`: The `eltype` after convolution with the KERNEL.
- `nb_channels`: The number of color channels.
- `use_dynamic_shmem`: Whether to use `CuStaticSharedArray` (`false`) or 
    `CuDynamicSharedArray` (`true`).

# Returns
Three arrays in shared memory:
1. `shmem_img1`: To store our block's current tile in the current image from the first 
    image batch. It will be the input for the convolutions, so we enlarge this tile at the
    boundaries. It satisfies `eltype(shmem_img1) == T1 && \
    size(shmem_img1) == (nb_channels, SHMEM_PRECONV_HEIGHT, SHMEM_PRECONV_WIDTH)`.
1. `shmem_img2`: Similar, for the second image batch. It has the same size as `shmem_img1`,
    but `eltype` `T2`.
1. `shmem_conv_u`: To store the u-convolution outputs. Its `eltype` is `Tc` and its `size`
    `(nb_channels, SHMEM_CONV_U_HEIGHT, SHMEM_CONV_U_WIDTH)`.
"""
function _get_shmem_fwd end

@inline function _get_shmem_fwd(
    T1, T2, Tc, 
    ::Val{nb_channels}, 
    ::Val{false}
) where nb_channels
    shmem_img1 = CuStaticSharedArray(
        T1,
        (nb_channels, SHMEM_PRECONV_HEIGHT, SHMEM_PRECONV_WIDTH)
    )  
    shmem_img2 = CuStaticSharedArray(T2, size(shmem_img1))
    shmem_conv_u = CuStaticSharedArray(
        Tc, 
        (nb_channels, SHMEM_CONV_U_HEIGHT, SHMEM_CONV_U_WIDTH), 
    )
    return shmem_img1, shmem_img2, shmem_conv_u
end

@inline function _get_shmem_fwd(
    T1, T2, Tc, 
    ::Val{nb_channels}, 
    ::Val{true}
) where nb_channels
    shmem_img1 = CuDynamicSharedArray(
        T1,
        (nb_channels, SHMEM_PRECONV_HEIGHT, SHMEM_PRECONV_WIDTH)
    )  
    offset = sizeof(shmem_img1)
    shmem_img2 = CuDynamicSharedArray(T2, size(shmem_img1), offset)
    offset += sizeof(shmem_img2)
    shmem_conv_u = CuDynamicSharedArray(
        Tc, 
        (nb_channels, SHMEM_CONV_U_HEIGHT, SHMEM_CONV_U_WIDTH),
        offset
    )
    return shmem_img1, shmem_img2, shmem_conv_u
end


"""
    _conv_u!(
        shmem_conv_u,
        img_val_function, 
        thread_idx_in_block,
        ::Val{nb_channels},
        order_function = identity
    ) where nb_channels
    -> Nothing

Compute the u-convolution of the image data accessed via the `img_val_function`. This
should refer to an enlarged tile of `size` `(SHMEM_CONV_U_HEIGHT, SHMEM_CONV_U_WIDTH)`. 

# Arguments
- `shmem_conv_u`: The convolved output is written away to this (shared memory) 3D array.
- `img_val_function`: Maps a 3D `CartesianIndex` into the (lazily computed) image intensity
    value at this location. E.g. `I -> @inbounds(shmem_img1[I] * shmem_img2[I])`.
- `thread_idx_in_block`: The linear position of the executing thread in its block.
- `nb_channels`: The number of color channels in the image.
- `order_function`: A function taking in a `CartesianIndex` in channel, height, width order
    (which is also the order we iterate in, optimal in a per-thread cache-friendly fashion)
    and putting out the same `CartesianIndex` in the order required for `img_val_fun` and 
    `shmem_conv_u` (which we assume is the same). Defaults to `identity`.
"""
@inline function _conv_u!(
    shmem_conv_u, 
    img_val_function, 
    thread_idx_in_block,
    ::Val{nb_channels},
    order_function = identity
) where nb_channels

    Tc = eltype(shmem_conv_u)
    output_idx = thread_idx_in_block  # 1D, local within the u-convolved padded tile
    output_idx_to_2D = CartesianIndices((SHMEM_CONV_U_HEIGHT, SHMEM_CONV_U_WIDTH))

    thread_pixel_counter = 1i32
    @inbounds while thread_pixel_counter <= MAX_CONV_U_PX_PER_THREAD
    # Slightly faster than   while output_idx <= NB_CONV_U_PXS   as the number of
    # iterations is (more directly) statically known.
        output_idx <= NB_CONV_U_PXS || break  # (or  <= length(output_idx_to_2D) )
        out_v, out_u = Tuple(output_idx_to_2D[output_idx])
        
        ch = 1i32 
        while ch <= nb_channels
            conv_val::Tc = 0  # convolution (cross-correlation) output value
            i = 0i32
            while i <= 2 * RADIUS
                ker_val = KERNEL[i + 1i32]
                I_in = CartesianIndex(ch, out_v, out_u + i)
                img_val = img_val_function(order_function(I_in))
                # Note that e.g.
                #   shmem_img1[ch, v, u] == img1[ch, out_v - RADIUS, out_u - RADIUS] 
                # where inbounds
                conv_val += img_val * ker_val
                i += 1i32
            end
               
            I_out = CartesianIndex(ch, out_v, out_u)
            shmem_conv_u[order_function(I_out)] = conv_val
            # Note that we write in per-thread optimal memory order, though not
            # in a warp-coalesced order.
            ch += 1i32
        end

        output_idx += NB_THREADS
        thread_pixel_counter += 1i32
    end

    sync_threads()
end

"""
    _conv_v(
        shmem_conv_u, 
        thread_idx_in_block,
        conv_uv_tile_start_u0, conv_uv_tile_start_v0, 
        img_width, img_height, ::Val{nb_channels},
        order_function = identity
    ) where nb_channels
    -> SMatrix{nb_channels, $MAX_PX_PER_THREAD_I64, eltype(shmem_conv_u)}

Compute the v-convolution of the u-convolved local image.

# Arguments
- `shmem_conv_u`: The input (in shared memory), which is the already u-convolved local
    image.
- `thread_idx_in_block`: The position of the executing thread in its (1D) block.
- `conv_uv_tile_start_u0`: The 0-indexed first global u-coordinate of the output image tile
    (i.e. after both the u- and now the v-convolution) we are working in.
- `conv_uv_tile_start_v0`: Similar, but the v-coordinate. 
- `img_width`: The number of pixels along the image's horizontal direction (`size(⋅, 3)`).
- `img_height`: The number of pixels along the image's vertical direction (`size(⋅, 2)`). 
- `nb_channels`: The number of color channels in the image (`size(⋅, 1)`), but statically
    known.
- `order_function`: A function taking in a `CartesianIndex` in channel, height, width
    order (which is also the (per-thread cache-friendly) order we iterate in) and putting
    out the same `CartesianIndex` in the order required for `shmem_conv_u`.
    Defaults to `identity`.

# Return
Since every thread might work on multiple pixels ($MAX_PX_PER_THREAD_I64 for our choice
of (`const`) settings) and every pixel might have multiple channels, we return an 
`SMatrix{nb_channels, $MAX_PX_PER_THREAD_I64}` of the convolution outputs our thread
computed.
"""
@inline function _conv_v(
    shmem_conv_u, 
    thread_idx_in_block,
    conv_uv_tile_start_u0, conv_uv_tile_start_v0, 
    img_width, img_height, ::Val{nb_channels},
    order_function = identity
) where nb_channels

    Tc = eltype(shmem_conv_u)
    thread_convolved_outputs = @SMatrix zeros(Tc, nb_channels, MAX_PX_PER_THREAD_I64)
    output_idx = thread_idx_in_block  # 1D, local within the uv-convolved tile
    output_idx_to_2D = CartesianIndices((SHMEM_CONV_UV_HEIGHT, SHMEM_CONV_UV_WIDTH))

    thread_pixel_counter = 1i32 
    @inbounds while thread_pixel_counter <= MAX_PX_PER_THREAD
    # Similar to the _conv_u! approach, potentially slightly (but in practice seemingly
    # negligibly) faster than   while output_idx <= NB_PXS_PER_TILE   as the number of
    # iterations is (more directly) statically known
        output_idx <= NB_PXS_PER_TILE || break  # (or  <= length(output_idx_to_2D) )
        out_v, out_u = Tuple(output_idx_to_2D[output_idx])

        if conv_uv_tile_start_u0 + out_u <= img_width && 
            conv_uv_tile_start_v0 + out_v <= img_height
            # No problem if we are out of bounds, as this will be rechecked in
            # _ssim_kernel_fwd!. But of course, if we can skip some computations, we should.
            # There might be missing entries in thread_convolved_outputs, but that's fine,
            # as we'll also skip them in the main kernel.
            ch = 1i32 
            while ch <= nb_channels
                conv_val::Tc = 0
                i = 0i32 
                while i <= 2 * RADIUS
                    I_in = CartesianIndex(ch, out_v + i, out_u)
                    conv_val += shmem_conv_u[order_function(I_in)] * KERNEL[i + 1i32]
                    i += 1i32
                end
                @reset thread_convolved_outputs[ch, thread_pixel_counter] = conv_val
                ch += 1i32
            end
        end
            
        output_idx += NB_THREADS
        thread_pixel_counter += 1i32
    end
    
    sync_threads()  
    # So that shmem_conv_u does not get overwritten in the next _conv_u! call
    return thread_convolved_outputs
end


"""
    _ssim_kernel_bwd!(
        dL_dimgs1, 
        N_dssims_dQ, N_dssims_dM, N_dssims_dP, 
        imgs1, imgs2,
        ::Val{nb_channels}, ::Val{single_image_pair},
        ::Val{use_dynamic_shmem},
        use_dssim, divide_by_two_in_dssim
    ) where {nb_channels, single_image_pair, use_dynamic_shmem}
    -> Nothing

Perform the backward pass of the (D)SSIM with respect to the first image batch, given the
relevant intermediate values from the forward pass.

We use the same grid-block-thread structure as in the forward pass.

# Arguments
- `dL_dimgs1`: The output gradient 'image' batch of the same format as `imgs1`, that is,
    normalized floating point format with `size` (c, h, w, b) where c is the number of color
    channels (possibly 1), h is the image height, w the width and b is the batch size
    (possibly 1). L is either the SSIM or the DSSIM depending on `use_dssim`.
- `N_dssims_dQ`, `N_dssims_dM`, `N_dssims_dP`: Intermediate gradients from the forward pass.
    Their size is (h, c, w, b). (Note that the 'd' here is for _d_ifferentiation in Leibniz
    notation (dy/dx), not for _d_ssim.)
- `imgs1`: The first input image batch in normalized floating point format, and of size
    (c, h, w, b). The gradients are computed with respect to this batch.
- `imgs2`: The second input image batch with the same format as the first. This is
    considered a constant for the gradients.
- `nb_channels`: The number of color channels c. Statically known for possible
    optimisation by the compiler (like loop unrolling). The amount of required shared
    memory in the kernel again grows linearly with `nb_channels`.
- `single_image_pair`: Whether the batch size is 1. Statically known for allowing the
    compiler to elide the batch-loop.
- `use_dynamic_shmem`: Whether to use `CuStaticSharedArray`s (`false`) or
    `CuDynamicSharedArray`s (`true`) for shared memory.
- `use_dssim`: Whether to compute the gradient with respect to the SSIM (`false`) or the 
    DSSIM (`true`).
- `divide_by_two_in_dssim`: There are two common formulas for the DSSIM: 1 - SSIM, used when
    this parameter is set to `false`, and (1 - SSIM) / 2, used when set to `true`.
"""
function _ssim_kernel_bwd!(
    dL_dimgs1, 
    N_dssims_dQ, N_dssims_dM, N_dssims_dP, 
    imgs1, imgs2, 
    nb_channels_val::Val{nb_channels}, ::Val{single_image_pair},
    use_dynamic_shmem_val,
    use_dssim, divide_by_two_in_dssim
) where {nb_channels, single_image_pair}

    # The general structure is very similar as in the forward pass.
    _, img_height, img_width, batch_size = size(imgs1)
    single_image_pair && ( batch_size = 1 )
    nb_pixel_components_per_image = nb_channels * img_height * img_width  # a.k.a. N
    TK = eltype(KERNEL); Td1 = eltype(dL_dimgs1); TdX = eltype(N_dssims_dQ)
    # We assume eltype(..._dM) == TdX == eltype(..._dP)
    Tc = promote_type(TK, TdX)  # used in convolution

    if use_dssim && divide_by_two_in_dssim
        N_inv = Td1(0.5) / nb_pixel_components_per_image  
        # (So technically no longer N^(-1))
    else
        N_inv = Td1(1) / nb_pixel_components_per_image
    end

    shmem_N_dL_dX, shmem_conv_u = _get_shmem_bwd(
        TdX, Tc, nb_channels_val, use_dynamic_shmem_val)
    # (Function barrier needed for type inference)

    @inline function _order_function(I)
        ch, v, u = Tuple(I)
        return CartesianIndex(v, ch, u)
    end  
    # For using _conv_u! and _conv_v with the memory-coalescing optimal memory order
    # of shmem_N_dL_dX

    thread_idx_in_block = threadIdx().x  # 1D index
    # blockDim().x == NB_THREADS

    nb_tiles_u = cld(img_width, TILE_WIDTH)
    nb_tiles_v = cld(img_height, TILE_HEIGHT)
    nb_tiles = nb_tiles_u * nb_tiles_v
    tile_idx_to_2D = CartesianIndices((nb_tiles_v, nb_tiles_u))
    # Converts a 1D tile index to 2D

    block_idx = blockIdx().x  # 1D index of our block
    nb_blocks = gridDim().x   # Number of requested blocks

    batch_idx = 1i32
    @inbounds while batch_idx <= batch_size
        tile_idx = block_idx  # 1D index of the current tile our block is working on
        block_tile_counter = 1i32
        while block_tile_counter <= MAX_NB_TILES_PER_BLOCK
            tile_idx <= nb_tiles || break
            # Determine tiles boundaries: (current) actual output tile + necessary
            # boundaries for convolution
            tile_v, tile_u = Tuple(tile_idx_to_2D[tile_idx])
            conv_uv_tile_start_u0 = (tile_u - 1i32) * TILE_WIDTH    # 0-indexed
            conv_uv_tile_start_v0 = (tile_v - 1i32) * TILE_HEIGHT   # a.k.a. tile_start_v0
            # Cf. _CONV_U_BLOCK_WIDTH and _HEIGHT
            conv_u_tile_start_u0 = conv_uv_tile_start_u0
            conv_u_tile_start_v0 = conv_uv_tile_start_v0 - RADIUS
            preconv_tile_start_u0 = conv_u_tile_start_u0 - RADIUS
            preconv_tile_start_v0 = conv_u_tile_start_v0

            # Load the enlarged N_dssims_dX image tiles to shared memory and convolve them
            @inline function _load_and_conv_uv(
                N_dssims_dX,
                batch_idx,
                skip_zero_padding
            )        
                load_idx = thread_idx_in_block 
                # local 1D index within enlarged tile to load
                load_idx_to_2D = CartesianIndices(
                    (SHMEM_PRECONV_HEIGHT, SHMEM_PRECONV_WIDTH))
                @inbounds while load_idx <= NB_PXS_TO_LOAD_TO_SHMEM
                    load_v, load_u = Tuple(load_idx_to_2D[load_idx])
                    # local 2D index
                    img_u = load_u + preconv_tile_start_u0  # global 2D index
                    img_v = load_v + preconv_tile_start_v0
                    ch = 1i32
                    if 1i32 <= img_u <= img_width && 1i32 <= img_v <= img_height
                        while ch <= nb_channels
                            val = N_dssims_dX[img_v, ch, img_u, batch_idx]
                            shmem_N_dL_dX[load_v, ch, load_u] = ifelse(use_dssim, -val, val)
                            ch += 1i32
                        end
                    elseif !skip_zero_padding
                        while ch <= nb_channels
                            shmem_N_dL_dX[load_v, ch, load_u] = TdX(0)
                            ch += 1i32
                        end
                    end
                    load_idx += NB_THREADS
                end
                sync_threads()

                _conv_u!(
                    shmem_conv_u,
                    I -> @inbounds(shmem_N_dL_dX[I]), 
                    thread_idx_in_block,
                    nb_channels_val,
                    _order_function
                )
                return _conv_v(
                    shmem_conv_u, 
                    thread_idx_in_block,
                    conv_uv_tile_start_u0, conv_uv_tile_start_v0, 
                    img_width, img_height, nb_channels_val,
                    _order_function)  
            end
            
            thread_pxs_N_dL_dimg1_via_M = _load_and_conv_uv(N_dssims_dM, batch_idx, false)
            # Contains for each of the pixel in (dL_d)img1 = (dL_d)imgs1[:, :, :, batch_idx]
            # that our thread deals with, the contribution of the loss (ssim / dssim),
            # passing through M = I_1 * K (see the online documentation for the meaning of
            # the variable names), scaled by the number of pixel components in each image.

            # shmem_N_dL_dX is now zero-padded, so we don't have to keep doing this
            thread_pxs_N_dL_dJ = _load_and_conv_uv(N_dssims_dQ, batch_idx, true)  
            thread_pxs_N_dL_dH = _load_and_conv_uv(N_dssims_dP, batch_idx, true)
            # J = img1 .^ 2; H = img1 .* img2
            
            # Combine to get the full dL_dimg1 for all pixels this thread handles
            pixel_idx_in_tile = thread_idx_in_block  # 1D
            pixel_idx_to_2D = CartesianIndices((TILE_HEIGHT, TILE_WIDTH))
            thread_pixel_counter = 1i32
            while thread_pixel_counter <= MAX_PX_PER_THREAD
                pixel_idx_in_tile <= NB_PXS_PER_TILE || break
                local_v, local_u = Tuple(pixel_idx_to_2D[pixel_idx_in_tile])
                global_v = conv_uv_tile_start_v0 + local_v  # idx within dL_dimg1
                global_u = conv_uv_tile_start_u0 + local_u
                if global_u <= img_width && global_v <= img_height    
                    ch = 1i32
                    while ch <= nb_channels
                        N_dL_dimg1_via_M = 
                            thread_pxs_N_dL_dimg1_via_M[ch, thread_pixel_counter]
                        N_dL_dimg1_via_Q =  # (or via J)
                            2 * imgs1[ch, global_v, global_u, batch_idx] * 
                                thread_pxs_N_dL_dJ[ch, thread_pixel_counter]
                        N_dL_dimg1_via_P =  # (or via H)
                            imgs2[ch, global_v, global_u, batch_idx] * 
                                thread_pxs_N_dL_dH[ch, thread_pixel_counter]

                        dL_dimgs1[ch, global_v, global_u, batch_idx] = 
                            (N_dL_dimg1_via_M + N_dL_dimg1_via_Q + N_dL_dimg1_via_P) * 
                                N_inv
                        ch += 1i32
                    end
                end
                
                pixel_idx_in_tile += NB_THREADS
                thread_pixel_counter += 1i32
            end
        
            tile_idx += nb_blocks
            block_tile_counter += 1i32
        end  # loop over tiles

        batch_idx += 1i32
    end  # loop over images in the batch

    return
end


"""
    _get_shmem_bwd(
        TdX, Tc, 
        ::Val{nb_channels}, 
        ::Val{use_dynamic_shmem}
    ) where {nb_channels, use_dynamic_shmem}

Return the shared memory arrays required for `_ssim_kernel_bwd!`.

# Arguments
- `TdX`: The `eltype` of the gradient buffers `N_dssims_dQ` etc.
- `Tc`: The `eltype` after convolution with the KERNEL.
- `nb_channels`: The number of color channels.
- `use_dynamic_shmem`: Whether to use `CuStaticSharedArray` (`false`) or 
    `CuDynamicSharedArray` (`true`).

# Returns
Two arrays in shared memory:
1. `shmem_N_dL_dX`: To store our block's current enlarged pre-convolution `N_dssims_dX` tile
    for the current image in the batch, for `X = M`, `X = Q` and `X = P`, possibly negated
    when computing the dssim. Its `eltype` is `TdX` and its `size` 
    `(SHMEM_CONV_U_HEIGHT, nb_channels, SHMEM_CONV_U_WIDTH)`.
1. `shmem_conv_u`: To store the u-convolution outputs. Its `eltype` is `Tc` and its `size`
    `(SHMEM_CONV_U_HEIGHT, nb_channels, SHMEM_CONV_U_WIDTH)`.
"""
function _get_shmem_bwd end

@inline function _get_shmem_bwd(
    TdX, Tc, 
    ::Val{nb_channels}, 
    ::Val{false}
) where nb_channels
    shmem_N_dL_dX = CuStaticSharedArray(
        TdX,
        (SHMEM_PRECONV_HEIGHT, nb_channels, SHMEM_PRECONV_WIDTH)
    )  
    shmem_conv_u = CuStaticSharedArray(
        Tc, 
        (SHMEM_CONV_U_HEIGHT, nb_channels, SHMEM_CONV_U_WIDTH), 
    )
    return shmem_N_dL_dX, shmem_conv_u
end

@inline function _get_shmem_bwd(
    TdX, Tc, 
    ::Val{nb_channels}, 
    ::Val{true}
) where nb_channels
    shmem_N_dL_dX = CuDynamicSharedArray(
        TdX,
        (SHMEM_PRECONV_HEIGHT, nb_channels, SHMEM_PRECONV_WIDTH)
    )  
    shmem_conv_u = CuDynamicSharedArray(
        Tc, 
        (SHMEM_CONV_U_HEIGHT, nb_channels, SHMEM_CONV_U_WIDTH),
        sizeof(shmem_N_dL_dX)
    )
    return shmem_N_dL_dX, shmem_conv_u
end



"""
    _ssim_fwd_bwd!(
        ssims, dL_dimgs1, 
        imgs1, imgs2,
        N_dssims_dQ, N_dssims_dM, N_dssims_dP,
        should_zero, 
        ::Val{skip_ssim}, ::Val{skip_backward}, 
        use_dssim, divide_by_two_in_dssim
    ) where {skip_backward, skip_ssim} 

Compute the Structural (Dis)Similarity Index Measure ((D)SSIM) between two image batches
and/or its gradient with respect to the first batch.

Very general method. Users should normally use one of the easier exported wrapper functions,
like [`ssim`](@ref).

Outputs and buffers (`ssims`, `dL_dimgs1`, `N_dssims_dQ`, `N_dssims_dM` and `N_dssims_dP`)
which are necessary but set to `nothing` will be allocated. In this case the `eltype` will
be `promote_type(eltype(img1), eltype(img2), $(eltype(KERNEL)))`.
Allocated buffers will be freed, while allocated outputs will be returned.

# Arguments
- `ssims`: The (output) `CuVector` of `length` the batch size, containing the (D)SSIM
    values.
- `dL_dimgs1`: The (output) gradient of `imgs1` with respect to the (D)SSIM value L. It uses
    the same normalised floating point format in `channels x height x width x batch size`
    memory order as `imgs1`. If not explicitly in this format, we will attempt to convert to
    it using [`_canonicalize_image`](@ref).
- `imgs1`: The first images batch in the aforementioned format. Also here, we will use
    `_canonicalize_image` in order to be able to handle e.g. a `CuMatrix{RGB{Float32}}`,
    which will be converted to a `CuArray{Float32, 4}` of size
    `(3, size(..., 1), size(..., 2), 1)`, using the same underlying memory.
- `imgs2`: The second image batch, in the same format.
- `N_dssims_dQ`: A buffer of the same `length` as `imgs1` and `imgs2`, but in
    height x channels x width x batch size order. We will attempt to use
    [`_canonicalize_gradient_buffer`](@ref) to convert to it, if possible. As we will not
    move memory around, only reinterpret the dimensions, `Colorant` `eltype`s are not
    allowed. As `eltype` we suggest `Float32` or more precise. Something like `N0f8` is
    probably not accurate enough, but will also fail as we cannot create `Normed` instances,
    due to a string interpolation in the message of an error potentially thrown in the
    constructor.
- `N_dssims_dM`: Idem.
- `N_dssims_dP`: Idem.
- `should_zero`: To get the correct (D)SSIM value we need `ssims` to start with values of
    `0`. When set to `true` we will perform this zeroing inside the method. Otherwise it is
    the responsibility of the caller to have done this in advance.
    Only relevant when `!skip_ssim`.
- `skip_ssim`: Whether to actually compute and store the (D)SSIM values in `ssims`.
- `skip_backward`: Whether to actually compute and store the gradients in `dL_dimgs1`.
- `use_dssim`: Whether to compute the SSIM or the DSSIM.
- `divide_by_two_in_dssim`: When set to `true` we use DSSIM = (1 - SSIM) / 2, otherwise
    DSSIM = 1 - SSIM.

# Returns
- The (potentially newly allocated) `ssims` and `dL_dimgs1`. When `imgs1` was not supplied
    in batch format, the first return will be a scalar (d)ssim.
"""
function _ssim_fwd_bwd!(
    ssims, dL_dimgs1, 
    imgs1, imgs2,
    N_dssims_dQ, N_dssims_dM, N_dssims_dP,
    should_zero, 
    skip_ssim_val::Val{skip_ssim}, skip_backward_val::Val{skip_backward}, 
    use_dssim, divide_by_two_in_dssim
) where {skip_backward, skip_ssim}

    cimgs1 = _canonicalize_image(imgs1)
    cimgs2 = _canonicalize_image(imgs2)
    dL_dcimgs1 = _canonicalize_image(dL_dimgs1)  # (possibly: nothing ↦ nothing)
    N_ssims = ssims  # (Just to make clear we still need to divide)

    size(cimgs1) == size(cimgs2) || 
        throw(DimensionMismatch(
            "The two image batches should have the same dimensions! \
             Received $(size(cimgs1)) and $(size(cimgs2))."))

    c, h, w, b = size(cimgs1)
    nb_pixel_components_per_image = c * h * w  # i.e. N
    T1 = eltype(cimgs1); T2 = eltype(cimgs2); TK = eltype(KERNEL)
    T = Tc_fwd = promote_type(T1, T2, TK)  
    # Type of the forward kernel's convolution output, and eltype of allocations we make
    # ourselves
    
    c_val = Val(c)
    single_image_pair_val = Val(b == 1)

    if !skip_ssim
        if isnothing(N_ssims)
            N_ssims = CUDA.zeros(T, b)
        elseif should_zero
            fill!(N_ssims, 0)
        end
        size(N_ssims) == (b,) || throw(DimensionMismatch(
            "Expected the (d)ssims to be a CuVector of length the batch size $b. \
             Received size $(size(N_ssims))."))
    end
 
    # Canonicalize, or allocate in canonical format the N_dssims_dX (or do nothing if we
    # don't need them)
    allocated_N_dssims_dXs = map((N_dssims_dQ, N_dssims_dM, N_dssims_dP)) do N_dssims_dX
        skip_backward && return (false, nothing)
        if isnothing(N_dssims_dX)
            allocated = true
            N_dssims_dX = CuArray{T}(undef, h, c, w, b)
        else
            allocated = false
            N_dssims_dX = _canonicalize_gradient_buffer(N_dssims_dX)
            size(N_dssims_dX) == (h, c, w, b) || throw(DimensionMismatch(
                "Expected all gradient buffers to have size ($h, $c, $w, $b) \
                (height, nb_channels, width, batch_size). Received size \
                $(size(N_dssims_dX))."))
        end
        return allocated, N_dssims_dX
    end  # NTuple{3, Tuple{Bool, CuArray}}
    N_dssims_dXs = last(zip(allocated_N_dssims_dXs...))  # NTuple{3, CuArray}


    # Forward pass
    nb_tiles = cld.((h, w), (TILE_HEIGHT, TILE_WIDTH))       # 2D
    nb_blocks = cld(prod(nb_tiles), MAX_NB_TILES_PER_BLOCK)  # 1D
    shmem_bytes = c * 
        (SHMEM_PRECONV_HEIGHT * SHMEM_PRECONV_WIDTH * (sizeof(T1) + sizeof(T2)) + 
         SHMEM_CONV_U_HEIGHT * SHMEM_CONV_U_WIDTH * sizeof(Tc_fwd))
    # shmem_img1, shmem_img2, shmem_conv_u

    max_static_shared_memory = attribute(
        device(), CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)  # also in bytes
    if shmem_bytes <= max_static_shared_memory
        @cuda threads=NB_THREADS blocks=nb_blocks _ssim_kernel_fwd!(
            N_ssims, 
            N_dssims_dXs...,
            cimgs1, cimgs2,
            c_val, single_image_pair_val,
            Val(false),  # static shared memory
            skip_ssim_val, skip_backward_val
        )
    else
        @warn("Requested more shared memory for the forward SSIM pass than is available by \
            default ($shmem_bytes > $max_static_shared_memory bytes). Switching to dynamic \
            shared memory and increasing the limits. This may decrease performance.",
            maxlog = 1)
        kernel = @cuda launch=false _ssim_kernel_fwd!(
            N_ssims,
            N_dssims_dXs...,
            cimgs1, cimgs2,
            c_val, single_image_pair_val,
            Val(true),
            skip_ssim_val, skip_backward_val
        )
        attributes(kernel.fun)[CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] =
            shmem_bytes
        kernel(N_ssims, 
            N_dssims_dXs...,
            cimgs1, cimgs2,
            c_val, single_image_pair_val,
            Val(true),
            skip_ssim_val, skip_backward_val;
            threads=NB_THREADS, blocks=nb_blocks, shmem=shmem_bytes)
    end
    if !skip_ssim
        isnothing(ssims) && ( N_ssims = _maybe_unpack_and_free!(N_ssims, imgs1) )
        # If ssims was provided, don't do any unpacking and definitely do not free it!

        ssims = _divide_by_N_and_maybe_d!(
            N_ssims, nb_pixel_components_per_image,
            use_dssim, divide_by_two_in_dssim)
    end


    # Backward pass
    if !skip_backward
        isnothing(dL_dcimgs1) && ( dL_dcimgs1 = CuArray{T}(undef, c, h, w, b) )

        TdX = eltype(N_dssims_dXs[1])
        Tc_bwd = promote_type(TK, TdX)  # used in convolution('s shared memory array)

        shmem_bytes = c * 
            (SHMEM_PRECONV_HEIGHT * SHMEM_PRECONV_WIDTH * sizeof(TdX) + 
             SHMEM_CONV_U_HEIGHT * SHMEM_CONV_U_WIDTH * sizeof(Tc_bwd))
        # shmem_N_dL_dX and shmem_conv_u

        if shmem_bytes <= max_static_shared_memory
            @cuda threads=NB_THREADS blocks=nb_blocks _ssim_kernel_bwd!(
                dL_dcimgs1, 
                N_dssims_dXs...,
                cimgs1, cimgs2, 
                c_val, single_image_pair_val,
                Val(false),
                use_dssim, divide_by_two_in_dssim
            )
        else
            @warn "Requested more shared memory for the backward SSIM pass than is \
            available by default ($shmem_bytes > $max_static_shared_memory bytes). \
            Switching to dynamic shared memory and increasing the limits. This may \
            decrease performance." maxlog=1
            kernel = @cuda launch=false _ssim_kernel_bwd!(
                dL_dcimgs1, 
                N_dssims_dXs...,
                cimgs1, cimgs2, 
                c_val, single_image_pair_val,
                Val(true),
                use_dssim, divide_by_two_in_dssim
            )
            attributes(kernel.fun)[CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] =
                shmem_bytes
            kernel(
                dL_dcimgs1, 
                N_dssims_dXs...,
                cimgs1, cimgs2, 
                c_val, single_image_pair_val,
                Val(true),
                use_dssim, divide_by_two_in_dssim;
                threads=NB_THREADS, blocks=nb_blocks, shmem=shmem_bytes
            )
        end

        if isnothing(dL_dimgs1) 
            dL_dimgs1 = _decanonicalize_gradient_image(dL_dcimgs1, imgs1)
        end
        # When dL_dimgs1 was provided (in whatever format), just return it (in that format)
    end


    # Cleanup
    foreach(allocated_N_dssims_dXs) do (allocated, N_dssims_dX)
        allocated && unsafe_free!(N_dssims_dX)
    end

    return ssims, dL_dimgs1
end


"""
    _divide_by_N_and_maybe_d!(
        N_ssims,
        N,
        use_dssim, divide_by_two_in_dssim
    )

Divide `N_ssims` by `N` to obtain `ssims`. Depending on the last two flag arguments we might
also convert this to `dssims`.

When `N_ssims` is a scalar, we return a new scalar. When it is a `CuArray`, we modify
`N_ssims` in-place and return it.
"""
function _divide_by_N_and_maybe_d! end

function _divide_by_N_and_maybe_d!(
    N_ssim::AbstractFloat,
    N,
    use_dssim, divide_by_two_in_dssim
)
    ssim = N_ssim / N
    if use_dssim
        if divide_by_two_in_dssim
            return (1 - ssim) / 2
        else
            return 1 - ssim
        end
    else
        return ssim
    end
end

function _divide_by_N_and_maybe_d!(
    N_ssims,
    N,
    use_dssim, divide_by_two_in_dssim
)
    # Combine all operations into a single kernel launch
    if use_dssim
        if divide_by_two_in_dssim
            N_ssims .= (1 .- (N_ssims ./ N)) ./ 2
        else
            N_ssims .= 1 .- (N_ssims ./ N)
        end
    else
        N_ssims ./= N
    end
    return N_ssims
end