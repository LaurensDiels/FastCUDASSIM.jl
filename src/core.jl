# Contains the core methods for the computation of the (D)SSIM and its gradients,
# but not in a particularly user-friendly form.

const WARPSIZE = 32i32  # == warpsize()

# Each image in an image batch is handled separately, reusing internal buffers (like shared
# memory). We split each image into non-overlapping tiles. Each tile is handled by a single
# block, but blocks might handle multiple (strided) tiles.
const TILE_HEIGHT = 32i32  
# ( == WARPSIZE for better memory coalescing )
const TILE_WIDTH = 8i32
# Obtained by testing many configurations on an RTX 3070.
# Also fits comfortably into standard shared memory (48 KiB) for Float32 RGB images.
const NB_PXS_PER_TILE = TILE_HEIGHT * TILE_WIDTH

const MAX_NB_TILES_PER_BLOCK = 1i32
# (So actually every block only deals with one tile per image in a batch.)
# Use a const MAX_ upper bound, so that the compiler can unroll the related loops.

# In our implementation each thread corresponds to one or more pixel locations (v, u) in
# every tile for our block. We use v and u instead of y and x to avoid confusion with the
# threadIdx dimensions (though we keep threadIdx linear to be able to deal with edges
# more efficiently).
const NB_THREADS = NB_PXS_PER_TILE  
# Can also be set to something different, like cld(NB_PXS_PER_TILE, 2)
const MAX_PX_PER_THREAD = cld(NB_PXS_PER_TILE, NB_THREADS)
# For certain choices of TILE_WIDTH, TILE_HEIGHT and NB_THREADS a thread might need to deal
# with multiple pixels in a tile. (Not relevant for the chosen constants: 1)
const MAX_PX_PER_THREAD_I64 = Int(MAX_PX_PER_THREAD)
# Need Int64 (not Int32) to use in SVector size

# Each thread has to compute the value in the SSIM map at its locations (v, u). This
# requires the 2D convolution of the input images at such a location, for which we need 
# [v - RADIUS : v + RADIUS, u - RADIUS : u + RADIUS].
# On the thread-block level, this means we need to load in a tile of the input pixels at our
# thread locations, together with a margin of RADIUS at all sides.

# We will compute the separable 2D convolutions via 2 1D convolutions (first u, then v), in
# shared memory.
const STD = 1.5f0    # Gaussian standard deviation
const RADIUS = 5i32  # Radius of the Gaussian kernel
const KERNEL_UNNORMALIZED = @SVector [
    exp(-0.5f0 * (x / STD)^2) for x = -RADIUS:RADIUS
]  
# (Also with the prefactor of 1 / (STD * sqrt(2 * Float32(π))), this does not sum to 1.)
const KERNEL = KERNEL_UNNORMALIZED ./ sum(KERNEL_UNNORMALIZED)

# Stabilization constants in the SSIM map formula denominator.
const C1 = 0.01f0 ^ 2
const C2 = 0.03f0 ^ 2
# This assumes intensities lie between 0 and 1.

# Since each block computes complete tiles in the SSIM map, after the
# convolutions we need a tile of the block size: TILE_HEIGHT x TILE_WIDTH.
const SHMEM_UV_CONV_HEIGHT = TILE_HEIGHT
const SHMEM_UV_CONV_WIDTH = TILE_WIDTH
# This is the block-level output of the v-convolution (after the u-convolution), which means
# the input must have size
const SHMEM_U_CONV_HEIGHT = SHMEM_UV_CONV_HEIGHT + 2 * RADIUS
const SHMEM_U_CONV_WIDTH = SHMEM_UV_CONV_WIDTH
# This in turn is the block-level output of the u-convolution, meaning its input, i.e.
# the required size before any convolution is
const SHMEM_PRECONV_WIDTH = SHMEM_U_CONV_WIDTH + 2 * RADIUS
const SHMEM_PRECONV_HEIGHT = SHMEM_U_CONV_HEIGHT
# We need to load in enlarged tiles of this size into shared memory, in order to produce
# the convolution output tile.
const NB_PXS_TO_LOAD_TO_SHMEM = SHMEM_PRECONV_HEIGHT * SHMEM_PRECONV_WIDTH
const MAX_SHMEM_LOAD_PX_PER_THREAD = cld(
    SHMEM_PRECONV_HEIGHT * SHMEM_PRECONV_WIDTH, NB_THREADS)
# (Relevant for the chosen constants.)

const NB_U_CONV_PXS = SHMEM_U_CONV_HEIGHT * SHMEM_U_CONV_WIDTH
const MAX_U_CONV_PX_PER_THREAD = cld(SHMEM_U_CONV_WIDTH * SHMEM_U_CONV_HEIGHT, NB_THREADS)
# (Relevant for the chosen constants.)

# Both for the forward and the backward pass, we need multiple 2D convolutions, which
# we will group together.
const NB_FWD_CONVS = 5i32  # img1, img1 .^ 2, img2, img2 .^ 2, img1 .* img2
const NB_FWD_CONVS_I64 = Int(NB_FWD_CONVS)
const NB_BWD_CONVS = 3i32  # dL_dQ, dL_dM, dL_dP (combined into dL_dQMP)
const NB_BWD_CONVS_I64 = Int(NB_BWD_CONVS)


"""
    _ssim_kernel_fwd!(
        N_ssims, 
        N_dssims_dQMP, 
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
We iterate over the images in a batch in the supplied order. Even though this does not
respect the memory order of the images, where the channels are grouped together, we
similarly loop over the color channels. Because all channels are treated independently,
this makes the convolutions and the entire method more efficient, even though the memory
loads from global memory might be a bit slower (though not too much as long as `nb_channels`
is relatively small). Additionally, this means we can reuse shared memory accross both
batch and channel iterations.

# Arguments
- `N_ssims`: A `CuVector` (or similar) of `length` the batch size. It will store for each
    input image the output sum of its SSIM map. To obtain the actual SSIMs, this still needs
    to be divided by the number `N` of pixel components per image 
    (`N = prod(size(imgs1)[1:3])`). We do not yet perform this division inside of this
    kernel for reasons of numerical stability.
    Only relevant when `!skip_ssim`.
    !!! The initial values must be zeros. !!!
- `N_dssims_dQMP`: Intermediate gradients to store for the backward pass, multiplied by `N`.
    See the mathematics section in the documentation for more information. For memory
    coalescing the `size` is `(h, 3, c, w, b)`, where `c` is the number of color channels
    (possibly 1) in the input images, `h` and `w` are the height and width of the images,
    and `b` (possibly 1) is the batch size.
    Only relevant when `!skip_backward`.
- `imgs1`: The first input image batch in normalized floating point format, and of `size`
    `(c, h, w, b)`. The eventual backward pass gradients are computed with respect to these
    images.
- `imgs2`: The second input image batch with the same format as the first. This is
    considered a constant for the gradients.
- `nb_channels`: The number of color channels `c`. Statically known for possible
    optimizations by the compiler (like loop unrolling). 
- `single_image_pair`: Whether the batch size is 1. Statically known for allowing the
    compiler to elide the batch-loop.
- `use_dynamic_shmem`: Whether to use `CuStaticSharedArray`s (`false`) or
    `CuDynamicSharedArray`s (`true`) for shared memory. In the latter case the `@cuda` call
    also needs to include the amount of requested shared memory, which can be obtained via
    [`_get_shmem_fwd_bytes`](@ref). When using dynamic shared memory, we can request more
    than with static memory. On the other hand, it is slightly slower (when we do not exceed
    the static bound).
- `skip_ssim`: Whether to output `N_ssims` (in-place). When we are only interested in the
    gradients and not the SSIMs themselves, this can be skipped.
- `skip_backward`: Whether to fill the gradient buffer `N_dssims_dQMP`. When we are only
    interested in the SSIMs themselves, and not the gradients, this can be skipped.
"""
function _ssim_kernel_fwd!(
    N_ssims, 
    N_dssims_dQMP, 
    imgs1, imgs2,
    ::Val{nb_channels}, ::Val{single_image_pair},
    use_dynamic_shmem_val,
    ::Val{skip_ssim}, ::Val{skip_backward}
) where {nb_channels, single_image_pair, skip_ssim, skip_backward}

    _, img_height, img_width, batch_size = size(imgs1)
    single_image_pair && ( batch_size = 1 )  # batch_size becomes statically known
    T1 = eltype(imgs1); T2 = eltype(imgs2); TK = eltype(KERNEL)
    Tc = promote_type(T1, T2, TK)  # used for convolution
    Ts = eltype(N_ssims)           # used for output ssims 

    shmem_img1, shmem_img2, shmem_u_conv = _get_shmem_fwd(
        T1, T2, Tc, use_dynamic_shmem_val)
    # (Needs a function barrier for type inference on shmem_img1 etc.)
    # These shared memory arrays are reused in subsequent batch, tile and channel
    # iterations.

    thread_idx_in_block = threadIdx().x  # 1D index
    # blockDim().x == NB_THREADS
    thread_idx_in_warp0 = (thread_idx_in_block - 1i32) & (WARPSIZE - 1i32)
    # mod1(thread_idx_in_block, WARPSIZE), but more efficient and zero-indexed (i.e. mod)

    nb_tiles_v = cld(img_height, TILE_HEIGHT)
    nb_tiles_u = cld(img_width, TILE_WIDTH)
    nb_tiles = nb_tiles_v * nb_tiles_u
    tile_idx_to_2D = CartesianIndices((nb_tiles_v, nb_tiles_u))
    # Converts a 1D tile index to 2D

    block_idx = blockIdx().x  # 1D index of our block
    nb_blocks = gridDim().x   # Number of requested blocks

    batch_idx = 1i32  # The image index in imgs1 and imgs2 we are currently considering
    # Conceptually within each iteration we only consider 
    #   img1 = imgs1[:, :, :, batch_idx]
    # and similar for img2
    @inbounds while batch_idx <= batch_size
    # (Every memory access in the entire function is inbounds)

        if !skip_ssim
            thread_sum_ssim::Ts = 0
            # sum over all ssim map pixels our thread deals with, for the current image
            # Note: not
            #   thread_sum_ssim = Ts(0)
            # as then after
            #   thread_sum_ssim += s
            # the type might change to the promotion type with s.
            # Because out of bounds threads do not have an s value to add, their
            # thread_sum_ssim type remains of type Ts. Having different types causes
            # a freeze in the block reduction below.
        end

        tile_idx = block_idx  # 1D index of the current tile our block is working on
        block_tile_counter = 1i32
        while block_tile_counter <= MAX_NB_TILES_PER_BLOCK
            tile_idx <= nb_tiles || break
            # Potentially faster than   while tile_idx <= nb_tiles   as the number of
            # iterations is statically known.

            # Determine tiles boundaries: (current) actual output tile + necessary
            # boundaries for convolution
            tile_v, tile_u = Tuple(tile_idx_to_2D[tile_idx])
            uv_conv_tile_start_v0 = (tile_v - 1i32) * TILE_HEIGHT  # 0-indexed
            uv_conv_tile_start_u0 = (tile_u - 1i32) * TILE_WIDTH   # a.k.a. tile_start_u0
            # Cf. SHMEM_U_CONV_HEIGHT and _WIDTH:
            u_conv_tile_start_v0 = uv_conv_tile_start_v0 - RADIUS
            u_conv_tile_start_u0 = uv_conv_tile_start_u0
            preconv_tile_start_v0 = u_conv_tile_start_v0
            preconv_tile_start_u0 = u_conv_tile_start_u0 - RADIUS
            
            ch = 1i32
            while ch <= nb_channels
                # Deal with img1[ch, :, :] == imgs1[ch, :, :, batch_idx]

                # -----------------------------------------------------
                # Load tile with boundary into shared memory
                #
                load_idx = thread_idx_in_block
                # local 1D index within enlarged tile to load
                load_idx_to_2D = CartesianIndices(
                    (SHMEM_PRECONV_HEIGHT, SHMEM_PRECONV_WIDTH))
                # Maps the local 1D index to 2D local pixel coordinates
                # (Note that length(load_idx_to_2D) == NB_PXS_TO_LOAD_TO_SHMEM)
                thread_pixel_counter = 1i32
                while thread_pixel_counter <= MAX_SHMEM_LOAD_PX_PER_THREAD
                    load_idx <= NB_PXS_TO_LOAD_TO_SHMEM || break
                    load_v, load_u = Tuple(load_idx_to_2D[load_idx])
                    # local 2D index, within the enlarged tile
                    img_v = load_v + preconv_tile_start_v0  # global 2D index,
                    img_u = load_u + preconv_tile_start_u0  # within the input images
                    if 1i32 <= img_v <= img_height && 1i32 <= img_u <= img_width
                        shmem_img1[load_v, load_u] = imgs1[ch, img_v, img_u, batch_idx]
                        shmem_img2[load_v, load_u] = imgs2[ch, img_v, img_u, batch_idx]
                    else
                        shmem_img1[load_v, load_u] = zero(T1)  # zero-padding
                        shmem_img2[load_v, load_u] = zero(T2)  # (Could also just write 0)
                    end 
                    load_idx += NB_THREADS
                    thread_pixel_counter += 1i32
                end
                sync_threads()
                #
                # shared memory load
                # -----------------------------------------------------

                # -----------------------------------------------------
                # u-convolve (the shared memory versions of)
                #   img1, img1 .^ 2, img1 .* img2, img2 and img2 .^ 2
                # current image tiles and color channel with the KERNEL
                #
                u_conv_idx = thread_idx_in_block  
                # 1D, local within the output u-convolved padded tile
                u_conv_idx_to_2D = CartesianIndices(
                    (SHMEM_U_CONV_HEIGHT, SHMEM_U_CONV_WIDTH))
                # (Note that length(u_conv_idx_to_2D) == NB_U_CONV_PXS)
                thread_pixel_counter = 1i32
                while thread_pixel_counter <= MAX_U_CONV_PX_PER_THREAD
                    u_conv_idx <= NB_U_CONV_PXS || break
                    out_v, out_u = Tuple(u_conv_idx_to_2D[u_conv_idx])
                    
                    # convolution (cross-correlation) output values
                    u_convd_img1_val     ::Tc = 0  # first moment
                    u_convd_img1_sq_val  ::Tc = 0  # second moment
                    u_convd_img2_val     ::Tc = 0
                    u_convd_img2_sq_val  ::Tc = 0
                    u_convd_img1_img2_val::Tc = 0
                    i = 0i32
                    while i <= 2i32 * RADIUS
                        img1_val = shmem_img1[out_v, out_u + i]
                        # Note that e.g.
                        #   shmem_img1[v, u] == img1[ch, out_v - RADIUS, out_u - RADIUS] 
                        # where inbounds
                        img2_val = shmem_img2[out_v, out_u + i]
                        ker_val = KERNEL[i + 1i32]

                        img1_ker_val = img1_val * ker_val
                        img2_ker_val = img2_val * ker_val
                        u_convd_img1_val      += img1_ker_val             
                        u_convd_img1_sq_val   += img1_val * img1_ker_val
                        u_convd_img2_val      += img2_ker_val
                        u_convd_img2_sq_val   += img2_val * img2_ker_val
                        u_convd_img1_img2_val += img1_val * img2_ker_val
                        i += 1i32
                    end
                    
                    shmem_u_conv[out_v, 1, out_u] = u_convd_img1_val
                    shmem_u_conv[out_v, 2, out_u] = u_convd_img1_sq_val
                    shmem_u_conv[out_v, 3, out_u] = u_convd_img2_val
                    shmem_u_conv[out_v, 4, out_u] = u_convd_img2_sq_val
                    shmem_u_conv[out_v, 5, out_u] = u_convd_img1_img2_val
                    # (With 5 == NB_FWD_CONVS_I64)

                    u_conv_idx += NB_THREADS
                    thread_pixel_counter += 1i32
                end
                sync_threads()
                #
                # u-convolution
                # -----------------------------------------------------
            
                # -----------------------------------------------------
                # v-convolution and SSIM map value calculation
                #
                thread_convolved_outputs = MVector{NB_FWD_CONVS_I64, Tc}(undef)
                # Will contain the (uv-)convolved versions of img1, img1.^2, etc.
                output_idx = thread_idx_in_block
                # 1D, local within the (uv-convolved / SSIM map) tile
                output_idx_to_2D = CartesianIndices(
                    (SHMEM_UV_CONV_HEIGHT, SHMEM_UV_CONV_WIDTH))
                # (Note that length(output_idx_to_2D) == NB_PXS_PER_TILE)

                thread_pixel_counter = 1i32 
                while thread_pixel_counter <= MAX_PX_PER_THREAD
                    output_idx <= NB_PXS_PER_TILE || break
                    local_v, local_u = Tuple(output_idx_to_2D[output_idx])
                    global_v = uv_conv_tile_start_v0 + local_v  # 2D index within img1, img2
                    global_u = uv_conv_tile_start_u0 + local_u  # and within the SSIM map
                    if global_u <= img_width && global_v <= img_height
                        # (1 <=  is not necessary, as uv_conv_tile_start_v0, _u0 >= 0)

                        # For the v-convolution, we exploit the fact that KERNEL is
                        # symmetric (in contrast to the u-convolution we cannot reuse
                        # intermediate variables)
                        # Start in the middle as this is the only non-duplicated KERNEL
                        # value (and hence is a good point to initialize
                        # thread_convolved_outputs)
                        central_ker_val = KERNEL[RADIUS + 1i32]
                        conv_idx = 1i32  # index for img1 ([1]), img1.^2 ([2]), ...
                        while conv_idx <= NB_FWD_CONVS
                            thread_convolved_outputs[conv_idx] = 
                                shmem_u_conv[local_v + RADIUS, conv_idx, local_u] * 
                                  central_ker_val

                            # Next move from the edges towards the center.
                            # This memory order should be better for the i_up index.
                            # For i_down hopefully the values are cached from the middle
                            # load above.
                            i_up = 0i32 
                            while i_up < RADIUS
                                i_down = 2i32 * RADIUS - i_up
                                ker_val = KERNEL[i_up + 1i32]  # == KERNEL[i_down + 1i32]
                                
                                thread_convolved_outputs[conv_idx] += 
                                    (shmem_u_conv[local_v + i_up,   conv_idx, local_u] + 
                                     shmem_u_conv[local_v + i_down, conv_idx, local_u]
                                    ) * ker_val
                                i_up += 1i32
                            end
                            conv_idx += 1i32
                        end

                        # Now compute the SSIM map value s in our pixel
                        # Notation cf. mathematics in documentation
                        m, q, m_R, q_R, p = thread_convolved_outputs

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
                            thread_sum_ssim += s
                        end
                        if !skip_backward
                            N_dssims_dQMP[global_v, 1, ch, global_u, batch_idx] = -s / t
                            N_dssims_dQMP[global_v, 2, ch, global_u, batch_idx] = 
                                2 * zt_inv * ((y - x) * m_R + s * (z - t) * m) 
                            N_dssims_dQMP[global_v, 3, ch, global_u, batch_idx] = 
                                2 * x * zt_inv
                        end 
                        
                    end  # inbounds check
                        
                    output_idx += NB_THREADS
                    thread_pixel_counter += 1i32
                end  # thread_pixel_counter loop
                
                # (We don't need to synchronize as shmem_u_conv does not get overwritten
                # until after the next shared memory load for imgs1 and imgs2, after which
                # there already is a sync_threads)

                #
                # v-convolution and SSIM map value
                # -----------------------------------------------------

                ch += 1i32
            end  # channel loop
        
            tile_idx += nb_blocks
            block_tile_counter += 1i32
        end  # tiles loop

        if !skip_ssim
            # We already reduced (via +) the ssim map values for all pixel components our
            # thread deals with. Now further reduce across threads.

            # First reduce at the warp level via shuffling:
            warp_ssim_sum = thread_sum_ssim
            offset = 1i32
            while offset < WARPSIZE
                warp_ssim_sum += shfl_down_sync(0xffffffff, warp_ssim_sum, offset)
                offset <<= 1
            end
            # (Note that also threads outside of the image boundaries need to
            # contribute to the reduction. Since thread_sum_ssim is initialized at 0,
            # this is not an issue.)

            # The warp leaders now contain the sum of the values of all threads in the warp.
            # Now globally reduce via atomics:
            if thread_idx_in_warp0 == 0i32
                CUDA.@atomic N_ssims[batch_idx] += warp_ssim_sum
                # If at this point we divide by the number of pixels components per image,
                # then e.g.
                #   ssim(img1, img1) = 0.9999624f0 < 1f0
                # for two full HD zero images, because of accumulating rounding errors. 
                # Therefore, we divide as late as possible: after the kernel.
            end
            # Due to clever compiler optimizations regarding atomics slightly faster than
            # performing block-level reduction and atomically reducing globally,
            # and faster than outputting block- or warp-level reduced SSIM maps and
            # using sum afterwards.    
        end

        batch_idx += 1i32
    end  # batch loop

    return
end


"""
    _get_shmem_fwd(
        T1, T2, Tc, 
        ::Val{use_dynamic_shmem}
    ) where {use_dynamic_shmem}

Return the shared memory arrays required for `_ssim_kernel_fwd!`.

# Arguments
- `T1`: The `eltype` of the first image batch.
- `T2`: The `eltype` of the second image batch.
- `Tc`: The `eltype` after convolution with the KERNEL.
- `use_dynamic_shmem`: Whether to use `CuStaticSharedArray` (`false`) or 
    `CuDynamicSharedArray` (`true`).

# Returns
Three arrays in shared memory:
1. `shmem_img1`: To store our block's current tile in the current image from the first 
    image batch. It will be the input for the convolutions, so we enlarge this tile at the
    boundaries. It satisfies `eltype(shmem_img1) == T1 && \
    size(shmem_img1) == (SHMEM_PRECONV_HEIGHT, SHMEM_PRECONV_WIDTH)`.
1. `shmem_img2`: Similar, for the second image batch. It has the same size as `shmem_img1`,
    but `eltype` `T2`.
1. `shmem_u_conv`: To store the u-convolution outputs. Its `eltype` is `Tc` and its `size`
    `(SHMEM_U_CONV_HEIGHT, SHMEM_U_CONV_WIDTH)`.
"""
function _get_shmem_fwd end

@inline function _get_shmem_fwd(
    T1, T2, Tc, 
    ::Val{false}
)
    shmem_img1 = CuStaticSharedArray(
        T1,
        (SHMEM_PRECONV_HEIGHT, SHMEM_PRECONV_WIDTH)
    )  
    shmem_img2 = CuStaticSharedArray(T2, size(shmem_img1))
    shmem_u_conv = CuStaticSharedArray(
        Tc, 
        (SHMEM_U_CONV_HEIGHT, NB_FWD_CONVS, SHMEM_U_CONV_WIDTH), 
    )
    return shmem_img1, shmem_img2, shmem_u_conv
end

@inline function _get_shmem_fwd(
    T1, T2, Tc, 
    ::Val{true}
)
    shmem_img1 = CuDynamicSharedArray(
        T1,
        (SHMEM_PRECONV_HEIGHT, SHMEM_PRECONV_WIDTH)
    )  
    offset = sizeof(shmem_img1)
    shmem_img2 = CuDynamicSharedArray(T2, size(shmem_img1), offset)
    offset += sizeof(shmem_img2)
    shmem_u_conv = CuDynamicSharedArray(
        Tc, 
        (SHMEM_U_CONV_HEIGHT, NB_FWD_CONVS, SHMEM_U_CONV_WIDTH),
        offset
    )
    return shmem_img1, shmem_img2, shmem_u_conv
end

"""
    _get_shmem_fwd_bytes(T1, T2, Tc)

Return the amount of shared memory in bytes requested by [`_get_shmem_fwd`](@ref).
"""
function _get_shmem_fwd_bytes(T1, T2, Tc)
    return (sizeof(T1) + sizeof(T2)) * SHMEM_PRECONV_HEIGHT * SHMEM_PRECONV_WIDTH +
            sizeof(Tc) * (SHMEM_U_CONV_HEIGHT * NB_FWD_CONVS * SHMEM_U_CONV_WIDTH)
end



"""
    _ssim_kernel_bwd!(
        dL_dimgs1, 
        N_dssims_dQMP, 
        imgs1, imgs2,
        ::Val{nb_channels}, ::Val{single_image_pair},
        ::Val{use_dynamic_shmem},
        use_dssim, divide_by_two_in_dssim
    ) where {nb_channels, single_image_pair, use_dynamic_shmem}
    -> Nothing

Perform the backward pass of the (D)SSIM with respect to the first image batch, given the
relevant intermediate gradients from the forward pass.

We use the same grid-block-thread structure as in the forward pass, though we now keep the
color channels together by iterating over them in more inner loops.

# Arguments
- `dL_dimgs1`: The output gradient 'image' batch of the same format as `imgs1`, that is,
    normalized floating point format with `size` `(c, h, w, b)` where `c` is the number of
    color channels (possibly 1), `h` is the image height, `w` the width, and `b` is the
    batch size (possibly 1). `L` is either the SSIM or the DSSIM depending on `use_dssim`.
- `N_dssims_dQMP`: Intermediate gradients from the forward pass.
    The `size` of this `CuArray` is (h, 3, c, w, b).
    (Note that the 'd' here is for _d_ifferentiation in Leibniz notation (dy/dx), not for
    _d_ssim.)
- `imgs1`: The first input image batch in normalized floating point format, and of `size`
    `(c, h, w, b)`. The gradients are computed with respect to this batch.
- `imgs2`: The second input image batch with the same format as the first. This is
    considered a constant for the gradients.
- `nb_channels`: The number of color channels `c`. Statically known for possible
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
    N_dssims_dQMP, 
    imgs1, imgs2, 
    nb_channels_val::Val{nb_channels}, ::Val{single_image_pair},
    use_dynamic_shmem_val,
    use_dssim, divide_by_two_in_dssim
) where {nb_channels, single_image_pair}

    # The general structure is very similar to in the forward pass.
    
    _, img_height, img_width, batch_size = size(imgs1)
    single_image_pair && ( batch_size = 1 )
    nb_pixel_components_per_image = nb_channels * img_height * img_width  # a.k.a. N
    Td1 = eltype(dL_dimgs1); TdQMP = eltype(N_dssims_dQMP); TK = eltype(KERNEL)
    Tc = promote_type(TK, TdQMP)  # used in convolution

    if use_dssim && divide_by_two_in_dssim
        N_inv = Td1(0.5) / nb_pixel_components_per_image  
        # (So technically no longer N^(-1))
    else
        N_inv = Td1(1) / nb_pixel_components_per_image
    end

    shmem_N_dL_dQMP, shmem_u_conv = _get_shmem_bwd(
        TdQMP, Tc, nb_channels_val, use_dynamic_shmem_val)
    # (Function barrier needed for type inference)

    thread_idx_in_block = threadIdx().x  # 1D index
    # blockDim().x == NB_THREADS

    nb_tiles_v = cld(img_height, TILE_HEIGHT)
    nb_tiles_u = cld(img_width, TILE_WIDTH)
    nb_tiles = nb_tiles_v * nb_tiles_u
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
            uv_conv_tile_start_v0 = (tile_v - 1i32) * TILE_HEIGHT  # 0-indexed
            uv_conv_tile_start_u0 = (tile_u - 1i32) * TILE_WIDTH   # a.k.a. tile_start_v0
            # Cf. SHMEM_U_CONV_BLOCK_HEIGHT and _WIDTH
            u_conv_tile_start_v0 = uv_conv_tile_start_v0 - RADIUS
            u_conv_tile_start_u0 = uv_conv_tile_start_u0
            preconv_tile_start_v0 = u_conv_tile_start_v0
            preconv_tile_start_u0 = u_conv_tile_start_u0 - RADIUS

            # -----------------------------------------------------
            # Load tile with boundary into shared memory
            #
            load_idx = thread_idx_in_block 
            # local 1D index within enlarged tile to load
            load_idx_to_2D = CartesianIndices(
                (SHMEM_PRECONV_HEIGHT, SHMEM_PRECONV_WIDTH))
            thread_pixel_counter = 1i32
            while thread_pixel_counter <= MAX_SHMEM_LOAD_PX_PER_THREAD
                load_idx <= NB_PXS_TO_LOAD_TO_SHMEM || break
                load_v, load_u = Tuple(load_idx_to_2D[load_idx])
                # local 2D index
                img_u = load_u + preconv_tile_start_u0  # global 2D index
                img_v = load_v + preconv_tile_start_v0
                ch = 1i32
                if 1i32 <= img_u <= img_width && 1i32 <= img_v <= img_height
                    while ch <= nb_channels
                        conv_idx = 1i32
                        # We need to perform convolutions of the three gradient 'images'
                        # in N_dssims_dQMP. This index then refers to Q ([1]), M ([2]) or
                        # P ([3]).
                        while conv_idx <= NB_BWD_CONVS
                            val = N_dssims_dQMP[img_v, conv_idx, ch, img_u, batch_idx]
                            shmem_N_dL_dQMP[load_v, conv_idx, ch, load_u] = ifelse(use_dssim, -val, val)
                            conv_idx += 1i32
                        end
                        ch += 1i32
                    end
                else
                    while ch <= nb_channels
                    conv_idx = 1i32
                        while conv_idx <= NB_BWD_CONVS
                            shmem_N_dL_dQMP[load_v, conv_idx, ch, load_u] = zero(TdQMP)
                            conv_idx += 1i32
                        end  
                        ch += 1i32
                    end  
                end
                load_idx += NB_THREADS
                thread_pixel_counter += 1i32
            end
            sync_threads()
            #
            # shared memory load
            # -----------------------------------------------------

            # -----------------------------------------------------
            # u-convolve with KERNEL
            #
            u_conv_idx = thread_idx_in_block
            # 1D, local within the u-convolved padded tile
            u_conv_idx_to_2D = CartesianIndices((SHMEM_U_CONV_HEIGHT, SHMEM_U_CONV_WIDTH))

            thread_pixel_counter = 1i32
            while thread_pixel_counter <= MAX_U_CONV_PX_PER_THREAD
                u_conv_idx <= NB_U_CONV_PXS || break
                out_v, out_u = Tuple(u_conv_idx_to_2D[u_conv_idx])
                # 2D local indices in the output u-convolved tile

                # Like in the forward pass's v-convolution, exploit KERNEL symmetry
                central_ker_val = KERNEL[RADIUS + 1i32]
                ch = 1i32
                while ch <= nb_channels
                    conv_idx = 1i32
                    while conv_idx <= NB_BWD_CONVS
                        shmem_u_conv[out_v, conv_idx, ch, out_u] = 
                            shmem_N_dL_dQMP[out_v, conv_idx, ch, out_u + RADIUS] * 
                              central_ker_val
                        conv_idx += 1i32
                    end
                    ch += 1i32
                end

                i_up = 0i32
                while i_up < RADIUS
                    i_down = 2i32 * RADIUS - i_up
                    ker_val = KERNEL[i_up + 1i32]  # == KERNEL[i_down + 1i32]
                    ch = 1i32
                    while ch <= nb_channels
                        conv_idx = 1i32
                        while conv_idx <= NB_BWD_CONVS
                            shmem_u_conv[out_v, conv_idx, ch, out_u] += 
                                (shmem_N_dL_dQMP[out_v, conv_idx, ch, out_u + i_up] + 
                                 shmem_N_dL_dQMP[out_v, conv_idx, ch, out_u + i_down]) * 
                                    ker_val
                            conv_idx += 1i32
                        end
                        ch += 1i32
                    end
                    i_up += 1i32
                end

                u_conv_idx += NB_THREADS
                thread_pixel_counter += 1i32
            end  # u_conv_idx / thread_pixel_counter loop
            sync_threads()
            #
            # u-convolution
            # -----------------------------------------------------

            # -----------------------------------------------------
            # v-convolution and dL_dimgs1 computation
            #
            output_idx = thread_idx_in_block  # 1D, local within the uv-convolved tile
            output_idx_to_2D = CartesianIndices((SHMEM_UV_CONV_HEIGHT, SHMEM_UV_CONV_WIDTH))

            thread_pixel_counter = 1i32 
            while thread_pixel_counter <= MAX_PX_PER_THREAD
                output_idx <= NB_PXS_PER_TILE || break
                local_v, local_u = Tuple(output_idx_to_2D[output_idx])
                # 2D index within shmem_u_conv
                global_v = uv_conv_tile_start_v0 + local_v 
                global_u = uv_conv_tile_start_u0 + local_u
                # 2D index within dL_dimgs1

                if global_u <= img_width && global_v <= img_height
                    ch = 1i32
                    while ch <= nb_channels
                        central_ker_val = KERNEL[RADIUS + 1i32]
                        # See the online documentation for the notation
                        N_dL_dJ          = shmem_u_conv[local_v + RADIUS, 1, ch, local_u] * 
                                                central_ker_val
                        N_dL_dimg1_via_M = shmem_u_conv[local_v + RADIUS, 2, ch, local_u] * 
                                                central_ker_val
                        N_dL_dH          = shmem_u_conv[local_v + RADIUS, 3, ch, local_u] * 
                                                central_ker_val

                        i_up = 0i32 
                        while i_up < RADIUS
                            i_down = 2i32 * RADIUS - i_up
                            ker_val = KERNEL[i_up + 1i32]
                            
                            N_dL_dJ          += 
                                (shmem_u_conv[local_v + i_up,   1, ch, local_u] + 
                                 shmem_u_conv[local_v + i_down, 1, ch, local_u]   ) * 
                                    ker_val
                            N_dL_dimg1_via_M += 
                                (shmem_u_conv[local_v + i_up,   2, ch, local_u] + 
                                 shmem_u_conv[local_v + i_down, 2, ch, local_u]   ) * 
                                    ker_val
                            N_dL_dH          += 
                                (shmem_u_conv[local_v + i_up,   3, ch, local_u] + 
                                 shmem_u_conv[local_v + i_down, 3, ch, local_u]   ) * 
                                    ker_val
                            i_up += 1i32
                        end

                        # Combine to get the full dL_dimgs1 values
                        N_dL_dimg1_via_Q =  # (or via J)
                            2 * imgs1[ch, global_v, global_u, batch_idx] * 
                                N_dL_dJ
                        N_dL_dimg1_via_P =  # (or via H)
                            imgs2[ch, global_v, global_u, batch_idx] * 
                                N_dL_dH

                        dL_dimgs1[ch, global_v, global_u, batch_idx] = 
                            (N_dL_dimg1_via_Q + N_dL_dimg1_via_M + N_dL_dimg1_via_P) * 
                                N_inv
                    
                        ch += 1i32
                    end  # channels loop
                end  # bounds check
                    
                output_idx += NB_THREADS
                thread_pixel_counter += 1i32
            end  # output_idx / thread_pixel_counter loop
        
            tile_idx += nb_blocks
            block_tile_counter += 1i32
        end  #  tiles loop

        batch_idx += 1i32
    end  # batch loop

    return
end


"""
    _get_shmem_bwd(
        TdQMP, Tc, 
        ::Val{nb_channels}, 
        ::Val{use_dynamic_shmem}
    ) where {nb_channels, use_dynamic_shmem}

Return the shared memory arrays required for `_ssim_kernel_bwd!`.

# Arguments
- `TdQMP`: The `eltype` of the gradients buffer `N_dssims_dQMP` etc.
- `Tc`: The `eltype` after convolution with the KERNEL.
- `nb_channels`: The number of color channels.
- `use_dynamic_shmem`: Whether to use `CuStaticSharedArray` (`false`) or 
    `CuDynamicSharedArray` (`true`).

# Returns
Two arrays in shared memory:
1. `shmem_N_dL_dQMP`: To store our block's current enlarged pre-convolution
    `N_dssims_dQMP` tile for the current image in the batch, possibly negated when computing
    the dssim. Its `eltype` is `TdQMP` and its `size` 
    `(SHMEM_U_CONV_HEIGHT, NB_BWD_CONVS, nb_channels, SHMEM_U_CONV_WIDTH)`.
1. `shmem_u_conv`: To store the u-convolution outputs. Its `eltype` is `Tc` and its `size`
    `(SHMEM_U_CONV_HEIGHT, NB_BWD_CONVS; nb_channels, SHMEM_U_CONV_WIDTH)`.
"""
function _get_shmem_bwd end

@inline function _get_shmem_bwd(
    TdQMP, Tc,
    ::Val{nb_channels},
    ::Val{false}
) where nb_channels
    shmem_N_dL_dQMP = CuStaticSharedArray(
        TdQMP,
        (SHMEM_PRECONV_HEIGHT, NB_BWD_CONVS, nb_channels, SHMEM_PRECONV_WIDTH)
    )  
    shmem_u_conv = CuStaticSharedArray(
        Tc, 
        (SHMEM_U_CONV_HEIGHT, NB_BWD_CONVS, nb_channels, SHMEM_U_CONV_WIDTH), 
    )
    return shmem_N_dL_dQMP, shmem_u_conv
end

@inline function _get_shmem_bwd(
    TdQMP, Tc, 
    ::Val{nb_channels},
    ::Val{true}
) where nb_channels
    shmem_N_dL_dQMP = CuDynamicSharedArray(
        TdQMP,
        (SHMEM_PRECONV_HEIGHT, NB_BWD_CONVS, nb_channels, SHMEM_PRECONV_WIDTH)
    )  
    shmem_u_conv = CuDynamicSharedArray(
        Tc, 
        (SHMEM_U_CONV_HEIGHT, NB_BWD_CONVS, nb_channels, SHMEM_U_CONV_WIDTH),
        sizeof(shmem_N_dL_dQMP)
    )
    return shmem_N_dL_dQMP, shmem_u_conv
end

"""
    _get_shmem_bwd_bytes(TdQMP, Tc)

Return the amount of shared memory in bytes requested by [`_get_shmem_bwd`](@ref).
"""
function _get_shmem_bwd_bytes(TdQMP, Tc, ::Val{nb_channels}) where nb_channels
    return nb_channels * 
        (sizeof(TdQMP) * SHMEM_PRECONV_HEIGHT * NB_BWD_CONVS * SHMEM_PRECONV_WIDTH
         + sizeof(Tc)  * SHMEM_U_CONV_HEIGHT  * NB_BWD_CONVS * SHMEM_U_CONV_WIDTH)
end



"""
    _ssim_fwd_bwd!(
        ssims, dL_dimgs1, 
        imgs1, imgs2,
        N_dssims_dQMP,
        should_zero, 
        ::Val{skip_ssim}, ::Val{skip_backward}, 
        use_dssim, divide_by_two_in_dssim
    ) where {skip_backward, skip_ssim} 

Compute the Structural (Dis)Similarity Index Measure ((D)SSIM) between two image batches
and/or its gradient with respect to the first batch.

Very general method. Users should normally use one of the easier exported wrapper functions,
like [`ssim`](@ref).

Outputs and buffers (`ssims`, `dL_dimgs1`, `N_dssims_dQMP`) which are necessary but set to
`nothing` will be allocated. In this case the `eltype` will be
`promote_type(eltype(imgs1), eltype(imgs2), $(eltype(KERNEL)))`.
Allocated buffers will be freed, while allocated outputs will be returned.

# Arguments
- `ssims`: The (output) `CuVector` of `length` the batch size, containing the (D)SSIM
    values.
- `dL_dimgs1`: The (output) gradient of `imgs1` with respect to the (D)SSIM value `L`. It
    uses the same normalised floating point format in
    `channels x height x width x batch size` memory order as `imgs1`. If not explicitly in
    this format, we will attempt to convert to it using [`_canonicalize_image`](@ref).
- `imgs1`: The first images batch in the aforementioned format. Also here, we will use
    [`_canonicalize_image`](@ref) in order to be able to handle e.g. a
    `CuMatrix{RGB{Float32}}`, which will be converted to a `CuArray{Float32, 4}` of size
    `(3, size(..., 1), size(..., 2), 1)`, using the same underlying memory.
- `imgs2`: The second image batch, in the same format.
- `N_dssims_dQMP`: A buffer of canonical `size`
    `height x $NB_BWD_CONVS_I64 x channels x width x batch size`. (The $NB_BWD_CONVS_I64
    here refers to the gradients with respect to `Q`, `M` and `P`. See the online
    documentation for their meaning.)
    We will attempt to use [`_canonicalize_gradient_buffer`](@ref) to convert to this
    `size`, if possible. As we will not move memory around, only reinterpret the dimensions,
    `Colorant` `eltype`s are not allowed. As `eltype` we suggest `Float32` or more precise.
    Something like `N0f8` is probably not accurate enough, but will also fail as we cannot
    create `Normed` instances, due to a string interpolation in the message of an error
    potentially thrown in the constructor.
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
    N_dssims_dQMP,
    should_zero, 
    skip_ssim_val::Val{skip_ssim}, skip_backward_val::Val{skip_backward}, 
    use_dssim, divide_by_two_in_dssim
) where {skip_backward, skip_ssim}

    cimgs1 = _canonicalize_image(imgs1)
    cimgs2 = _canonicalize_image(imgs2)
    N_ssims = ssims  # (Just to make clear we still need to divide)

    size(cimgs1) == size(cimgs2) || 
        throw(DimensionMismatch(
            "The two image batches should have the same dimensions! \
             Received (canonical) sizes $(size(cimgs1)) and $(size(cimgs2))."))

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
            "Expected the (d)ssims to be a CuVector of length the batch size $b! \
             Received size $(size(N_ssims))."))
    end
 
    # Canonicalize N_dssims_dQMP, or allocate it in canonical format (or do nothing if we
    # don't need it)
    allocated_N_dssims_dQMP = !skip_backward && isnothing(N_dssims_dQMP)
    if allocated_N_dssims_dQMP
        N_dssims_dQMP = CuArray{T}(undef, h, NB_BWD_CONVS_I64, c, w, b)
    elseif !skip_backward  # && !isnothing(N_dssims_dQMP)
        N_dssims_dQMP = _canonicalize_gradient_buffer(N_dssims_dQMP)
        size(N_dssims_dQMP) == (h, NB_BWD_CONVS_I64, c, w, b) || throw(DimensionMismatch(
            "Given images in canonical size $(size(cimgs1)), expected the intermediate \
            gradients buffer N_dssims_dQMP to have size \
            ($h, $NB_BWD_CONVS_I64, $c, $w, $b), but received $(size(N_dssims_dQMP))!"
        ))
    end

    # --------------------
    # Forward pass
    #
    nb_tiles = cld.((h, w), (TILE_HEIGHT, TILE_WIDTH))       # 2D
    nb_blocks = cld(prod(nb_tiles), MAX_NB_TILES_PER_BLOCK)  # 1D
    shmem_bytes = _get_shmem_fwd_bytes(T1, T2, Tc_fwd)

    max_static_shared_memory = attribute(
        device(), CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)  # also in bytes
    if shmem_bytes <= max_static_shared_memory
        @cuda threads=NB_THREADS blocks=nb_blocks _ssim_kernel_fwd!(
            N_ssims, 
            N_dssims_dQMP,
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
            N_dssims_dQMP,
            cimgs1, cimgs2,
            c_val, single_image_pair_val,
            Val(true),  # dynamic
            skip_ssim_val, skip_backward_val
        )
        attributes(kernel.fun)[CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] =
            shmem_bytes
        kernel(N_ssims, 
            N_dssims_dQMP,
            cimgs1, cimgs2,
            c_val, single_image_pair_val,
            Val(true),
            skip_ssim_val, skip_backward_val;
            threads=NB_THREADS, blocks=nb_blocks, shmem=shmem_bytes)
    end

    # Convert N_ssims to (d)ssims
    if !skip_ssim
        isnothing(ssims) && ( N_ssims = _maybe_unpack_and_free!(N_ssims, imgs1) )
        # If ssims was provided, don't do any unpacking and definitely do not free it!

        ssims = _divide_by_N_and_maybe_d!(
            N_ssims, nb_pixel_components_per_image,
            use_dssim, divide_by_two_in_dssim)
    end
    #
    # forward pass
    # --------------------

    # --------------------
    # Backward pass
    #
    if !skip_backward
        if isnothing(dL_dimgs1) 
            dL_dcimgs1 = CuArray{T}(undef, c, h, w, b)
        else
            dL_dcimgs1 = _canonicalize_image(dL_dimgs1)
            size(dL_dcimgs1) == size(cimgs1) || throw(DimensionMismatch(
                "Expected the output gradient dL_dimgs1 to have the same canonical size \
                 $(size(cimgs1)) as the first input image batch! \
                 Instead received (canonical) size $(size(dL_dcimgs1))."))
        end

        TdQMP = eltype(N_dssims_dQMP)
        Tc_bwd = promote_type(TK, TdQMP)  # used in convolution('s shared memory array)
        shmem_bytes = _get_shmem_bwd_bytes(TdQMP, Tc_bwd, c_val)
        if shmem_bytes <= max_static_shared_memory
            @cuda threads=NB_THREADS blocks=nb_blocks _ssim_kernel_bwd!(
                dL_dcimgs1, 
                N_dssims_dQMP,
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
                N_dssims_dQMP,
                cimgs1, cimgs2, 
                c_val, single_image_pair_val,
                Val(true),
                use_dssim, divide_by_two_in_dssim
            )
            attributes(kernel.fun)[CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] =
                shmem_bytes
            kernel(
                dL_dcimgs1, 
                N_dssims_dQMP,
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
    #
    # backward pass
    # --------------------

    # Cleanup
    allocated_N_dssims_dQMP && unsafe_free!(N_dssims_dQMP)
    # N_ssims was already freed earlier, if necessary

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