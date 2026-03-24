# Allocation-free conversions to and from the expected formats in _ssim_fwd_bwd!
# (mostly CuArray{<:AbstractFloat, 4})

"""
    _canonicalize_image(img)

Convert `img` to the expected canonical format of channels x height x width x batch size
with `eltype` some Float-like, if possible.

Does not allocate or rearrange memory. In particular, CPU-arrays will not be uploaded to the
GPU, so at some point in the `ssim` pipeline an error will be thrown in this case. We also
do not make sure the values end up between 0 and 1.

When not all 4 dimensions are supplied, we consider the supplied dimension in order of
`height`, `width`, `channels` (and `batch_size`). E.g. a `CuArray{Float32, 3}` will be
considered as a single image of `size` `(channels, height, width)`. In particular, it is
not possible to supply a batch of grayscale images as `(height, width, batch_size)` with
`eltype` `Float32`. To achieve this, use `Gray{Float32}`.
"""
function _canonicalize_image end

_canonicalize_image(img) = img  # Generic fallback
_canonicalize_image(img::AbstractArray{<:Any, 3}) = _canonicalize_image(
    reshape(img, size(img)..., 1))  # Single image (with channels) (_not grayscale batch_)
# c x h x w -> c x h x w x 1
_canonicalize_image(img::AbstractMatrix) = _canonicalize_image(
    reshape(img, 1, size(img)...))  # Single grayscale (but not Gray) image
_canonicalize_image(img::AbstractMatrix{<:Colorant}) = _canonicalize_image(channelview(img))
# Single Gray/RGB(/...) image
# e.g. CuMatrix{RGB}  -> 3 x h x w -> 3 x h x w x 1
#      CuMatrix{Gray} ->     h x w -> 1 x h x w     -> 1 x h x w x 1
_canonicalize_image(img::AbstractArray{<:Colorant, 3}) = 
    _canonicalize_image(channelview(img))  # color image batch: h x w x b -> c x h x w x b
_canonicalize_image(img::AbstractArray{<:Colorant{<:Any, 1}, 3}) = 
    _canonicalize_image(reshape(channelview(img), 1, size(img)...))  # grayscale batch
# h x w x b -> 1 x h x w x b


"""
    _canonicalize_gradient_buffer(N_dssims_dX)

Like [`_canonicalize_image`](@ref), but for the height x channels x width x batch size
format used in the gradient buffers `N_dssims_dQ`, `N_dssims_dM` and `N_dssims_dP`.
"""
function _canonicalize_gradient_buffer end

_canonicalize_gradient_buffer(gb) = gb
_canonicalize_gradient_buffer(gb::AbstractArray{<:Any, 3}) = 
    _canonicalize_gradient_buffer(reshape(gb, size(gb)..., 1))  # h x c x w -> h x c x w x 1
_canonicalize_gradient_buffer(gb::AbstractMatrix) = 
    _canonicalize_gradient_buffer(reshape(gb, size(gb, 1), 1, size(gb, 2)))  
# h x w -> h x 1 x w -> h x 1 x w x 1

# No support for e.g. CuMatrix{RGB}, as this is 3 x h x w, not h x 3 x w


"""
    _decanonicalize_gradient_image(dL_dimg, img)

Convert `dL_dimg` from channels x height x width x batch size format to the one used
by `img`. The inverse of [`_canonicalize_image`](@ref).
"""
function _decanonicalize_gradient_image end

_decanonicalize_gradient_image(::Nothing, _) = nothing
_decanonicalize_gradient_image(dL_dimg, img) = __decanonicalize_gradient_image(dL_dimg, img)

# First argument is now guaranteed to be !isnothing
__decanonicalize_gradient_image(dL_dimg, img) = reshape(dL_dimg, size(img)...)
__decanonicalize_gradient_image(
    dL_dimg, img::AbstractArray{C}
) where {N, C <: Colorant{<:Any, N}} = 
    colorview(base_color_type(C), reshape(dL_dimg, N, size(img)...))
    # (or  reshape(dL_dimg, size(dL_dimg)[1:3]...)  )
# e.g. CuMatrix{RGB{N0f8}}:  
#          CuArray{Float32}  3 x h x w x 1 -> 3 x h x w -> CuMatrix{RGB{Float32}} h x w
#      CuArray{RGB{Float64}, 3}:
#          CuArray{Float64}  3 x h x w x b -> same -> CuArray{RGB{Float64}, 3} h x w x b
__decanonicalize_gradient_image(
    dL_dimg, img::AbstractArray{C}
) where {C <: Colorant{<:Any, 1}} = 
    colorview(base_color_type(C), reshape(dL_dimg, size(img)...))
#      CuMatrix{Gray{Float64}}:
#          CuArray{Float64}  1 x h x w x 1 -> h x w -> CuMatrix{Gray{Float64}} h x w
#      CuArray{Gray{N0f8}, 3}:
#          CuArray{Float32}  1 x h x w x b -> h x w x b -> CuMatrix{Gray{Float64}} h x w x b


"""
    _maybe_unpack_and_free!(vals, img)

Extract the singleton value from `vals` and free the array, if no batch axis is present in
`img`. Otherwise, return `vals`.
"""
function _maybe_unpack_and_free!(vals, img)
    if _should_unpack_and_free(vals, img)
        v = @allowscalar vals[]
        unsafe_free!(vals)
        return v
    else
        return vals
    end
end

"""
    _should_unpack_and_free(vals, img)

Check whether the canonical (D)SSIM outputs `vals` should be extracted as they form a
singleton and `img` does not contain a batch axis.
"""
function _should_unpack_and_free end

_should_unpack_and_free(_, _) = false
_should_unpack_and_free(vals, ::AbstractArray{<:Any, N}) where N = 
    N <= 3 && size(vals) == (1,)
_should_unpack_and_free(vals, ::AbstractArray{<:Colorant, N}) where N = 
    N <= 2 && size(vals) == (1,)
