"""
Package for computing the Structural Similarity Index Measure (SSIM) and its gradients on
NVIDIA GPUs.

# Quick start
```
julia> using FastCUDASSIM, CUDA

julia> img1 = CUDA.rand(Float32, 3, 512, 768); img2 = similar(img1);

julia> ssim(img1, img2)  # Actual value depends on RNG above
4.7778763f-6
```

# Key points
* CUDA only
* We use zero-padding in the convolutions. The convolution kernel is the typical Gaussian
  with standard deviation ``\\sigma = 1.5`` and window size 11 x 11. We represent it using
  `Float32`.
* We expect Float-like image intensities in [0, 1]. This includes `N0f8`.
* Internally we use `channels x height x width x batch size` format for the input image
  batches, and `height x channels x width x batch size` for internal buffers (`N_dssims_dQ`
  and friends). Supplied inputs need not have this explicit `size`, but must have an
  equivalent memory layout.

# Image formats
* To expand on the last point, consider a `CuMatrix{RGB{N0f8}}` of size `(256, 384)`. It is
  equivalent (via `channelview`) to a `CuArray{N0f8, 3}` of size `(3, 256, 384)`, and
  (via a further `reshape`) to a `CuArray{N0f8, 4}` with size `(3, 256, 384, 1)`. We will
  perform these conversions automatically. Therefore, this input is supported.
* In these conversions we will (if necessary) first expand the channels dimension, followed
  by the batch dimension. Therefore, a `CuArray{Float32}` of `size` `(x, y, z)` will be
  considered to be a single `x`-channel image of height `y` and width `z`. To input a
  grayscale batch consisting of `z` images of `size` `(x, y)`, either use a
  `CuArray{Gray{Float32}, 3}` with `size` `(x, y, z)`, or explicitly set the number of color
  channels to 1 via a `CuArray{Float32, 4}` of `size` `(1, x, y, z)`.
* When we internally allocate an output (D)SSIM or gradient batch (either because an
  allocating method like [`dssim_with_gradient`](@ref) is called, or because in one of the
  mutating methods like [`dssim_with_gradient!`](@ref) the supplied value was `nothing`),
  we will convert this back to a format similar to that of the first input image (batch).
  For example, if no batch axis appears here, then we will return a scalar (D)SSIM value
  instead of a `CuVector` of `length` `1`. There might be some promotions, though. For
  example, `ssim_gradient(imgs1, imgs2)` for `imgs1` a `CuArray{RGB{N0f8}, 3}` of `size`
  `(256, 384, 16)` and `imgs2` similar, will return a `CuArray{RGB{Float32}, 3}` gradient
  batch, again of `size` `(256, 384, 16)`.
"""
module FastCUDASSIM

using CUDA
using CUDA: i32
using GPUArrays: @allowscalar, unsafe_free!

using StaticArrays
using Accessors

using ImageCore

using PrecompileTools
using Preferences


include("core.jl")
include("conversions.jl")
include("interface.jl")

if CUDA.functional()
    include("precompile.jl")
else
    @warn "CUDA is not functional! Skipping precompilation"
end


export ssim,   ssim_gradient,   ssim_with_gradient 
export ssim!,  ssim_gradient!,  ssim_with_gradient!
export dssim,  dssim_gradient,  dssim_with_gradient 
export dssim!, dssim_gradient!, dssim_with_gradient!


end