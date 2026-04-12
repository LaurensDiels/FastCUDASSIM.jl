# Input and output formats

We assume that image pixel intensities are normalized between 0 and 1. This is _not_ checked in the code, i.e. fully the responsibility of the caller.

## `size`s

Internally we always use input and output image batches with `channels x height x width x batch size` layout, which we will refer to as _canonical format_. However, this does not mean we require a `CuArray{T, 4}`, only that our input needs to be able to be converted to it using `reshape`s adding singleton dimensions, and `channelview`. For example, a `CuMatrix{RGB{N0f8}]` of `size` `(256, 384)` is equivalent to a `CuArray{N0f8, 3}` of `size` `(3, 256, 384)`, which in turn we can `reshape` to include the batch axis, resulting in a `CuArray{N0f8, 4}` with `size` `(3, 256, 384, 1)`. Such conversions are taken care of automatically.

```julia-repl
julia> using FastCUDASSIM, CUDA, ImageCore

julia> img1 = CUDA.rand(RGB{N0f8}, 256, 384); img2 = CUDA.rand(RGB{N0f8}, 256, 384);

julia> ssim(img1, img2)
0.015220874f0
```
This is mostly equivalent to
```julia-repl
julia> ssim(reshape(channelview(img1), 3, size(img1)..., 1), reshape(channelview(img2), 3, size(img2)..., 1))
1-element CuArray{Float32, 1, CUDA.DeviceMemory}:
 0.015220872
```
We will come back to the difference in output format.

In these size conversions we first add a channel axis if necessary, then a batch axis. This implies that non-`Colorant` images with three dimensions will be considered as a single color image, to which we need to add a singleton batch dimension. That is, an image of `eltype` (say) `Float32` and `size` `(x, y, z)` will be interpreted as an `x`-channel image with height `y` and width `z`, equivalent to a singleton batch of `size` `(x, y, z, 1)`. In particular, we do not consider it as a grayscale batch of `z` images of size `(x, y)`, which would have canonical `size` `(1, x, y, z)`. It is still possible to provide such a grayscale batch of course, either by explicitly providing it in the latter format, or by using `Gray{Float32}` as `eltype`, and `size` `(x, y, z)`.

In the code example above the only difference was in the output format. The idea here is that we will try to make the output similar to the first input image. In the first example we had not provided a batch axis, and therefore the output does not have one either. I.e. we just return a scalar. In the second example the input to `ssim` did have a batch axis, and therefore so does the produced output `ssim` value `CuVector`. The same conversion principle applies to the output gradients.
```julia-repl
julia> img1 = CUDA.rand(Float32, 256, 384); img2 = CUDA.rand(Float32, 256, 384);

julia> ssim_gradient(img1, img2)
256×384 CuArray{Float32, 2, CUDA.DeviceMemory}:
  9.98839f-6 ...
  ...        ...

julia> ssim_gradient(reshape(img1, 1, size(img1)...), reshape(img2, 1, size(img2)...))
1×256×384 CuArray{Float32, 3, CUDA.DeviceMemory}:
[:, :, 1] =
 9.98839f-6 ...
 ...        ...

julia> ssim_gradient(reshape(img1, 1, size(img1)..., 1), reshape(img2, 1, size(img2)..., 1))
1×256×384×1 CuArray{Float32, 4, CUDA.DeviceMemory}:
[:, :, 1, 1] =
 9.98839f-6 ...
 ...        ...
```
In all these examples the `size` of the returned gradient is the same as that of the first input argument. While we advise against it, it is possible to use different formats for the two input images, as they will be canonicalized seperately. For the output format we solely rely on the first image.
```julia-repl
julia> ssim_gradient(reshape(img1, 1, size(img1)..., 1), img2)
1×256×384×1 CuArray{Float32, 4, CUDA.DeviceMemory}:
...
```

Making the output similar to the input includes matching `Colorant`s, if relevant:
```julia-repl
julia> img1 = CUDA.rand(RGB{Float32}, 256, 384, 1); img2 = CUDA.rand(RGB{Float32}, 256, 384, 1);

julia> ssim_gradient(img1, img2)
256×384×1 CuArray{RGB{Float32}, 3, CUDA.DeviceMemory}:
[:, :, 1] =
 RGB(5.14282f-6, 9.09983f-6, 4.99638f-6)  ...
 ...                                      ...
```

For the in-place methods we will again perform `size` conversions automatically, whenever reasonably possible. An important remark here is that the internal gradients buffer `N_dssims_dQMP` uses canonical `height x 3 x channels x width x batch size` memory order (for memory coalescing). We will still automatically add the singleton channel and batch axes when necessary, but now `Colorant` `eltype`s are not supported. Additionally, in e.g. [`ssim!`](@ref) we always expect a `CuVector` of `length` the (possibly implicit) batch size, never a scalar.

The following example is purely illustrative. Again, to avoid confusion we advise to use as consistent input formats as possible.
```julia-repl
julia> img1 = CUDA.rand(256, 384); img2 = CUDA.rand(1, 256, 384, 1);  # Both (1, 256, 384, 1) in canonical format

julia> ssim_vector = CuArray{Float32}(undef, 1);              # Must be a CuVector
       dL_dimg1 = CuArray{Float32}(undef, 1, 256, 384);       # Will be converted to canonical size (1, 256, 384, 1)
       N_dssim_dQMP = CuArray{Float32}(undef, 256, 3, 384);   #                                     (256, 3, 1, 384, 1)
    
julia> CUDA.@sync ssim_with_gradient!(ssim_vector, dL_dimg1, img1, img2, N_dssim_dQMP)  # Returns ssim_vector and dL_dimg1 (in their input format)
(Float32[0.01444148], Float32[2.195323f-5 ... ;;; ...])
```

## `eltype`s
Our Gaussian SSIM kernel is hard-coded as `Float32`. When it is used in the convolutions, the output will be of the promotion type of the `eltype` of the input image with `Float32`. When an allocating method is used, we will then use the promotion type of both images' `eltype`s and `Float32` for the output. For example,
```julia-repl
julia> img1 = CUDA.rand(Float16, 24, 32); img2 = CUDA.rand(Float16, size(img1));

julia> ssim(img1, img2)
0.09606406f0

julia> promote_type(eltype(img1), eltype(img2), Float32)
Float32

julia> ssim(img1, Float64.(img2))
0.09606405144438597

julia> promote_type(eltype(img1), Float64, Float32)
Float64
```
For `N0f8` and `N0f16` the promotion type is also `Float32`.
```julia-repl
julia> ssim_with_gradient(cu(N0f16.(Array(img1))), cu(Gray{N0f8}.(Array(img2))))
(0.096031874f0, Float32[-0.00219706 ...])
```
In this example we first of all point out that the `CuMatrix{Gray{N0f8}}` is internally converted to a `CuArray{N0f8, 4}`, where we added the singleton gray color channel and batch axes. Then for the output we have `promote_type(N0f16, N0f8, Float32) === Float32`. Secondly, we need to pass through the CPU to construct the `Normed` inputs. The reason for this is that the `Normed` constructor has a check possibly throwing an error with a string-interpolated message. This is not supported on the device.

For in-place methods the convolutions will still take place as described before, but in the final stage when directly preparing the values to write away, we will work with the provided type. `Normed` `eltype`s are not supported, as explained above.
```julia-repl
julia> CUDA.@sync ssim!(CUDA.zeros(Float16, 1), img1, img2, false)
1-element CuArray{Float16, 1, CUDA.DeviceMemory}:
 0.0962
```
Of course, we would advise against using too low a precision. We primarily wrote the code with `Float32` in mind and do not guarantee sufficient numerical stabitility for lower precision.


## `nothing`
Finally, we mention that for in-place methods like [`ssim_with_gradient!`](@ref) you can use `nothing` for one of the outputs or for the intermediate gradients buffer. In this case we will internally allocate the required arrays with appropriate `eltype`s (the same ones as would be used for the allocating methods like [`ssim_with_gradient`](@ref)). Buffers provided as `nothing` will be automatically `unsafe_free!`d. The same is true for the singleton `ssims` `CuVector` if we do not return it, but its only scalar value.
```julia-repl
julia> img1 = CUDA.rand(Float32, 32, 48); img2 = CUDA.rand(Float32, size(img1)); dL_dimg1 = similar(img1);

julia> CUDA.@sync ssim_with_gradient!(nothing, dL_dimg1, img1, img2, nothing)
(0.071204804f0, Float32[0.0012562195 ...])

julia> CUDA.@sync ssim_with_gradient!(nothing, dL_dimg1, reshape(img1, (1, size(img1)..., 1)), img2, nothing)
(Float32[0.071204804], Float32[0.0012562195 ...])
```
In both cases we internally allocate a singleton `CuVector` for the single SSIM value. In the first case we extract its value and free the `CuVector`, while in the second case we return (and obviously do not free) the `CuVector`.
