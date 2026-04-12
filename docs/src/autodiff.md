# Automatic differentiation

We include [ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl) `rrule`s for `ssim` and `dssim` as a package extension. Consequently automatic differentiation using [Zygote.jl](https://github.com/FluxML/Zygote.jl) is supported out of the box. Note that like in [`ssim_gradient`](@ref) we only differentiate with respect to the first image (batch). The same conventions about the input and output format apply.

An example:
```julia-repl
julia> using FastCUDASSIM, CUDA, Zygote, ImageCore

julia> img1 = CUDA.rand(Gray{N0f8}, 2, 3); img2 = CUDA.rand(Gray{N0f8}, 2, 3);

julia> Zygote.withgradient(ssim, img1, img2)
(val = 0.6785328f0, grad = (Gray{Float32}[0.0048529073 0.38347715 0.36758304; 0.20223245 0.5144576 0.11673692], nothing))
```
In particular, the gradient with respect to `img2` is `nothing`. Also note that for such tiny images our SSIM value is quite high due to the zero-padding.

When using [`dssim`](@ref) as (part of) a training loss, ensure the reference image is the second input.
```julia-repl
julia> render = CUDA.rand(Float32, 2, 3); ground_truth = CUDA.rand(Float32, 2, 3);

julia> Zygote.withgradient(y -> dssim(y, ground_truth), render)
(val = 0.101208925f0, grad = (Float32[-0.26408646 -0.19994365 -0.011348972; 0.029529918 -0.077309705 0.029900908],))
```

---

Although less convenient and flexible, when possible consider using the in-place functions like [`dssim_with_gradient!`](@ref) to squeeze out every bit of performance (cf. the [benchmarks](benchmarks.md)).
```julia-repl
julia> render_batch = CUDA.rand(Float32, 3, 32, 48, 128)  # Batch of 128 3-channel images of height 32 and width 48 pixels

julia> ground_truth_batch = CUDA.rand(Float32, size(render_batch));

julia> dL_render_batch = similar(render_batch);
       dssims = CuArray{Float32}(undef, 128);                       # batch size
       N_dssims_dQMP = CuArray{Float32}(undef, 32, 3, 3, 48, 128);  # height x 3 x channels x width x batch size

julia> dssim_with_gradient!(dssims, dL_render_batch, render_batch, ground_truth_batch, N_dssims_dQMP);

julia> # (...): Use dL_render_batch to update the parameters of the model outputting render_batch (in-place)

julia> dssim_with_gradient!(dssims, dL_render_batch, render_batch, ground_truth_batch, N_dssims_dQMP);
```