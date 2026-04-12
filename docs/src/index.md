# FastCUDASSIM.jl

Fast calculation of the [Structural Similarity Index Measure](https://en.wikipedia.org/wiki/Structural_similarity_index_measure) (SSIM) and its gradients, on NVIDIA GPUs.


## Quick start

```julia-repl
julia> using Pkg; Pkg.add(["CUDA", "FastCUDASSIM"])

julia> using CUDA, FastCUDASSIM

julia> img1 = CUDA.rand(Float32, 3, 512, 768); img2 = similar(img1);

julia> ssim(img1, img2)  # Actual value depends on the RNG above
4.7802605f-6

julia> ssim_gradient(img1, img2)  # w.r.t. img1
3×512×768 CuArray{Float32, 3, CUDA.DeviceMemory}:
[:, :, 1] =
 -4.05717f-11 ...
 ...          ...

julia> Pkg.add("Zygote"); using Zygote

julia> Zygote.gradient(x -> ssim(x, img2), img1)
(Float32[-4.057169f-11 ...],)

julia> Pkg.add("TestImages"); using TestImages

julia> dssim(cu(testimage("cameraman.tif")), cu(testimage("mandril_gray.tif")))  # CuMatrix{Gray{N0f8}} inputs
0.8373802f0
```

## Key points
- Fast by avoiding global memory as much as reasonably possible. See the [benchmarks](benchmarks.md).
- Support for CUDA only: no other graphics APIs, no CPU fallback.
- Reverse-mode [autodiff integration](autodiff.md) via [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl). Works out of the box with [Zygote.jl](https://github.com/FluxML/Zygote.jl).
- The convolutions use zero padding and the usual Gaussian kernel with (conceptual) window size $11 \times 11$ and $\sigma = 1.5$, in `Float32` precision.
- Image intensities are assumed to lie in $[0, 1]$.
- Images should use `[channels x] height x width [[x batch size]]` memory layout. See [Input and output formats](formats.md) for more details.



## Contents
```@contents
Pages = (p -> p.second).(Main.PAGE_NAMES_AND_FILES)
```