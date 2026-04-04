# FastCUDASSIM

[![Build Status](https://github.com/LaurensDiels/FastCUDASSIM.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/LaurensDiels/FastCUDASSIM.jl/actions/workflows/CI.yml?query=branch%3Amaster)[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://laurensdiels.github.io/FastCUDASSIM.jl/stable/)[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://laurensdiels.github.io/FastCUDASSIM.jl/dev/)


Fast computation of the [Structural Similarity Index Measure](https://en.wikipedia.org/wiki/Structural_similarity_index_measure) (SSIM) and its gradients on NVIDIA GPUs.

## Quick start
```julia-repl
julia> using Pkg; Pkg.add(["CUDA", "FastCUDASSIM"])

julia> using CUDA, FastCUDASSIM

julia> nb_channels = 3; height = 128; width = 192; batch_size = 4;

julia> x = CUDA.rand(Float32, nb_channels, height, width, batch_size); y = CUDA.rand(Float32, size(x));

julia> ssims, gradients = ssim_with_gradient(x, y);

julia> ssims  # The concrete values here depend on the RNG above
4-element CuArray{Float32, 1, CUDA.DeviceMemory}:
 0.02415132
 0.029480096
 0.034078166
 0.032211296

julia> gradients  # of the ssims w.r.t. the images in x
3×128×192×4 CuArray{Float32, 4, CUDA.DeviceMemory}:
[:, :, 1, 1] =
 -1.94285f-6 ...
 ...         ...
```

## Benchmark
* SSIM for a single full HD RGB image pair:
```
    * ImageQualityIndexes.jl (CPU):           638     ms
    * SSIMLoss.jl (GPU, `crop = false`):       60.3   ms
    * pytorch_msssim.py (GPU):                  8.35  ms
    * fused_ssim.py:                            0.875 ms
    * FastCUDASSIM.jl:                          0.988 ms
```
* DSSIMs and gradients for a batch of 32 grayscale images of size 256 x 256:
```
    * SSIMLoss.jl (Zygote):                     ERROR: Gradient Thunk(ChainRules.var"#...) should be a tuple
    * pytorch_msssim.py (GPU):                  12.0   ms
    * fused_ssim.py:                             0.765 ms
    * FastCUDASSIM.jl (Zygote):                  1.10  ms
    * FastCUDASSIM.jl (`dssim_with_gradient!`):  0.824 ms
```
on an Intel i7-7700K and NVIDIA RTX 3070. The tested implementations are:
* [ImageQualityIndexes.jl](https://github.com/JuliaImages/ImageQualityIndexes.jl)
* [SSIMLoss.jl](https://github.com/nikopj/SSIMLoss.jl)
* [pytorch-msssim](https://github.com/VainF/pytorch-msssim)
* [fused-ssim](https://github.com/rahul-goel/fused-ssim)
