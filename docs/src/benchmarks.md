# Benchmarks

By avoiding global memory as much as reasonably possible we are much faster than implementations using high-level convolution calls. The Julia package [SSIMLoss.jl](https://github.com/nikopj/SSIMLoss.jl) falls under this last category, as does the Python package [pytorch-msssim](https://github.com/VainF/pytorch-msssim). On the other hand [fused-ssim](https://github.com/rahul-goel/fused-ssim) is a Python package (well, CUDA + other backends, with PyTorch bindings) with a similar approach to ours. Below we will compare the (CUDA) performance of these packages, as well as [ImageQualityIndexes.jl](https://github.com/JuliaImages/ImageQualityIndexes.jl) as a CPU-only method for reference. 

We should point out that fused\_ssim.py uses zero-padding like us, while SSIMLoss.jl uses either valid padding (i.e. cropping) or symmetric padding. pytorch\_msssim uses valid padding. Writing `c` for the number of color channels in an image, `h` for its width and `w` for its height, ImageQualityIndexes.jl and FastCUDASSIM.jl use images in `(c, h, w)` format, while SSIMLoss.jl uses `(h, w, c)`, and the Python packages use `(w, h, c)` when converted to column-major order.

The benchmarking code can be found under `benchmark` in the GitHub repo. The results on a system equipped with an intel i7-7700K CPU and RTX 3070 GPU are shown below. We always present the median execution time. 

* SSIM for a single full HD image pair (1 | 3 channels):
```
    * ImageQualityIndexes.jl:                 234     ms  |  546     ms
    * SSIMLoss.jl (`crop = false`):            19.9   ms  |   58.1   ms
    * pytorch_msssim.py:                        4.16  ms  |    8.27  ms
    * fused_ssim.py:                            0.269 ms  |    0.891 ms
    * FastCUDASSIM.jl:                          0.284 ms  |    0.831 ms
```
* DSSIMs and gradients for a batch of 32 images of size 256 x 256 (1 | 3 channels):
```
    * SSIMLoss.jl (Zygote):                     ERROR: Gradient Thunk(ChainRules.var"#...) should be a tuple
    * pytorch_msssim.py:                        11.9   ms  |  15.6   ms
    * fused_ssim.py:                             0.732 ms  |   1.78  ms
    * FastCUDASSIM.jl (Zygote):                  0.825 ms  |   2.17  ms
    * FastCUDASSIM.jl (`dssim_with_gradient!`):  0.642 ms  |   1.74  ms
```

In these benchmarks we used
* Julia 1.12.5
* CUDA 5.9.7 (runtime 13.1, driver 591.86.0, compiler 13.2)
* ImageQualityIndexes 0.3.7
* SSIMLoss 1.0.0
* FastCUDASSIM 0.2.0
* Python 3.10.20
* PyTorch 2.7.1+cu118
* pytorch_msssim 1.0.0 (latest commit: b057b07, August 16, 2023)
* fused_ssim 1.0.0 (latest commit: a7c48d6, February 2, 2026).