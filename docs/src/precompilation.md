# Precompilation

We have an extensive precompilation suite which should cover most common argument types for all our methods. However, running it takes quite some time. Additionally, the CUDA kernels themselves are currently not yet persistently stored[^1] so that the effect of precompilation is only limited (though certainly noticeable). For this reason we use [Preferences.jl](https://github.com/JuliaPackaging/Preferences.jl) to also allow for partial precompilation, or none at all. We provide three options:
* `"off"`: Do not precompile anything from FastCUDASSIM.jl itself.
* `"fast"` (the default): Precompile all methods for image batches with 1 or 3 color channels and `eltype` `Float32`. Also precompile `ssim` and `dssim` for `Gray{N0f8}` and `RGB{N0f8}`.
* `"full"`: The full suite.
The desired option can be selected via
```julia
using Preferences
set_preferences!("FastCUDASSIM", "precompilation" => <option name>)
```
within your environment. If the `LocalPreferences.toml` file already exists with a section `[FastCUDASSIM]` containing `precompilation`, you can use the option `force = true` to overwrite the value (see also below).

The effects of the different options are shown below, on a system with an intel i7-7700K CPU and NVIDIA RTX 3070 GPU. Between each run we manually deleted the `.julia/compiled/v1.x/FastCUDASSIM.jl` directory, though this is not necessary: after a `set_preferences!` call and after reloading FastCUDASSIM.jl, precompilation will again be triggered. This includes when going down from e.g. `"full"` to `"fast"`.
* `off`:
```julia-repl
julia> using Preferences, CUDA

julia> set_preferences!("FastCUDASSIM", "precompilation" => "off")

julia> using FastCUDASSIM
Precompiling FastCUDASSIM finished.
  1 dependency successfully precompiled in 19 seconds. 116 already precompiled.

julia> x = CUDA.rand(Float32, 128, 256); y = similar(x);

julia> @time ssim_with_gradient(x, y);
 16.066590 seconds (30.73 M allocations: 1.472 GiB, 1.65% gc time, 182.96% compilation time: 1% of which was recompilation)

julia> @time ssim_with_gradient(x, y);
  0.003356 seconds (8.91 k allocations: 455.367 KiB, 1416.90% compilation time)
```

* `fast`:
```julia-repl
julia> using Preferences, CUDA

julia> set_preferences!("FastCUDASSIM", "precompilation" => "fast"; force = true)

julia> using FastCUDASSIM
Precompiling FastCUDASSIM finished.
  1 dependency successfully precompiled in 67 seconds. 116 already precompiled.

julia> x = CUDA.rand(Float32, 128, 256); y = similar(x);  # covered by the "fast" precompilation

julia> @time ssim_with_gradient(x, y);  # better than with "off", but not great
 11.367837 seconds (22.07 M allocations: 1.054 GiB, 1.10% gc time, 183.18% compilation time: 2% of which was recompilation)

julia> @time ssim_with_gradient(x, y);
  0.005376 seconds (12.07 k allocations: 619.776 KiB, 1 lock conflict, 586.16% compilation time)
```

* `full`:
```julia-repl
julia> using Preferences, CUDA

julia> set_preferences!("FastCUDASSIM", "precompilation" => "full"; force = true)

julia> using FastCUDASSIM
Precompiling FastCUDASSIM finished.
  1 dependency successfully precompiled in 281 seconds. 116 already precompiled.

julia> x = CUDA.rand(Float32, 128, 256); y = similar(x);

julia> @time ssim_with_gradient(x, y);
 11.501233 seconds (20.55 M allocations: 1005.402 MiB, 2.08% gc time, 180.51% compilation time: 5% of which was recompilation)

julia> @time ssim_with_gradient(x, y);
  0.002357 seconds (8.51 k allocations: 431.929 KiB, 1051.01% compilation time)
```


[^1]: See <https://discourse.julialang.org/t/how-to-precompile-cuda-kernel-itself/121849/> and the relevant GitHub issues.