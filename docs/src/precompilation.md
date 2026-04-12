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

The effects of the different options are shown below, on a system with an intel i7-7700K CPU and NVIDIA RTX 3070 GPU. Between each run we manually deleted the `.julia/compiled/v1.x/FastCUDASSIM` directory, though this is not necessary: after a `set_preferences!` call and after reloading FastCUDASSIM.jl, precompilation will again be triggered. This includes when going down from e.g. `"full"` to `"fast"`.
* `off`:
```julia-repl
julia> using Preferences, CUDA

julia> set_preferences!("FastCUDASSIM", "precompilation" => "off")

julia> using FastCUDASSIM
Precompiling FastCUDASSIM finished.
  1 dependency successfully precompiled in 16 seconds. 116 already precompiled.

julia> x = CUDA.rand(Float32, 128, 256); y = CUDA.rand(Float32, size(x)); x64 = Float64.(x); y64 = Float64.(y);

julia> @time ssim_with_gradient(x, y);
  4.379760 seconds (11.13 M allocations: 541.534 MiB, 0.95% gc time, 172.82% compilation time: <1% of which was recompilation)

julia> @time ssim_with_gradient(x, y);
  0.000721 seconds (131 allocations: 4.484 KiB, 4970.40% compilation time)

julia> @time ssim(x64, y64);
  1.757906 seconds (4.93 M allocations: 239.094 MiB, 0.77% gc time, 171.46% compilation time: <1% of which was recompilation)

julia> @time ssim(x64, y64);
  0.000681 seconds (3.31 k allocations: 152.594 KiB, 5023.12% compilation time)
```

* `fast`:
```julia-repl
julia> using Preferences, CUDA

julia> set_preferences!("FastCUDASSIM", "precompilation" => "fast"; force = true)

julia> using FastCUDASSIM
Precompiling FastCUDASSIM finished.
  1 dependency successfully precompiled in 54 seconds. 116 already precompiled.

julia> x = CUDA.rand(Float32, 128, 256); y = CUDA.rand(Float32, size(x)); x64 = Float64.(x); y64 = Float64.(y);

julia> @time ssim_with_gradient(x, y);  # Covered by the fast precompilation: better than with "off", but not great
  1.543940 seconds (4.62 M allocations: 219.592 MiB, 1.46% gc time, 131.09% compilation time: <1% of which was recompilation)

julia> @time ssim_with_gradient(x, y);
  0.000512 seconds (2.38 k allocations: 113.859 KiB, 7366.85% compilation time)

julia> @time ssim(x64, y64);  # Not covered
  1.767369 seconds (4.62 M allocations: 224.392 MiB, 1.53% gc time, 162.41% compilation time: <1% of which was recompilation)

julia> @time ssim(x64, y64);
  0.001063 seconds (3.46 k allocations: 165.531 KiB, 4001.41% compilation time)
```

* `full`:
```julia-repl
julia> using Preferences, CUDA

julia> set_preferences!("FastCUDASSIM", "precompilation" => "full"; force = true)

julia> using FastCUDASSIM
Precompiling FastCUDASSIM finished.
  1 dependency successfully precompiled in 150 seconds. 116 already precompiled.

julia> x = CUDA.rand(Float32, 128, 256); y = CUDA.rand(Float32, size(x)); x64 = Float64.(x); y64 = Float64.(y);

julia> @time ssim_with_gradient(x, y);
  1.858585 seconds (4.69 M allocations: 223.484 MiB, 0.67% gc time, 141.02% compilation time: 6% of which was recompilation)

julia> @time ssim_with_gradient(x, y);
  0.000527 seconds (1.06 k allocations: 48.172 KiB, 8075.60% compilation time)

julia> @time ssim(x64, y64);  # Now also covered
  0.760213 seconds (2.15 M allocations: 104.389 MiB, 1.83% gc time, 134.06% compilation time: 14% of which was recompilation)

julia> @time ssim(x64, y64);
 0.001688 seconds (4.95 k allocations: 226.125 KiB, 2254.99% compilation time)
```


[^1]: See <https://discourse.julialang.org/t/how-to-precompile-cuda-kernel-itself/121849/> and the relevant GitHub issues.