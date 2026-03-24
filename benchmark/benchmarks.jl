# Run via `julia --project=benchmark ./benchmark/benchmarks.jl` when the current
# working directory is the FastCUDASSIM.jl folder

using FastCUDASSIM
using BenchmarkTools
using CUDA
using GPUArrays: unsafe_free!

using InteractiveUtils: versioninfo
using Random: rand!
using ImageCore
using ImageQualityIndexes

using cuDNN  # Needed for SSIMLoss CUDA extension
using SSIMLoss
using Zygote

function indented_display(s::AbstractString, indent::Integer)
    println(replace(s, r"^"m => " " ^ indent))
    # Add spaces to the start (^) of each line (m)
end

function indented_display(x, indent::Integer)
    # Capture display(x) as a (colored) String
    s = sprint(io -> show(IOContext(io, :color => true), MIME"text/plain"(), x))
    indented_display(s, indent)
end

function indented_display(indent::Integer)
    return s -> indented_display(s, indent)
end  # useful for piping


BenchmarkTools.DEFAULT_PARAMETERS.seconds = 10
BenchmarkTools.DEFAULT_PARAMETERS.evals = 1


println("Info:")
println("  Julia: ")
indented_display(sprint(versioninfo), 4)
println("  CUDA: ")
indented_display(sprint(CUDA.versioninfo), 4)
println("  ImageQualityIndexes ", pkgversion(ImageQualityIndexes))
println("  SSIMLoss ", pkgversion(SSIMLoss))
println("  FastCUDASSIM ", pkgversion(FastCUDASSIM))
print("\n" ^ 2)

println("Single full HD RGB image pair SSIM:")
c, h, w, b = 3, 1080, 1920, 1

println("  ImageQualityIndexes (CPU):")
img1_iqi = rand(RGB{Float32}, h, w)
img2_iqi = rand(RGB{Float32}, h, w)
@benchmark(
    assess_ssim($img1_iqi, $img2_iqi; crop = false);
    setup = ( rand!($img1_iqi); rand!($img2_iqi) )
) |> indented_display(4)
println()

println("  SSIMLoss (GPU):")
img1_sl = CUDA.rand(Float32, h, w, c, b)
img2_sl = CUDA.rand(Float32, h, w, c, b)
kernel = ssim_kernel(img1_sl)
@benchmark(
    CUDA.@sync SSIMLoss.ssim($img1_sl, $img2_sl, $kernel; crop = false);
    setup = ( rand!($img1_sl); rand!($img2_sl) )
) |> indented_display(4)
println()

println("  FastCUDASSIM:")
img1_fcs = CUDA.rand(Float32, c, h, w)
img2_fcs = CUDA.rand(Float32, c, h, w)
@benchmark(
    CUDA.@sync FastCUDASSIM.ssim($img1_fcs, $img2_fcs);
    setup = ( rand!($img1_fcs); rand!($img2_fcs) )
) |> indented_display(4)
println()

print("\n" ^ 2)

println("Grayscale batch DSSIMs and gradients:")
c, h, w, b = 1, 256, 256, 32

# SSIMLoss's backward pass no longer seems to work:
#   ERROR: Gradient Thunk(ChainRules.var"#...) should be a tuple
#=
println("  SSIMLoss (GPU, Zygote):")
imgs1_sl = CUDA.rand(Float32, h, w, c, b)
imgs2_sl = CUDA.rand(Float32, h, w, c, b)
kernel = ssim_kernel(imgs1_sl)
@benchmark(
    CUDA.@sync Zygote.withgradient(
        imgs1_sl -> sum(ssim_loss(imgs1_sl, $imgs2_sl, $kernel; crop = false)), 
        $imgs1_sl
    );  # sum to get a scalar
    setup = ( rand!($imgs1_sl); rand!($imgs2_sl) )
) |> indented_display(4)
println()
=#

println("  FastCUDASSIM (Zygote):")
imgs1_fcs = CUDA.rand(Float32, c, h, w, b)
imgs2_fcs = CUDA.rand(Float32, c, h, w, b)
@benchmark(
    CUDA.@sync Zygote.withgradient(
        imgs1_fcs -> sum(dssim(imgs1_fcs, $imgs2_fcs)), 
        $imgs1_fcs
    ); # sum to get a scalar
    setup = ( rand!($imgs1_fcs); rand!($imgs2_fcs) ),
) |> indented_display(4)
println()

GC.gc()
CUDA.reclaim()
# In principle this is not necessary, but otherwise we skirt on the edge of spilling into
# shared system-GPU memory. Occasionally this already seems to affect the mean timing
# above, though the median should be fine.

println("  FastCUDASSIM (direct, in-place):")
dssims = CuArray{Float32}(undef, b)
dL_dimgs1 = similar(imgs1_fcs)
N_dssims_dQ = similar(dL_dimgs1)
N_dssims_dM = similar(N_dssims_dQ)
N_dssims_dP = similar(N_dssims_dQ)
@benchmark(
    CUDA.@sync begin
        dssim_with_gradient!(
            $dssims, $dL_dimgs1,
            $imgs1_fcs, $imgs2_fcs,
            $N_dssims_dQ, $N_dssims_dM, $N_dssims_dP)
        sum($dssims)  # To be fair w.r.t. the Zygote benchmark
    end;
    setup = ( rand!($imgs1_fcs); rand!($imgs2_fcs) )
) |> indented_display(4)