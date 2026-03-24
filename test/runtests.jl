using FastCUDASSIM
using FastCUDASSIM: C1, C2
using Test

using CUDA
using GPUArrays: @allowscalar
using ImageCore
using TestImages
using Zygote

if CUDA.functional()
    @testset "FastCUDASSIM.jl" begin
        include("sanity_checks.jl")
        include("single_pixel.jl")
        include("forwarding.jl")
        include("dssim.jl")
        include("sizes.jl")
        include("eltypes.jl")
        include("batch_order.jl")
        include("autodiff.jl")
        include("correctness.jl")
    end
else
    @warn "CUDA is not functional! Skipping tests"
end