@testset "Method forwarding" begin
    # a.k.a. consistency between ssim, ssim!, etc.

    c, h, w, b = 3, 200, 300, 1
    imgs1 = CuArray{Float32}(undef, c, h, w, b)
    imgs1 .= reshape(range(0, 1, length = h), 1, :, 1, 1)
    # (Constant on image rows)

    imgs2 = CuArray{Float32}(undef, c, h, w, b)
    imgs2 .= reshape(range(0, 1, length = w), 1, 1, :, 1)
    # (Constant on image columns)

    ssims = CUDA.zeros(b)
    dL_dimgs1 = similar(imgs1)
    ssim_with_gradient!(ssims, dL_dimgs1, imgs1, imgs2, nothing, false)

    @test ssims ≈ ssim(imgs1, imgs2)
    @test dL_dimgs1 ≈ ssim_gradient(imgs1, imgs2)
    @test ssims ≈ ssim_with_gradient(imgs1, imgs2)[1]
    @test dL_dimgs1 ≈ ssim_with_gradient(imgs1, imgs2)[2]
    
    ssims_b = similar(ssims)
    ssim!(ssims_b, imgs1, imgs2)
    @test ssims ≈ ssims_b

    dL_dimgs1_b = similar(dL_dimgs1)
    ssim_gradient!(dL_dimgs1_b, imgs1, imgs2, nothing)
    @test dL_dimgs1 ≈ dL_dimgs1_b
end