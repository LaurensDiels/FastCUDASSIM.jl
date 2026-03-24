@testset "Sanity checks" begin
    img0 = CUDA.zeros(Float32, 200, 300)
    img1 = CUDA.ones(Float32, 200, 300)
    @test ssim(img0, img0) ≈ 1
    @test ssim(img1, img1) ≈ 1

    @test isapprox(
        ssim(img0, img1), C1 / (1 + C1), 
        atol = 1e-5
    )  # Without padding we have equality in theory
    @test isapprox(
        ssim(img0, img1), C1 / (1 + C1),
        atol = 1e-4
    )  # C1 is small, so should be approximately 0
    @test dssim(img0, img1) ≈ 1
    @test dssim(img0, img1, true) ≈ 0.5
end