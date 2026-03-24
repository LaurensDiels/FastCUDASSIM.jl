@testset "dssim" begin
    c, h, w, b = 3, 200, 300, 1
    imgs1 = CuArray{Float32}(undef, c, h, w, b)
    imgs1 .= reshape(range(0, 1, length = h), 1, :, 1, 1)

    imgs2 = CuArray{Float32}(undef, c, h, w, b)
    imgs2 .= reshape(range(0, 1, length = w), 1, 1, :, 1)
    
    @test dssim(imgs1, imgs2) ≈ 1.f0 .- ssim(imgs1, imgs2)
    @test dssim(imgs1, imgs2, true) ≈ (1.f0 .- ssim(imgs1, imgs2)) ./ 2

    @test dssim_gradient(imgs1, imgs2) ≈ -ssim_gradient(imgs1, imgs2)
    @test dssim_gradient(imgs1, imgs2, true) ≈ -ssim_gradient(imgs1, imgs2) ./ 2
end