@testset "autodiff" begin
    h, w = 200, 300

    img1 = CuArray{Float32}(undef, h, w)
    img1 .= range(0, 1, length = h)
    img2 = similar(img1)
    img2 .= range(0, 1, length = w)'

    zs, (zg,) = Zygote.withgradient(img1 -> ssim(img1, img2), img1)
    fs, fg = ssim_with_gradient(img1, img2)
    @test zs ≈ fs
    @test zg ≈ fg

    gimg1 = colorview(Gray, img1)
    gimg2 = colorview(Gray, img2)
    zs, (zg,) = Zygote.withgradient(gimg1 -> ssim(gimg1, gimg2), gimg1)
    fs, fg = ssim_with_gradient(gimg1, gimg2)
    @test zs ≈ fs
    @test zg ≈ fg  # in particular, Gray{Float32}

    imgs1 = reshape(img1, 1, size(img1)..., 1)  # explicit channels and batch axes
    imgs2 = reshape(img2, 1, size(img2)..., 1)
    zd, (zg,) = Zygote.withgradient(imgs1 -> sum(dssim(imgs1, imgs2, true)), imgs1)
    # sum as we need a scalar output, not a singleton
    fd, fg = dssim_with_gradient(imgs1, imgs2, true)
    @test zd ≈ sum(fd)
    @test zg ≈ fg 
end