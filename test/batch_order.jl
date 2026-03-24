@testset "Batch order" begin
    c, h, w = 1, 200, 300

    img_a = CuArray{Float32}(undef, c, h, w)
    img_a .= reshape(range(0, 1, length = h), 1, :, 1)

    img_b = CuArray{Float32}(undef, c, h, w)
    img_b .= reshape(range(0, 1, length = w), 1, 1, :)

    imgs1 = cat(img_a, img_b, img_a, img_b; dims=4)  # b = 4
    imgs2 = cat(img_b, img_a, img_b, img_a; dims=4)

    ssims, grads = ssim_with_gradient(imgs1, imgs2)

    @test @allowscalar all(ssims .≈ ssims[1])
    # Test that 
    # 1) The ssim between img_a and img_b is the same in the first and third batch entry
    # 2) Same for img_b and img_a in the second and fourth
    # 3) ssim is symmetric (bonus test!)

    @test view(grads, :, :, :, 1) ≈ view(grads, :, :, :, 3)
    @test view(grads, :, :, :, 2) ≈ view(grads, :, :, :, 4)
end