@testset "eltypes" begin
    img_mandrill_RGBN0f8 = cu(testimage("mandrill"))  # 512 x 512
    img_mandrill_Float32 = Float32.(channelview(img_mandrill_RGBN0f8))
    img_mandrill_Float64 = Float64.(channelview(img_mandrill_RGBN0f8))
    img_mandrill_GrayFloat16 = Gray{Float16}.(img_mandrill_RGBN0f8)

    img_airplane_RGBN0f8 = cu(testimage("airplaneF16"))  # also 512 x 512
    img_airplane_Float32 = Float32.(channelview(img_airplane_RGBN0f8))
    img_airplane_Float64 = Float64.(channelview(img_airplane_RGBN0f8))
    img_airplane_GrayFloat16 = Gray{Float16}.(img_airplane_RGBN0f8)

    @test typeof(ssim(img_mandrill_RGBN0f8, img_airplane_RGBN0f8)) === Float32
    @test typeof(ssim(img_mandrill_Float32, img_airplane_Float64)) === Float64
    @test typeof(ssim(img_mandrill_GrayFloat16, img_airplane_GrayFloat16)) == Float32

    @test ssim(img_mandrill_RGBN0f8, img_airplane_RGBN0f8) ≈ 
        ssim(img_mandrill_Float32, img_airplane_Float32) ≈
        ssim(img_mandrill_Float64, img_airplane_Float64)
    # The Gray{Float16} version will be a bit different, as all others are RGB

    @test eltype(
        ssim_gradient(img_mandrill_RGBN0f8, img_airplane_RGBN0f8)
    ) === RGB{Float32}
    @test eltype(ssim_gradient(img_mandrill_Float64, img_airplane_RGBN0f8)) === Float64
    @test eltype(ssim_gradient(img_airplane_RGBN0f8, img_mandrill_Float64)) === RGB{Float64}
    # Use the format of the first image (up to promotion)

    dL_dimg_mandrill = CuArray{Float16}(undef, size(img_mandrill_Float32)...)
    dssim_gradient!(
        dL_dimg_mandrill, 
        img_mandrill_Float32, img_airplane_Float32, 
        nothing, nothing, nothing
    )  # The first test is that this does not error
    @test maximum(abs.(
        dL_dimg_mandrill .- dssim_gradient(img_mandrill_Float32, img_airplane_Float32))
        ) < 1e-3
    # The second that the Float16 version is close enough to the Float32 version
end