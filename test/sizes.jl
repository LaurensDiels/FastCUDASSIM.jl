@testset "sizes" begin
    # tests conversions.jl
    h, w = 200, 300

    img1_nn = CuArray{Float32}(undef, h, w)  # no channels, no batch
    img1_nn .= range(0, 1, length = h)
    img2_nn = similar(img1_nn)
    img2_nn .= range(0, 1, length = w)'
    s_nn, g_nn = ssim_with_gradient(img1_nn, img2_nn)
    @test s_nn isa Float32
    @test size(g_nn) == (h, w) == size(img1_nn)

    img1_gn = CuArray{Gray{Float32}}(undef, h, w)  # gray, no batch
    channelview(img1_gn) .= img1_nn  
    img2_gn = similar(img1_gn)
    channelview(img2_gn) .= img2_nn
    s_gn, g_gn = ssim_with_gradient(img1_gn, img2_gn)
    @test s_nn isa Float32
    @test size(g_gn) == (h, w) == size(img1_gn)

    img1_cn = CuArray{Float32}(undef, 1, h, w)  # channels, no batch
    img1_cn .= reshape(img1_nn, size(img1_cn)...)
    img2_cn = similar(img1_cn)
    img2_cn .= reshape(img2_nn, size(img2_cn)...)
    s_cn, g_cn = ssim_with_gradient(img1_cn, img2_cn)
    @test s_cn isa Float32
    @test size(g_cn) == (1, h, w) == size(img1_cn)

    img1_rn = CuArray{RGB{Float32}}(undef, h, w)  # rgb color, no batch
    channelview(img1_rn) .= img1_cn               # red == green == blue
    img2_rn = CuArray{RGB{Float32}}(undef, h, w)
    channelview(img2_rn) .= img2_cn
    s_rn, g_rn = ssim_with_gradient(img1_rn, img2_rn)
    @test s_rn isa Float32
    @test size(g_rn) == (h, w) == size(img1_rn)

    # Note that   no channels, batch   is not supported

    img1_gb = CuArray{Gray{Float32}}(undef, h, w, 1)  # gray, batch
    img1_gb .= img1_gn
    img2_gb = CuArray{Gray{Float32}}(undef, h, w, 1)
    img2_gb .= img2_gn
    s_gb, g_gb = ssim_with_gradient(img1_gb, img2_gb)
    @test s_gb isa CuArray{Float32} && size(s_gb) == (1,)
    @test size(g_gb) == (h, w, 1) == size(img1_gb)

    img1_cb = CuArray{Float32}(undef, 1, h, w, 1)  # channels, batch
    img1_cb .= reshape(img1_nn, size(img1_cb)...)
    img2_cb = similar(img1_cb)
    img2_cb .= reshape(img2_nn, size(img2_cb)...)
    s_cb, g_cb = ssim_with_gradient(img1_cb, img2_cb)
    @test s_cb isa CuArray{Float32} && size(s_cb) == (1,)
    @test size(g_cb) == (1, h, w, 1)

    img1_rb = CuArray{RGB{Float32}}(undef, h, w, 1)  # rgb, batch
    img1_rb .= img1_rn
    img2_rb = CuArray{RGB{Float32}}(undef, h, w, 1)
    img2_rb .= img2_rn
    s_rb, g_rb = ssim_with_gradient(img1_rb, img2_rb)
    @test s_rb isa CuArray{Float32} && size(s_rb) == (1,)
    @test size(g_rb) == (h, w, 1) == size(img1_rb)


    @test @allowscalar s_nn ≈ s_gn ≈ s_cn ≈ s_gb[] ≈ s_cb[]
    @test @allowscalar s_nn ≈ s_rn ≈ s_rb[] 
    # Because we simply average over the color channels
    

    # In-place test
    ssims = CUDA.zeros(1)
    dL_dimg1_cn = similar(img1_cn)
    N_dssims_dQMP_nn = CuArray{Float32}(undef, h, 3, w)
    s_ip, g_ip = ssim_with_gradient!(
        ssims, dL_dimg1_cn,
        img1_nn, img2_cb,
        N_dssims_dQMP_nn)
    # Test that this does not error (despite the total inconsistency of the formats)

    @test ssims == s_ip ≈ s_cb          # In particular, s_ip is not scalar
    @test dL_dimg1_cn == g_ip ≈ g_cn    #                g_ip is not in img1_nn-like format
end