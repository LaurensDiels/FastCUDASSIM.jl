@testset "Comparison to fused_ssim.py" begin
    # Note: similar, but not identical to SSIMLoss.jl as that uses symmetric padding
    # (for crop = false), while we use zero-padding

    img_grass = cu(testimage("grass_512"))        # Gray{N0f8}, 512 x 512 (height x width)
    img_grass_he = cu(testimage("grass_he_512"))  # same

    s, g = ssim_with_gradient(img_grass, img_grass_he)
    @test s ≈ 0.919865  # SSIMLoss (crop = false): 0.91921055f0
    @test @allowscalar all(
        (g[1, 1].val     ≈ -5.686356f-6,
         g[100, 200].val ≈  4.541836f-6, 
         g[300, 150].val ≈  9.299385f-6, 
         g[512, 1].val   ≈ -5.252310f-7, 
         g[1, 512].val   ≈ -2.057583f-6)
    )
    # fused_ssim values via PythonCall in properly set up environment:
    #=
        ENV["JULIA_PYTHONCALL_EXE"] = ... (python / python.exe)
        using PythonCall
        using DLPack
        torch = pyimport("torch")
        fused_ssim = pyimport("fused_ssim" => "fused_ssim")
        
        pyimg_grass = DLPack.share(
            Float32.(channelview(img_grass)),
            torch.from_dlpack
        ).T.unsqueeze(0).unsqueeze(0).requires_grad_()  # 1 x 1 x 512 x 512
        # PyTorch uses batch size x channels x height x width in row-major format
        # Our column-major (channels x) height x width (x batch size) then becomes
        # (batch size x) width x height (x channels), so we have to permute.
        pyimg_grass_he = DLPack.share(
            Float32.(channelview(img_grass_he)),
            torch.from_dlpack
        ).T.unsqueeze(0).unsqueeze(0)
        s = fused_ssim(pyimg_grass, pyimg_grass_he) 
        s.backward()
        g = from_dlpack(pyimg_grass.grad[0, 0].T)
        # (Because the 2D Gaussian convolution kernel is symmetric, we would get the same
        #  result if we forgot all transposes.)
    =#
    
    img_lighthouse = cu(testimage("lighthouse"))  # RGB{N0f8}, 512 x 768
    img_monarch = cu(testimage("monarch_color"))

    s, g = ssim_with_gradient(img_lighthouse, img_monarch)
    @test s ≈ 0.31308  # SSIMLoss (crop = false): 0.3064171f0
    @test @allowscalar all(
        (red(g[1, 1])       ≈ -2.140367f-7,
         green(g[200, 500]) ≈  5.295946f-6, 
         blue(g[400, 300])  ≈  6.369429f-7, 
         red(g[256, 384])   ≈  1.874141f-6,
         green(g[512, 1])   ≈  1.053598f-6, 
         blue(g[1, 768])    ≈ -1.233621f-6
        )
    )
    # Compared to before, now use
    #
    #   pyimg_lighthouse = DLPack.share(
    #       Float32.(channelview(img_lighthouse)), 
    #       torch.from_dlpack
    #   ).permute((2, 1, 0)).unsqueeze(0).requires_grad_()
    #
    # and
    #   g = from_dlpack(pyimg_lighthouse.grad[0])   # size 3x512x768
end