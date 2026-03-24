@testset "Single pixel tests" begin
    R = FastCUDASSIM.RADIUS
    K = FastCUDASSIM.KERNEL
    W = K[R + 1]^2  # Central weight in 2D kernel

    # (This of course all assumes my manual calculations are correct.)
    function ssim_px(px_val_a, px_val_b)
        # After convolving img_a, img_a .^ 2 with the 2D kernel we have 
        #   conv_a = px_val_a * W
        #   conv_a_sq = px_val_a^2 * W
        # due to zero-padding, resulting in
        #   var_a = conv_a_sq - conv_a^2 = px_val_a^2 * (W - W^2).
        # Plugging all such values into the SSIM formula yields
        x = 2 * px_val_a * px_val_b * W^2 + C1
        y = 2 * px_val_a * px_val_b * (W - W^2) + C2
        z = (px_val_a^2 + px_val_b^2) * W^2 + C1
        t = (px_val_a^2 + px_val_b^2) * (W - W^2) + C2
        s = x * y / (z * t)

        dx = 2 * px_val_b * W^2  # dx_dpx_val_a
        dy = 2 * px_val_b * (W - W^2)
        dz = 2 * px_val_a * W^2
        dt = 2 * px_val_a * (W - W^2)
        ds = ((dx * y + x * dy) * z * t - x * y * (dz * t + z * dt)) / (z * t)^2

        return s, ds
    end
    # Sanity check's sanity check:
    #   s(a) = ssim_px(a, 0) = C1 * C2 / ((a^2 * W^2 + C1) * (a^2 * (W - W^2) + C2))
    #       (as conv_0 == conv_0_sq == var_0 == cov_a0 == 0; conv_a == conv_a_sq == W)
    #   s'(a) = - C1 * C2 * (4 * a^3 * W^2 * (W - W^2) 
    #                           + 2 * a * (W - W^2) * C1 + 2 * a * W^2 * C2) / 
    #               ((a^2 * W^2 + C1) * (a^2 * (W - W^2) + C2))^2  
    @test all(
        ssim_px(1.f0, 0.f0) .≈ (
            C1 * C2 / ((W^2 + C1) * (W - W^2 + C2)),
            -C1 * C2 * (4 * W^2 * (W - W^2) + 2 * (W - W^2) * C1 + 2 * W^2 * C2) /
                ((W^2 + C1) * (W - W^2 + C2))^2
        )
    )

    function ssim_px_test(px_val_a, px_val_b)
        img_a = CuArray([px_val_a;;;;])
        img_b = CuArray([px_val_b;;;;])

        s, g = ssim_with_gradient(img_a, img_b)
        @allowscalar begin
            s = s[]
            g = g[]
        end
        all(ssim_px(px_val_a, px_val_b) .≈ (s, g))
    end

    @test ssim_px_test(0.f0, 0.1f0)
    @test ssim_px_test(0.2f0, 0.5f0)
    @test ssim_px_test(inv(Float32(π)), inv(Float32(ℯ)))
end