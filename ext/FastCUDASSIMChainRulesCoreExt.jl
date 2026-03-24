"""
Adds `rrule`s for `ssim` and `dssim`.
"""
module FastCUDASSIMChainRulesCoreExt

using FastCUDASSIM
using ChainRulesCore

"""
    ChainRulesCore.rrule(::typeof(ssim), imgs1, imgs2)

The reverse-mode rule for the gradient of [`ssim`](@ref). Note that `imgs2` is considered
to be constant.
"""
function ChainRulesCore.rrule(
    ::typeof(ssim), 
    imgs1, imgs2
)
    ssims, dssims_dimgs1 = ssim_with_gradient(imgs1, imgs2)
    scalar_ssim = !isa(ssims, AbstractArray)

    function pullback(dL_dssims) 
        # L is a scalar (component of the final output) so dL_dssims is similar (in size)
        # to ssims
        
        # The chain rule product dL_dimgs1 = dL_dssims * dssims_dimgs1 occurs separately for
        # each image (as the different images within batch do not interact)
        dL_dimgs1 = dssims_dimgs1
        if scalar_ssim
            # dL_dssims is also a scalar
            dL_dimgs1 .*= dL_dssims
        else
            # Conceptually size(dL_dssims) = (b,)
            # Prepend ones in the size until we reach the same number of dimensions as imgs1
            dL_dimgs1 .*= reshape(dL_dssims, (1 for _ in 1 : ndims(imgs1)-1)..., :)
        end
        # We allocated dssims_dimgs1 ourselves in ssim_with_gradient, so it's fine to 
        # overwrite

        return NoTangent(), dL_dimgs1, NoTangent()
        # No gradient for ssim itself, or for imgs2
    end

    return ssims, pullback
end


"""
    ChainRulesCore.rrule(::typeof(dssim), imgs1, imgs2, divide_by_two_in_dssim = false)

The reverse-mode rule for the gradient of [`dssim`](@ref). Note that `imgs2` is considered
to be constant (as is `divide_by_two_in_dssim`).
"""
function ChainRulesCore.rrule(
    ::typeof(dssim), 
    imgs1, imgs2, divide_by_two_in_dssim = false
)
    # Completely similar to the ssim rrule
    dssims, ddssims_dimgs1 = dssim_with_gradient(imgs1, imgs2, divide_by_two_in_dssim)
    scalar_dssim = !isa(dssims, AbstractArray)

    function pullback(dL_ddssims) 
        dL_dimgs1 = ddssims_dimgs1
        if scalar_dssim
            dL_dimgs1 .*= dL_ddssims
        else
            dL_dimgs1 .*= reshape(dL_ddssims, (1 for _ in 1 : ndims(imgs1)-1)..., :)
        end
        return NoTangent(), dL_dimgs1, NoTangent(), NoTangent()  
        # Also no gradient for the flag
    end

    return dssims, pullback
end

end