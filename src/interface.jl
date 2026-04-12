# User-friendly (or at least friendlier) wrappings of the core _ssim_fwd_bwd!.

"""
    ssim_with_gradient!(
        ssims, dL_dimgs1, 
        imgs1, imgs2,
        N_dssims_dQMP,
        should_zero = true
    )
    -> (ssims, dL_dimgs1)

Compute the SSIMs between two image batches, as well as the gradients with respect to the
first one.

See the module documentation for the input and output formats.

# Arguments
- `ssims`: The (output) `CuVector` which will contain the SSIM values.
- `dL_dimgs1`: The (output) gradients of `imgs1` with respect to the SSIM value L.
- `imgs1`: The first image batch, with respect to which we compute the gradients.
- `imgs2`: The second image batch.
- `N_dssims_dQMP`: Internal buffer for intermediate gradients.
- `should_zero`: To get the correct SSIM value we need `ssims` to start with values of `0`.
    When set to `true` we will perform this zeroing inside the method. Otherwise it is the
    responsibility of the caller to have done this in advance.
    Defaults to `true`.

# Returns
`(ssims, dL_dimgs1)`
"""
function ssim_with_gradient!(
    ssims, dL_dimgs1, 
    imgs1, imgs2,
    N_dssims_dQMP,
    should_zero = true
)
    return _ssim_fwd_bwd!(
        ssims, dL_dimgs1,
        imgs1, imgs2,
        N_dssims_dQMP,
        should_zero,
        Val(false), Val(false),
        false, false  # (not used)
    )
end


"""
    dssim_with_gradient!(
        dssims, dL_dimgs1, 
        imgs1, imgs2,
        N_dssims_dQMP,
        should_zero = true,
        divide_by_two_in_dssim = true
    )
    -> (dssims, dL_dimgs1)

Compute the DSSIMs between two image batches, as well as the gradients with respect to the
first one.

Like [`ssim_with_gradient!`](@ref), but for the structural dissimilarity DSSIM = 1 - SSIM
when `!divide_by_two_in_dssim` (by default), or DSSIM = (1 - SSIM) / 2.
"""
function dssim_with_gradient!(
    dssims, dL_dimgs1, 
    imgs1, imgs2,
    N_dssims_dQMP,
    should_zero = true,
    divide_by_two_in_dssim = false
)
    return _ssim_fwd_bwd!(
        dssims, dL_dimgs1,
        imgs1, imgs2,
        N_dssims_dQMP,
        should_zero,
        Val(false), Val(false),
        true, divide_by_two_in_dssim
    )
end


"""
    ssim_gradient!(
        dL_dimgs1, 
        imgs1, imgs2,
        N_dssims_dQMP,
    )
    -> dL_dimgs1

Compute the gradients of the SSIM between two image batches with respect to the first one.

Like [`ssim_with_gradient!`](@ref), but without the computation of the SSIM itself.
"""
function ssim_gradient!(
    dL_dimgs1, 
    imgs1, imgs2,
    N_dssims_dQMP,
)
    return _ssim_fwd_bwd!(
        nothing, dL_dimgs1,
        imgs1, imgs2,
        N_dssims_dQMP,
        false,  # not used
        Val(true), Val(false),
        false, false
    )[2]
end


"""
    dssim_gradient!(
        dL_dimgs1, 
        imgs1, imgs2,
        N_dssims_dQMP,
        divide_by_two_in_dssim = false
    )
    -> dL_dimgs1

Compute the gradients of the DSSIMs between two image batches with respect to the first one.

Like [`dssim_with_gradient!`](@ref), but without the computation of the DSSIM itself, and
like [`ssim_gradient!`](@ref), but for the DSSIM = 1 - SSIM when `!divide_by_two_in_dssim`
(the default) or DSSIM = (1 - SSIM) / 2 otherwise.
"""
function dssim_gradient!(
    dL_dimgs1, 
    imgs1, imgs2,
    N_dssims_dQMP,
    divide_by_two_in_dssim = false
)
    return _ssim_fwd_bwd!(
        nothing, dL_dimgs1,
        imgs1, imgs2,
        N_dssims_dQMP,
        false,  # not used
        Val(true), Val(false),
        true, divide_by_two_in_dssim
    )[2]
end


"""
    ssim_with_gradient(imgs1, imgs2)

Compute the SSIMs between two image batches, as well as the gradients with respect to the
first one.

This method allocates (and frees) an internal buffer and of course also allocates the 
output gradient 'image' batch. See [`ssim_with_gradient!`](@ref) for the non-allocating
version.

See the module documentation for expected formats for the inputs and output.

# Arguments
- `imgs1`: The first image batch.
- `imgs2`: The second image batch.

# Returns
1. The SSIM values.
2. The gradients of these values with respect to `imgs1`.
"""
function ssim_with_gradient(
    imgs1, imgs2,
)
    return ssim_with_gradient!(
        nothing, nothing,  # will be allocated and returned
        imgs1, imgs2,
        nothing,           # will be allocated and freed
        false
    )
end


"""
    dssim_with_gradient(
        imgs1, imgs2, 
        divide_by_two_in_dssim = false
    )

Compute the DSSIMs between two image batches, as well as their gradients with respect to the
first one.

Like [`ssim_with_gradient`](@ref), but for the structural dissimilarity DSSIM = 1 - SSIM
when `!divide_by_two_in_dssim` (the default), or DSSIM = (1 - SSIM) / 2 otherwise. See also
[`dssim_with_gradient!`](@ref) for the non-allocating version.
"""
function dssim_with_gradient(
    imgs1, imgs2, divide_by_two_in_dssim = false
)
    return dssim_with_gradient!(
        nothing, nothing,  # will be allocated and returned
        imgs1, imgs2,
        nothing,           # will be allocated and freed
        false, divide_by_two_in_dssim
    )
end


"""
    ssim_gradient(imgs1, imgs2) 

Compute the gradients of the SSIMs between two image batches with respect to the first. 

Like [`ssim_with_gradient`](@ref), but does not compute the SSIM itself.

This method allocates (and frees) an internal buffer and of course also allocates the 
output gradient 'image' batch. See [`ssim_gradient!`](@ref) for the non-allocating version.
"""
function ssim_gradient(
    imgs1, imgs2,
)
    return ssim_gradient!(
        nothing,  # will be allocated and returned
        imgs1, imgs2,
        nothing,  # will be allocated and freed
    )
end


"""
    dssim_gradient(
        imgs1, imgs2, 
        divide_by_two_in_dssim = false
    ) 

Compute the gradients of the DSSIMs between two image batches with respect to the first. 

Like [`dssim_with_gradient`](@ref), but does not compute the DSSIM itself.

Similar to [`ssim_gradient`](@ref), but for the DSSIM = 1 - SSIM when 
`!divide_by_two_in_dssim` (the default), or DSSIM = (1 - SSIM) / 2 otherwise.
This method allocates (and frees) an internal buffer and of course also allocates the 
output gradient 'image' batch. See [`dssim_gradient!`](@ref) for the non-allocating version.
"""
function dssim_gradient(
    imgs1, imgs2, divide_by_two_in_dssim = false
)
    return dssim_gradient!(
        nothing,  # will be allocated and returned
        imgs1, imgs2,
        nothing,  # will be allocated and freed
        divide_by_two_in_dssim
    )
end


"""
    ssim!(
        ssims, 
        imgs1, imgs2, 
        should_zero = true
    )
    -> ssims

Compute the SSIMs between two image batches.

Like [`ssim_with_gradient!`](@ref) but does not compute the gradient. See this function for
the meaning of the arguments.
"""
function ssim!(
    ssims, 
    imgs1, imgs2, 
    should_zero = true
)
    return _ssim_fwd_bwd!(
        ssims, nothing,  # (will not be allocated)
        imgs1, imgs2,
        nothing,         # (will not be allocated)
        should_zero,
        Val(false), Val(true),
        false, false  # (not used)
    )[1]
end


"""
    dssim!(
        dssims, 
        imgs1, imgs2, 
        should_zero = true,
        divide_by_two_in_dssim = false
    )
    -> dssims

Compute the DSSIMs between two image batches.

Like [`ssim!`](@ref), but for the structural dissimilarity DSSIM = 1 - SSIM if 
`!divide_by_two_in_dssim` (the default) or DSSIM = (1 - SSIM) / 2 otherwise. 

See also [`dssim_with_gradient!`](@ref).
"""
function dssim!(
    dssims, 
    imgs1, imgs2, 
    should_zero = true,
    divide_by_two_in_dssim = false
)
    return _ssim_fwd_bwd!(
        dssims, nothing,  # (will not be allocated)
        imgs1, imgs2,
        nothing,          # (will not be allocated)
        should_zero,
        Val(false), Val(true),
        true, divide_by_two_in_dssim
    )[1]
end


"""
    ssim(imgs1, imgs2) 

Compute the Structural Similarity Index Measure (SSIM) values between two image batches.

Like [`ssim_with_gradient`](@ref), but without the gradient.

See the module documentation for the input and output formats. Also see [`ssim!`](@ref)
for the non-allocating version.
"""
function ssim(imgs1, imgs2)
    return ssim!(nothing, imgs1, imgs2, false)
end


"""
    dssim(
        imgs1, imgs2, 
        divide_by_two_in_dssim = false
    )

Compute the Structural DisSimilarity Index Measure (DSSIM) values between two image batches.

Like [`dssim_with_gradient`](@ref), but without the gradient.

See also [`ssim`](@ref), and [`dssim!`](@ref) for the non-allocating version of `dssim`. 
"""
function dssim(imgs1, imgs2, divide_by_two_in_dssim = false)
    return dssim!(nothing, imgs1, imgs2, false, divide_by_two_in_dssim)
end