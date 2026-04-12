@enum PrecompilationLevel off fast full

"""
Completely precompiling FastCUDASSIM can take a while, so we use Preferences.jl to provide
options precompiling the methods for fewer argument type combinations. The values of this
enum are $(join((`$pcl` for pcl in instances(PrecompilationLevel)), ", ", " and ")).
"""
PrecompilationLevel

"""
Read in the Preferences.jl "precompilation" value for our package, if present, defaulting
to "fast" if not.
"""
function _get_precompilation_level()
    precompilation_level_symbol = Symbol(@load_preference("precompilation", "fast"))
    precompilation_level = nothing
    for pcl in instances(PrecompilationLevel)
        if Symbol(pcl) === precompilation_level_symbol
            precompilation_level = pcl
            break
        end
    end
    if isnothing(precompilation_level)
        @warn "Invalid precompilation level provided: \"$precompilation_level_symbol\". \
               The valid options are \
               $(join(
                    ("\"$pcl\"" for pcl in instances(PrecompilationLevel)), 
                    ", ", " and ")
               ) \
               Using \"fast\" instead."
        precompilation_level = fast
    end
    return precompilation_level
end

"""
    _prepare_arguments(precompilation_level)

Prepare arguments of many different types and `ndims` to use in `ssim_with_gradient!` etc.
For `precompilation_level ===`
* `off`: no types; we return an empty `Vector`
* `fast`: `Float32` (1 or 3 channels), `Gray{N0f8}`, `RGB{N0f8}`
* `full`: `Float16`, `Float32`, `Float64`, `N0f8` (all 1 up to 4 channels), as well as
    `Gray` and `RGB` variants.
The elements in the returned `Vector` (when non-empty) are `Tuple`s of arguments of the same
underlying type. These arguments are `imgs1`, `imgs2`, `ssims`, `dL_dimgs1`, the
intermediate gradients buffer `N_dssims_dQMP` and finally two flags `only_run_ssim` and
`only_run_forward`. The former is used to indicate whether we should limit ourselves to
precompiling `ssim` and `dssim`. This is the case in `fast` mode for `Gray` and `RGB`.
The latter flag is used for skipping the methods involving the backward pass (so here
`ssim!` is fine, but `ssim_with_gradient` is not). For certain configurations like 4 channel
`Float64`, the backward pass requires too much shared memory. 
When the buffer cannot exist for the `Vector` entry type (e.g. for `RGB`), we create empty
(`size(...) == (0,)`) `CuArray`s.
"""
function _prepare_arguments(precompilation_level)
    precompilation_level === off && return []

    h, w, b = 100, 200, 2
    base_types = (Float16, Float32, Float64, N0f8)
    color_types = ((Gray{base_type} for base_type in base_types)..., 
                    (RGB{base_type} for base_type in base_types)...)
    all_types_and_sizes = []
    # Many combinations below will end up with the same types in the kernels, so in
    # principle we could try to filter the duplicates out. But the point of FastCUDASSIM
    # is that it's fast, so we don't really have to worry about running already compiled
    # kernels multiple times.

    for T in base_types
        push!(all_types_and_sizes, (T, (h, w), (h, 3, w)))             
        # type, size(imgs1), size(N_dssims_dQMP)

        for c = 1:4
            push!(all_types_and_sizes, (T, (c, h, w), (h, 3, c, w)))
            push!(all_types_and_sizes, (T, (c, h, w, b), (h, 3, c, w, b)))
        end
    end

    for T in color_types
        push!(all_types_and_sizes, (T, (h, w), (h, 3, w)))             
        push!(all_types_and_sizes, (T, (h, w, b), (0,)))
        # Buffers are not allowed to be e.g. RGB{Float32}, as this is in (c, h, w, b)
        # memory order instead of the expected (h, c, w, b)
    end

    if precompilation_level === fast
        types_and_sizes = filter(all_types_and_sizes) do x
            x[1] == Float32 && x[2][1] ∈ (1, 3)  ||  x[1] ∈ (Gray{N0f8}, RGB{N0f8})
        end
    elseif precompilation_level === full
        types_and_sizes = all_types_and_sizes
    end

    arguments = map(types_and_sizes) do (T, sz, buff_sz)
        S = eltype(T)  # S in base_types, also when T in color_types
        # e.g. T === RGB{N0f8} --> S === N0f8; T === Float64 --> S === Float64
        # We'll check later whether e.g. eltype(ssims) === S makes sense (not for N0f8)
        bb = sz[end] == b ? b : 1  # explicit batch size ↦ b, implicit ↦ 1
        only_run_ssim =  precompilation_level == fast && T !== Float32
        # and dssim, but not any of the mutating or gradient-computing methods 
        only_run_forward = only_run_ssim ||
            T === Float64 && length(sz) >= 3 && sz[1] >= 4  # 4 or more channels
        return (
            CUDA.rand(T, sz),                                # imgs1
            CUDA.rand(T, sz),                                # imgs2
            CuArray{S}(undef, bb),                           # ssims
            CuArray{T}(undef, sz),                           # dL_dimgs1
            CuArray{T}(undef, buff_sz),                      # N_dssims_dQMP
            only_run_ssim,
            only_run_forward
        )
    end
    # (Note that we are not precompiling for type mixes, like
    #   img1 isa CuArray{Float32}) && img2 isa CuMatrix{RGB{N0f8}} .)
    return arguments
end

"""
    _precompile(arguments)

Run all exported `FastCUDASSIM` methods using the `arguments` provided (from
`_prepare_arguments`).
"""
function _precompile(arguments)
    for (imgs1, imgs2,
         ssims, dL_dimgs1, 
         N_dssims_dQMP,
         only_run_ssim,
         only_run_forward
         ) in arguments

        T = eltype(imgs1); S = eltype(ssims)

        ssim(imgs1, imgs2)
        dssim(imgs1, imgs2)

        only_run_ssim && continue

        ssim!(nothing, imgs1, imgs2)
        dssim!(nothing, imgs1, imgs2)

        if S <: AbstractFloat
            # Don't attempt to output ssims in e.g. N0f8 format
            ssim!(ssims, imgs1, imgs2)
            dssim!(ssims, imgs1, imgs2)
        end

        only_run_forward && continue

        ssim_gradient(imgs1, imgs2)
        dssim_gradient(imgs1, imgs2)

        ssim_gradient!(nothing, imgs1, imgs2, nothing)
        dssim_gradient!(nothing, imgs1, imgs2, nothing)

        if S <: AbstractFloat
            # e.g. okay for T === Float16 and RGB{Float32}, but not for N0f8 
            # or RGB{N0f8}
            ssim_gradient!(dL_dimgs1, imgs1, imgs2, nothing)
            dssim_gradient!(dL_dimgs1, imgs1, imgs2, nothing) 

            if T <: AbstractFloat
                # We don't support e.g. RGB{Float32} for the buffers, as they are
                # not in the correct memory order.
                ssim_gradient!(dL_dimgs1, imgs1, imgs2, N_dssims_dQMP)
                dssim_gradient!(dL_dimgs1, imgs1, imgs2, N_dssims_dQMP)
            end
        end

        ssim_with_gradient(imgs1, imgs2)
        dssim_with_gradient(imgs1, imgs2)

        ssim_with_gradient!(nothing, nothing, imgs1, imgs2, nothing)
        dssim_with_gradient!(nothing, nothing, imgs1, imgs2, nothing)

        if S <: AbstractFloat
            ssim_with_gradient!(ssims, dL_dimgs1, imgs1, imgs2, nothing)
            dssim_with_gradient!(ssims, dL_dimgs1, imgs1, imgs2, nothing)

            if T <: AbstractFloat
                ssim_with_gradient!(ssims, dL_dimgs1, imgs1, imgs2, N_dssims_dQMP)
                dssim_with_gradient!(ssims, dL_dimgs1, imgs1, imgs2, N_dssims_dQMP)
            end
        end
    end
end



@setup_workload begin
    precompilation_level = _get_precompilation_level()
    arguments = _prepare_arguments(precompilation_level)
    @compile_workload begin
        Base.with_logger(Base.CoreLogging.ConsoleLogger(Base.CoreLogging.Error)) do
            # Ignore the shared memory warning for 4-channel Float64
            _precompile(arguments)
        end
    end
end