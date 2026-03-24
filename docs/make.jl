using Documenter, FastCUDASSIM


PAGE_NAMES_AND_FILES = [
    "Home" => "index.md",
    "Benchmarks" => "benchmarks.md",
    "Input and output formats" => "formats.md",
    "Automatic differentiation" => "autodiff.md",
    "Precompilation" => "precompilation.md",
    "Mathematics" => "math.md",
    "Reference" => "reference.md"
]

makedocs(
    sitename = "FastCUDASSIM.jl",
    pages = PAGE_NAMES_AND_FILES,
    format = Documenter.HTML(;
        mathengine = Documenter.KaTeX(
            Dict(
                :macros => Dict(
                    raw"\vect" => raw"\boldsymbol{#1}",
                    raw"\matr" => raw"\boldsymbol{#1}",
                    raw"\Var" => raw"\operatorname{Var}",
                    raw"\Cov" => raw"\operatorname{Cov}",
                    raw"\E" => raw"\operatorname{E}",
                    raw"\SSIM" => raw"\operatorname{SSIM}",
                )
            )
        )
    )
)


deploydocs(
    repo = "github.com/LaurensDiels/FastCUDASSIM.jl.git",
)