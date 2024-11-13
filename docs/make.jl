using Documenter
using DocumenterCitations
using MethodOfMoments

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"), style=:authoryear)

makedocs(
    modules = [MethodOfMoments],
    format = Documenter.HTML(
        canonical = "https://junyuan-chen.github.io/MethodOfMoments.jl/stable/",
        prettyurls = get(ENV, "CI", nothing) == "true",
        edit_link = "main"
    ),
    sitename = "MethodOfMoments.jl",
    authors = "Junyuan Chen",
    pages = [
        "Home" => "index.md",
        "Manual" => [
            "Generalized Method of Moments" => "man/nonlineargmm.md",
            "Linear GMM" => "man/lineargmm.md",
            "Bayesian Quasi-Likelihood" => "man/bayesian.md"
            ],
        "References" => "refs.md"
    ],
    workdir = joinpath(@__DIR__, ".."),
    checkdocs = :none,
    plugins = [bib]
)

deploydocs(
    repo = "github.com/junyuan-chen/MethodOfMoments.jl.git",
    devbranch = "main"
)
