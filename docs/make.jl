using Documenter, Literate, CTOperators
# Loaded before makedocs: loading Bonito inside an example would load its Documenter extension in the
# middle of the build, whose methods Documenter then cannot see (world age).
using Bonito, WGLMakie

# Examples are Literate scripts in literate/<section>/, rendered to src/generated/<section>/
# and run when the documentation is built.
const LITERATE = joinpath(@__DIR__, "literate")
const GENERATED = joinpath(@__DIR__, "src", "generated")
for dir in readdir(LITERATE), file in filter(endswith(".jl"), readdir(joinpath(LITERATE, dir)))
  Literate.markdown(joinpath(LITERATE, dir, file), joinpath(GENERATED, dir); documenter=true)
end

DocMeta.setdocmeta!(CTOperators, :DocTestSetup, :(using CTOperators); recursive=true)

makedocs(;
  modules=[CTOperators],
  sitename="CTOperators.jl",
  authors="Niklas Hackelberg and contributors",
  repo=Remotes.GitHub("JuliaImageRecon", "CTOperators.jl"),
  format=Documenter.HTML(;
    prettyurls=get(ENV, "CI", "false") == "true",   # also in utils/bonito_page.jl
    canonical="https://JuliaImageRecon.github.io/CTOperators.jl",
    edit_link="main",
  ),
  # Diátaxis: a tutorial to learn, how-to guides for tasks, explanation for understanding,
  # reference for looking things up
  pages=[
    "Home" => "index.md",
    "Getting Started" => "generated/examples/getting_started.md",
    "How-to Guides" => [
      "Set up a Geometry" => "generated/howto/geometries.md",
      "Run on a GPU" => "generated/howto/gpu.md",
      "Ordered Subsets and Normalisation" => "generated/howto/subsets.md",
      "Differentiate through a Projector" => "generated/howto/differentiation.md",
    ],
    "Explanation" => [
      "Coordinates and Conventions" => "generated/explanation/conventions.md",
      "Siddon's Method" => "generated/explanation/siddon.md",
      "Joseph's Method" => "generated/explanation/joseph.md",
    ],
    "Reference" => [
      "API" => "api.md",
    ],
  ],
)

deploydocs(repo="github.com/JuliaImageRecon/CTOperators.jl.git", devbranch="main", push_preview=true)
