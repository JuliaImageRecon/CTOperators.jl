using Documenter, Literate, CTOperators

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
  repo=Remotes.GitHub("nHackel", "CTOperators.jl"),
  format=Documenter.HTML(;
    prettyurls=get(ENV, "CI", "false") == "true",
    canonical="https://nHackel.github.io/CTOperators.jl",
    edit_link="main",
  ),
  # Diátaxis: a tutorial to learn, how-to guides for tasks, explanation for understanding,
  # reference for looking things up
  pages=[
    "Home" => "index.md",
    "Getting Started" => "generated/examples/getting_started.md",
    "Reference" => [
      "API" => "api.md",
    ],
  ],
)

deploydocs(repo="github.com/nHackel/CTOperators.jl.git", devbranch="main", push_preview=true)
