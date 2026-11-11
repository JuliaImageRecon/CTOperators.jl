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
  pagesonly=true,
  checkdocs=:exports,
  authors="Niklas Hackelberg and contributors",
  repo=Remotes.GitHub("JuliaImageRecon", "CTOperators.jl"),
  format=Documenter.HTML(;
    prettyurls=get(ENV, "CI", "false") == "true",   # also in utils/bonito_page.jl
    canonical="https://JuliaImageRecon.github.io/CTOperators.jl",
    edit_link="main",
    collapselevel=1,   # the navigation shows the sections, collapsed
  ),
  # Diátaxis: tutorials to learn, how-to guides for tasks, explanation for understanding,
  # reference for looking things up
  pages=[
    "Home" => "index.md",
    "Getting Started" => "generated/examples/getting_started.md",
    "Tutorials" => [
      "Parallel Beam in 2D" => "generated/tutorials/parallel_beam.md",
      "Fan Beam with a Curved Detector" => "generated/tutorials/fan_beam.md",
      "Circular Cone-Beam CT" => "generated/tutorials/cone_beam.md",
      "Helical Cone-Beam CT" => "generated/tutorials/helical.md",
      "Sparse Views and Limited Angles" => "generated/tutorials/sparse_views.md",
      "Tomosynthesis" => "generated/tutorials/tomosynthesis.md",
    ],
    "How to" => [
      "Geometries" => [
        "Set Up a Geometry" => "generated/howto/geometries.md",
        "Use Geometries of Sinograms.jl" => "generated/howto/sinograms.md",
      ],
      "Reconstruction" => [
        "Ordered Subsets and Normalization" => "generated/howto/subsets.md",
        "Use Masks" => "generated/howto/masks.md",
        "Use Several Rays per Cell" => "generated/howto/multiray.md",
        "Differentiate through a Projector" => "generated/howto/differentiation.md",
      ],
      "Devices" => [
        "Run on a GPU" => "generated/howto/gpu.md",
        "Choose the Precision" => "generated/howto/precision.md",
      ],
      "Custom Components" => [
        "Write a Projector" => "generated/howto/write_projector.md",
        "Write a Sampling Strategy" => "generated/howto/write_sampling.md",
      ],
    ],
    "Explanation" => [
      "Scans" => [
        "Parameters of a Scan" => "generated/explanation/scan_parameters.md",
        "Coordinates and Conventions" => "generated/explanation/conventions.md",
      ],
      "Projection Methods" => [
        "System Matrix" => "generated/explanation/system_matrix.md",
        "Siddon's Method" => "generated/explanation/siddon.md",
        "Joseph's Method" => "generated/explanation/joseph.md",
        "Several Rays per Cell" => "generated/explanation/multiray.md",
        "Distance-Driven Method" => "generated/explanation/distancedriven.md",
        "Voxel-Driven Method" => "generated/explanation/voxeldriven.md",
        "Comparing Projectors" => "generated/explanation/comparing.md",
      ],
      "Operators" => [
        "Masks" => "generated/explanation/masks.md",
        "Normal Operator" => "generated/explanation/normal.md",
        "Workgroups and Ray Order on GPUs" => "generated/explanation/launch.md",
      ],
    ],
    "Reference" => [
      "Images" => "api/images.md",
      "Projection Geometries" => "api/geometries.md",
      "Operators" => "api/operators.md",
      "Projectors" => "api/projectors.md",
      "Internals" => "api/internals.md",
    ],
  ],
)

deploydocs(repo="github.com/JuliaImageRecon/CTOperators.jl.git", devbranch="main", push_preview=true)
