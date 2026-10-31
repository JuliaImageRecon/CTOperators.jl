# Tests on a GPU device: `julia test/gpu/run.jl <backend>` with backend cuda, rocm or metal.
# Runs in a temporary environment with CTOperators from this repository and the backend package,
# so the GPU packages never become test dependencies of CTOperators.

using Pkg

const BACKENDS = Dict("cuda" => "CUDA", "rocm" => "AMDGPU", "metal" => "Metal")
length(ARGS) == 1 && haskey(BACKENDS, ARGS[1]) ||
  error("usage: julia test/gpu/run.jl <backend>, backend one of $(join(keys(BACKENDS), ", "))")

Pkg.activate(; temp=true)
Pkg.develop(path=joinpath(@__DIR__, "..", ".."))
Pkg.add([BACKENDS[ARGS[1]], "Adapt", "Test"])

include(joinpath(@__DIR__, ARGS[1] * ".jl"))
