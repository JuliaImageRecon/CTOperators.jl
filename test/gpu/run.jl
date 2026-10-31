# The test items tagged :gpu on a GPU backend: `julia test/gpu/run.jl <backend>` with backend
# jlarrays (the GPU code path on the CPU), cuda, rocm or metal. Runs in a temporary environment with
# CTOperators from this repository, its test dependencies and the backend package, so the GPU
# packages never become test dependencies. JLArrays is not one either: all its releases (up to
# 0.4.2) restrict KernelAbstractions to 0.9, which would keep `Pkg.test()` from testing
# KernelAbstractions 0.10.

using Pkg, TOML

const BACKENDS = Dict("jlarrays" => "JLArrays", "cuda" => "CUDA", "rocm" => "AMDGPU", "metal" => "Metal")
length(ARGS) == 1 && haskey(BACKENDS, ARGS[1]) ||
  error("usage: julia test/gpu/run.jl <backend>, backend one of $(join(keys(BACKENDS), ", "))")

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
# the test items load the package's dependencies and its test targets
const PROJECT = TOML.parsefile(joinpath(ROOT, "Project.toml"))
const TESTDEPS = [collect(keys(PROJECT["deps"])); PROJECT["targets"]["test"]]

Pkg.activate(; temp=true)
Pkg.develop(path=ROOT)
Pkg.add([BACKENDS[ARGS[1]]; TESTDEPS])

ENV["CTOPERATORS_TEST_DEVICE"] = ARGS[1]
using TestItemRunner
TestItemRunner.run_tests(ROOT; filter=ti -> :gpu in ti.tags)
