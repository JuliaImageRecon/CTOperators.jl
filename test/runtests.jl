# Test items (TestItemRunner): run all with `Pkg.test()`, a selection by tags with e.g.
# `Pkg.test(test_args=["geometry"])`, single items from VS Code. Items tagged :gpu run on Array here
# and on JLArrays or a GPU device with `julia test/gpu/run.jl <backend>`.
#
# Tags: :image, :geometry, :operators, :siddon, :chainrules, :gpu, :aqua

using TestItemRunner

const TAGS = Symbol.(ARGS)

@run_package_tests filter = ti -> isempty(TAGS) || any(in(ti.tags), TAGS)
