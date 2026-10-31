using CTOperators
using Test

# CPU tests, then the KernelAbstractions code paths through JLArrays (GPU-style arrays whose
# kernels run on the CPU) and Aqua. Tests on GPU devices run separately: test/gpu/run.jl.
@testset "CTOperators" begin
  include("gpu/jlarrays.jl")
  include("aqua.jl")
end
