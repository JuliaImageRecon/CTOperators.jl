using CTOperators
using Test

include("utils.jl")

# CPU tests, then the KernelAbstractions code paths through JLArrays (GPU-style arrays whose
# kernels run on the CPU) and Aqua. Tests on GPU devices run separately: test/gpu/run.jl.
@testset "CTOperators" begin
  include("image/voxelgrid.jl")
  include("geometry/geometry.jl")
  include("gpu/jlarrays.jl")
  include("aqua.jl")
end
