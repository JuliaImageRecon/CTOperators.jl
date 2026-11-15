# Forward and back projection of the VoxelDrivenProjector for layouts of its launch (runs of R voxels
# along x per thread) and the device's default, on the FDK benchmark problem: a circular cone-beam
# scan (360 views over one turn, 256 x 128 cells of 2 mm, 1 mm at the axis), 128^3 voxels of 1 mm,
# Float32, the default scale. Minimum of 5 calls after a first one.
#
#   julia --project=benchmark/envs/fdk -t 24 benchmark/voxeldriven.jl        # CPU, 24 threads
#   julia --project=benchmark/envs/fdk benchmark/voxeldriven.jl rocm         # AMD GPU
#
# Measure on a quiet machine. Results: benchmark/README.md.

using CTOperators, LinearAlgebra
using ImageGeoms: ImageGeom

const DEVICE = get(ARGS, 1, "cpu")
if DEVICE == "rocm"
  using AMDGPU
  storage, sync = ROCArray, AMDGPU.synchronize
else
  storage, sync = Array, () -> nothing
end

function best(f, n=5)
  f()
  sync()
  minimum(1:n) do _
    t = time_ns()
    f()
    sync()
    (time_ns() - t) / 1e6
  end
end

image = ImageGeom(; dims=(128, 128, 128), deltas=(1.0f0, 1.0f0, 1.0f0))
g = circular_cone_beam(; angles=range(0, 2pi, 361)[1:360], dso=500, dsd=1000, nu=256, nv=128, du=2.0)
x = storage(rand(Float32, 128, 128, 128))
y = storage(rand(Float32, 256, 128, 360))
xb, yf = similar(x), similar(y)

println("VoxelDrivenProjector, $DEVICE, $(Threads.nthreads()) threads")
for launch in (nothing, LaunchConfig(; layout=(1,)), LaunchConfig(; layout=(2,)), LaunchConfig(; layout=(4,)),
               LaunchConfig(; layout=(8,)), LaunchConfig(; layout=(16,)))
  op = VoxelDrivenProjector(image, g; storage, launch)
  back = best(() -> mul!(xb, adjoint(op), y))
  forward = best(() -> mul!(yf, op, x))
  name = isnothing(launch) ? "default" : "layout $(launch.layout)"
  println("  ", rpad(name, 16), "back ", round(back; digits=1), " ms, forward ", round(forward; digits=1), " ms")
end
