# Times the launch configurations of a projector (or of FDK) on one device, to choose a
# `LaunchConfig` for it:
#   julia --project=benchmark/envs/ctoperators [-t N] benchmark/tune_launch.jl <cpu|cuda|rocm> [device] [problem] [projector]
# device: the GPU, counted from 1 (default 1; ignored on the CPU); problem: one of
# benchmark/src/problems.jl (default cone256, for FDK fdk128); projector: Siddon, Joseph,
# DistanceDriven, VoxelDriven or FDK (default Joseph). Prints the times (minimum over repeated
# calls) of the forward and the back projection (FDK: the reconstruction of the problem's data) for
# workgroup sizes and layouts (tiles of detector cells for the ray-driven projectors, runs of voxels
# for the voxel-driven projector and FDK), the device's defaults first, and the fastest
# configurations last.

using CTOperators, BenchmarkTools, Printf
using LinearAlgebra: mul!
include(joinpath(@__DIR__, "src", "problems.jl"))

const BACKEND_NAME = get(ARGS, 1, "cuda")
const DEVICE = parse(Int, get(ARGS, 2, "1"))
const PROJECTOR_NAME = get(ARGS, 4, "Joseph")
const PROBLEM = get(ARGS, 3, PROJECTOR_NAME == "FDK" ? "fdk128" : "cone256")
const PROJECTOR = Dict("Siddon" => SiddonProjector, "Joseph" => JosephProjector, "DistanceDriven" => DistanceDrivenProjector,
                       "VoxelDriven" => VoxelDrivenProjector, "FDK" => FDK)[PROJECTOR_NAME]

if BACKEND_NAME == "cuda"
  using CUDA
  const ArrayType, sync = CuArray, CUDA.synchronize
  CTOperators.KA.device!(CUDABackend(), DEVICE)
  devicename() = CUDA.name(CUDA.device())
elseif BACKEND_NAME == "rocm"
  using AMDGPU
  const ArrayType, sync = ROCArray, AMDGPU.synchronize
  CTOperators.KA.device!(ROCBackend(), DEVICE)
  devicename() = AMDGPU.HIP.name(AMDGPU.device())
elseif BACKEND_NAME == "cpu"
  const ArrayType, sync = Array, () -> nothing
  devicename() = "CPU, $(Threads.nthreads()) threads"
else
  error("backend must be cpu, cuda or rocm, got $BACKEND_NAME")
end

p = PROBLEMS[PROBLEM]()
x = ArrayType(Float32.(phantom_image(p)))
y = ArrayType(PROJECTOR === FDK ? projection_data(p) : rand(Float32, datasize(p.geometry)...))
xb, yf = similar(x), similar(y)

# minimum time in ms of f, which launches kernels, including the wait for the device
time_ms(f) = (f(); sync(); minimum(@benchmark ($f(); $sync()) seconds = 2 evals = 1).time / 1e6)

# forward and back projection; for FDK the reconstruction (and no second time)
function measure(launch)
  if PROJECTOR === FDK
    plan = FDK(p.image, p.geometry; window=:ramp, storage=ArrayType, launch)
    return time_ms(() -> mul!(xb, plan, y)), NaN
  end
  op = PROJECTOR(p.image, p.geometry; storage=ArrayType, launch)
  time_ms(() -> mul!(yf, op, x)), time_ms(() -> mul!(xb, adjoint(op), y))
end

# the layouts of the projector: tiles of detector cells, or runs of voxels along x
voxels = PROJECTOR in (VoxelDrivenProjector, FDK)
layouts = voxels ? ((), (2,), (4,), (8,), (16,)) : ((), (8, 4), (8, 8), (16, 4), (16, 8), (16, 16))
workgroupsizes = voxels ? (nothing, 64, 256) : (nothing, 32, 64, 128, 256)
configs = Any[nothing]   # the defaults of the device
for workgroupsize in workgroupsizes, layout in layouts
  push!(configs, LaunchConfig(; workgroupsize, layout))
end

label(::Nothing) = "defaults of the device"
function label(c::LaunchConfig)
  layout = isempty(c.layout) ? (voxels ? "one voxel per thread" : "data order") : string(c.layout)
  "workgroupsize=$(something(c.workgroupsize, "derived")), layout=$layout"
end
times(tf, tb) = PROJECTOR === FDK ? @sprintf("reconstruction %8.2f ms", tf) :
                @sprintf("forward %8.2f ms   back %8.2f ms", tf, tb)

@printf("%s on %s (device %d), %s\n\n", nameof(PROJECTOR), devicename(), DEVICE, PROBLEM)
results = []
for c in configs
  tf, tb = measure(c)
  push!(results, (c, tf, tb))
  @printf("%-52s %s\n", label(c), times(tf, tb))
end

for (name, column) in (PROJECTOR === FDK ? (("reconstruction", 2),) : (("forward", 2), ("back", 3)))
  println("\nfastest $name:")
  for (c, tf, tb) in sort(results; by=r -> r[column])[1:3]
    @printf("  %-50s %s\n", label(c), times(tf, tb))
  end
end
