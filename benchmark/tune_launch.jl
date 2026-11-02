# Times the launch configurations of a projector on one GPU, to choose `LaunchConfig` for a device:
#   julia --project=benchmark/envs/ctoperators benchmark/tune_launch.jl <cuda|rocm> [device] [problem] [projector]
# device: the GPU, counted from 1 (default 1); problem: one of benchmark/src/problems.jl (default
# cone256); projector: Siddon or Joseph (default Joseph). Prints forward and back projection times
# (minimum over repeated calls) for workgroup sizes and tiles of detector cells, the device's
# defaults first, and the fastest configurations last.

using CTOperators, BenchmarkTools, Printf
using LinearAlgebra: mul!
include(joinpath(@__DIR__, "src", "problems.jl"))

const BACKEND_NAME = get(ARGS, 1, "cuda")
const DEVICE = parse(Int, get(ARGS, 2, "1"))
const PROBLEM = get(ARGS, 3, "cone256")
const PROJECTOR = Dict("Siddon" => SiddonProjector, "Joseph" => JosephProjector)[get(ARGS, 4, "Joseph")]

if BACKEND_NAME == "cuda"
  using CUDA
  const ArrayType, backend, sync = CuArray, CUDABackend(), CUDA.synchronize
  const devicename() = CUDA.name(CUDA.device())
elseif BACKEND_NAME == "rocm"
  using AMDGPU
  const ArrayType, backend, sync = ROCArray, ROCBackend(), AMDGPU.synchronize
  const devicename() = AMDGPU.HIP.name(AMDGPU.device())
else
  error("backend must be cuda or rocm, got $BACKEND_NAME")
end
CTOperators.KA.device!(backend, DEVICE)

p = PROBLEMS[PROBLEM]()
x = ArrayType(Float32.(phantom_image(p)))
y = ArrayType(rand(Float32, datasize(p.geometry)...))
xb, yf = similar(x), similar(y)

# minimum time in ms of f, which launches kernels, including the wait for the device
time_ms(f) = (f(); sync(); minimum(@benchmark ($f(); $sync()) seconds = 2 evals = 1).time / 1e6)

function measure(launch)
  op = PROJECTOR(p.image, p.geometry; storage=ArrayType, launch)
  time_ms(() -> mul!(yf, op, x)), time_ms(() -> mul!(xb, adjoint(op), y))
end

configs = Any[nothing]   # the defaults of the device
for workgroupsize in (nothing, 32, 64, 128, 256), tile in (nothing, (8, 4), (8, 8), (16, 4), (16, 8), (16, 16))
  push!(configs, LaunchConfig(; workgroupsize, tile))
end

label(::Nothing) = "defaults of the device"
label(c::LaunchConfig) = "workgroupsize=$(something(c.workgroupsize, "backend")), tile=$(something(c.tile, "none"))"

@printf("%s on %s (device %d), %s\n\n", nameof(PROJECTOR), devicename(), DEVICE, PROBLEM)
results = []
for c in configs
  tf, tb = measure(c)
  push!(results, (c, tf, tb))
  @printf("%-44s forward %8.2f ms   back %8.2f ms\n", label(c), tf, tb)
end

println("\nfastest forward:")
for (c, tf, tb) in sort(results; by=r -> r[2])[1:3]
  @printf("  %-44s forward %8.2f ms   back %8.2f ms\n", label(c), tf, tb)
end
println("fastest back:")
for (c, tf, tb) in sort(results; by=r -> r[3])[1:3]
  @printf("  %-44s forward %8.2f ms   back %8.2f ms\n", label(c), tf, tb)
end
