# FDK of CTOperators.jl against the FDK of Sinograms.jl (MIT; CPU only, threaded over voxels) on one
# circular cone-beam problem: the same Sinograms.jl geometry (flat detector), image and data, both
# with the plain ramp filter. Times the setup of the plan and the reconstruction (minimum of
# repeated calls after a first one), and the error of both against the phantom.
#
#   julia --project=benchmark/envs/fdk -t 24 benchmark/fdk.jl          # CPU, 24 threads
#   julia --project=benchmark/envs/fdk -t 1 benchmark/fdk.jl           # CPU, 1 thread
#   julia --project=benchmark/envs/fdk benchmark/fdk.jl rocm           # CTOperators on an AMD GPU
#
# Measure on a quiet machine. Results: benchmark/README.md.

using CTOperators, Sinograms, ImagePhantoms, LinearAlgebra
using ImageGeoms: ImageGeom, axis

const DEVICE = get(ARGS, 1, "cpu")
const REPEATS = Threads.nthreads() == 1 ? 3 : 5

if DEVICE == "rocm"
  using AMDGPU
  storage, sync = ROCArray, AMDGPU.synchronize
else
  storage, sync = Array, () -> nothing
end

function best(f, n)
  f()
  sync()
  minimum(1:n) do _
    t = time_ns()
    f()
    sync()
    (time_ns() - t) / 1e9
  end
end

# 128^3 voxels of 1 mm; 360 views over one turn, 256 x 128 cells of 2 mm (1 mm at the axis)
ig = ImageGeom(; dims=(128, 128, 128), deltas=(1.0f0, 1.0f0, 1.0f0))
rg = CtFanFlat(; ns=256, nt=128, ds=2.0f0, dt=2.0f0, na=360, dsd=1000.0f0, dod=500.0f0)
g = projection_geometry(rg)
ob = ellipsoid(ImagePhantoms.ellipsoid_parameters(; fovs=(128, 128, 128)))
truth = Float32.(phantom(axis(ig, 1), axis(ig, 2), axis(ig, 3), ob, 2))
y = JosephProjector(ig, g) * truth
relative_error(x) = norm(x - truth) / norm(truth)

println("CTOperators FDK, $DEVICE, $(Threads.nthreads()) threads")
setup = best(() -> FDK(ig, g; window=:ramp, storage), REPEATS)
plan = FDK(ig, g; window=:ramp, storage)
yd = storage(y)
recon = best(() -> plan * yd, REPEATS)
x = Array(plan * yd)
println("  setup $(round(setup; sigdigits=3)) s, reconstruction $(round(recon; sigdigits=3)) s, error $(round(relative_error(x); sigdigits=3))")

if DEVICE == "cpu"
  println("Sinograms fdk, $(Threads.nthreads()) threads")
  setup = best(() -> plan_fbp(rg, ig), REPEATS)
  splan = plan_fbp(rg, ig)
  recon = best(() -> Sinograms.fdk(splan, y), REPEATS)
  xs = Sinograms.fdk(splan, y)
  println("  setup $(round(setup; sigdigits=3)) s, reconstruction $(round(recon; sigdigits=3)) s, error $(round(relative_error(xs); sigdigits=3))")
  println("  difference of the reconstructions $(round(norm(x - xs) / norm(xs); sigdigits=3))")
end
