# One library on a list of problems, in that library's environment (benchmark/envs/<lib>):
#   julia --project=benchmark/envs/<lib> --threads=N worker.jl <lib> <out.csv> <budget> <problem>...
# Per problem: setup (plan) time, forward and back projection (minimum and median over repeated
# calls within the time budget), relative error of the forward projection of the phantom against
# its analytic line integrals (where the library takes our rays). Appends one CSV row per problem.

using BenchmarkTools, Printf

const LIB, OUT, BUDGET = ARGS[1], ARGS[2], parse(Float64, ARGS[3])
const PROBLEM_NAMES = ARGS[4:end]

# Defaults for CPU libraries; adapters for devices override them: the device in the results, moving
# arrays to the device, and waiting for the device to finish (timings include it).
device_name() = "CPU"
device_array(state, a) = a
synchronize(state) = nothing

include(joinpath(@__DIR__, "..", "src", "problems.jl"))
include(joinpath(@__DIR__, "adapters", LIB * ".jl"))

const HEADER = "lib,version,method,device,problem,nvoxels,nrays,threads,setup_s,forward_min_s,forward_median_s," *
               "backward_min_s,backward_median_s,rel_error"

function measure(p::Problem)
  t_setup = @elapsed state = setup(p)
  t_setup = min(t_setup, @elapsed setup(p))          # second call: without compilation
  x = device_array(state, phantom_image(p))
  y = device_array(state, zeros(Float32, datashape(state, p)))
  xb = similar(x)
  forward!(state, y, x); backward!(state, xb, y); synchronize(state)    # compile
  bf = @benchmark (forward!($state, $y, $x); synchronize($state)) seconds = BUDGET evals = 1
  # back projection of random data: the phantom's projection is zero for the rays that miss it,
  # which some libraries skip
  yr = device_array(state, rand(Float32, size(y)))
  bb = @benchmark (backward!($state, $xb, $yr); synchronize($state)) seconds = BUDGET evals = 1
  err = NaN
  if measures_error(p)
    forward!(state, y, x)
    idx, exact = error_rays(p)
    err = norm(Float64.(vec(Array(y))[idx]) - exact) / norm(exact)
  end
  (minimum(bf).time / 1e9, median(bf).time / 1e9, minimum(bb).time / 1e9, median(bb).time / 1e9, t_setup, err)
end

isfile(OUT) || write(OUT, HEADER * "\n")
for name in PROBLEM_NAMES
  p = PROBLEMS[name]()
  supports(p) || continue
  fmin, fmed, bmin, bmed, tsetup, err = measure(p)
  row = join([LIB, library_version(), "\"$(method_name())\"", "\"$(device_name())\"", name, prod(p.image.dims), nrays(p.geometry),
              Threads.nthreads(), @sprintf("%.4g", tsetup), @sprintf("%.4g", fmin), @sprintf("%.4g", fmed),
              @sprintf("%.4g", bmin), @sprintf("%.4g", bmed), @sprintf("%.4g", err)], ",")
  open(io -> println(io, row), OUT, "a")
  @printf("%-24s %-12s forward %8.2f ms  back %8.2f ms  setup %7.2f s  error %.4f\n", LIB, name,
          1e3fmin, 1e3bmin, tsetup, err)
end
