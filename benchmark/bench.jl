# Benchmarks of CTOperators.jl against other projector libraries. Run from the repository root with
# any Julia; the orchestrator uses the standard library only.
#
#   julia benchmark/bench.jl list                      suites, problems and libraries
#   julia benchmark/bench.jl run quick                 all libraries on the quick suite
#   julia benchmark/bench.jl run standard --libs ctoperators-joseph,parallelproj --threads 1,24
#   julia benchmark/bench.jl run standard --libs ctoperators-siddon-rocm,ctoperators-joseph-rocm --threads 1
#   julia benchmark/bench.jl run fdk --threads 1,24      reconstructions (FDK) of CTOperators.jl and Sinograms.jl
#
# Options: --libs (comma separated), --threads (comma separated, default: all cores), --budget
# (seconds of repeated calls per measurement, default 2). Each library runs in a worker process with
# its own environment (benchmark/envs/<lib>, instantiated on first use), so competitors never become
# dependencies of CTOperators.jl. Results: benchmark/out/<time>-<suite>.csv.

using Dates, Printf

const ROOT = @__DIR__
const LIBS = ["ctoperators-siddon", "ctoperators-joseph", "ctoperators-distancedriven", "ctoperators-voxeldriven",
              "parallelproj", "radonka"]
# GPU libraries: only run when named with --libs (they need the device)
const GPU_LIBS = ["ctoperators-siddon-rocm", "ctoperators-joseph-rocm", "ctoperators-distancedriven-rocm",
                  "ctoperators-voxeldriven-rocm", "radonka-rocm", "ctoperators-siddon-cuda", "ctoperators-joseph-cuda",
                  "ctoperators-distancedriven-cuda", "ctoperators-voxeldriven-cuda", "parallelproj-cuda"]
# reconstructions, for the suites in RECONSTRUCTION_SUITES; the GPU ones only when named with --libs
const RECONSTRUCTION_LIBS = ["ctoperators-fdk", "sinograms-fdk", "sinograms-fdk-nodecon1"]
const GPU_RECONSTRUCTION_LIBS = ["ctoperators-fdk-rocm", "ctoperators-fdk-cuda"]

# the environment of a library: envs/<lib> if it exists (parallelproj-cuda: the CUDA build), else the
# part of its name before "-" (one environment for the methods and devices of CTOperators.jl)
function environment(lib)
  own = joinpath(ROOT, "envs", lib)
  isdir(own) ? own : joinpath(ROOT, "envs", first(split(lib, "-")))
end

include(joinpath(ROOT, "src", "suites.jl"))

function option(args, name, default)
  i = findfirst(==("--" * name), args)
  i === nothing ? default : args[i+1]
end

function instantiate(lib)
  env = environment(lib)
  run(`$(Base.julia_cmd()) --project=$env -e "using Pkg; Pkg.instantiate()"`)
end

function run_suite(suite, libs, threads, budget, device)
  haskey(SUITES, suite) || error("unknown suite $suite; see `julia benchmark/bench.jl list`")
  # a reconstruction suite takes reconstructions, a projector suite projectors
  recon = suite in RECONSTRUCTION_SUITES
  for lib in libs
    lib in (recon ? [RECONSTRUCTION_LIBS; GPU_RECONSTRUCTION_LIBS] : [LIBS; GPU_LIBS]) ||
      error("$lib is not a $(recon ? "reconstruction" : "projector") library; see `julia benchmark/bench.jl list`")
  end
  mkpath(joinpath(ROOT, "out"))
  out = joinpath(ROOT, "out", Dates.format(now(), "yyyymmdd-HHMMSS") * "-$suite.csv")
  worker = joinpath(ROOT, "worker", "worker.jl")
  for lib in libs
    instantiate(lib)
    for nt in threads
      env = environment(lib)
      cmd = `$(Base.julia_cmd()) --project=$env --threads=$nt $worker $lib $out $budget $(SUITES[suite])`
      # OpenMP libraries (parallelproj); the GPU of the GPU libraries
      run(addenv(cmd, "OMP_NUM_THREADS" => string(nt), "BENCH_DEVICE" => device))
    end
  end
  println("results: ", out)
end

function main(args)
  isempty(args) && return println("usage: julia benchmark/bench.jl list | run <suite> [options]")
  if args[1] == "list"
    for (s, ps) in sort(collect(SUITES)); println("suite $s: ", join(ps, ", ")); end
    println("libraries: ", join(LIBS, ", "))
    println("GPU libraries (with --libs): ", join(GPU_LIBS, ", "))
    println("reconstructions (suites ", join(RECONSTRUCTION_SUITES, ", "), "): ", join(RECONSTRUCTION_LIBS, ", "))
    println("GPU reconstructions (with --libs): ", join(GPU_RECONSTRUCTION_LIBS, ", "))
  elseif args[1] == "run"
    default = args[2] in RECONSTRUCTION_SUITES ? RECONSTRUCTION_LIBS : LIBS
    libs = split(option(args, "libs", join(default, ",")), ",")
    threads = parse.(Int, split(option(args, "threads", string(Sys.CPU_THREADS)), ","))
    run_suite(args[2], libs, threads, option(args, "budget", "2"), option(args, "device", "1"))
  else
    error("unknown command $(args[1])")
  end
end

main(ARGS)
