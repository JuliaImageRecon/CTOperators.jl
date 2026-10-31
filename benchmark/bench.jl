# Benchmarks of CTOperators.jl against other projector libraries. Run from the repository root with
# any Julia; the orchestrator uses the standard library only.
#
#   julia benchmark/bench.jl list                      suites, problems and libraries
#   julia benchmark/bench.jl run quick                 all libraries on the quick suite
#   julia benchmark/bench.jl run standard --libs ctoperators,parallelproj --threads 1,24
#
# Options: --libs (comma separated), --threads (comma separated, default: all cores), --budget
# (seconds of repeated calls per measurement, default 2). Each library runs in a worker process with
# its own environment (benchmark/envs/<lib>, instantiated on first use), so competitors never become
# dependencies of CTOperators.jl. Results: benchmark/out/<time>-<suite>.csv.

using Dates, Printf

const ROOT = @__DIR__
const LIBS = ["ctoperators", "parallelproj", "radonka"]

include(joinpath(ROOT, "src", "suites.jl"))

function option(args, name, default)
  i = findfirst(==("--" * name), args)
  i === nothing ? default : args[i+1]
end

function instantiate(lib)
  env = joinpath(ROOT, "envs", lib)
  run(`$(Base.julia_cmd()) --project=$env -e "using Pkg; Pkg.instantiate()"`)
end

function run_suite(suite, libs, threads, budget)
  haskey(SUITES, suite) || error("unknown suite $suite; see `julia benchmark/bench.jl list`")
  mkpath(joinpath(ROOT, "out"))
  out = joinpath(ROOT, "out", Dates.format(now(), "yyyymmdd-HHMMSS") * "-$suite.csv")
  worker = joinpath(ROOT, "worker", "worker.jl")
  for lib in libs
    instantiate(lib)
    for nt in threads
      env = joinpath(ROOT, "envs", lib)
      cmd = `$(Base.julia_cmd()) --project=$env --threads=$nt $worker $lib $out $budget $(SUITES[suite])`
      run(addenv(cmd, "OMP_NUM_THREADS" => string(nt)))   # OpenMP libraries (parallelproj)
    end
  end
  println("results: ", out)
end

function main(args)
  isempty(args) && return println("usage: julia benchmark/bench.jl list | run <suite> [options]")
  if args[1] == "list"
    for (s, ps) in sort(collect(SUITES)); println("suite $s: ", join(ps, ", ")); end
    println("libraries: ", join(LIBS, ", "))
  elseif args[1] == "run"
    libs = split(option(args, "libs", join(LIBS, ",")), ",")
    threads = parse.(Int, split(option(args, "threads", string(Sys.CPU_THREADS)), ","))
    run_suite(args[2], libs, threads, option(args, "budget", "2"))
  else
    error("unknown command $(args[1])")
  end
end

main(ARGS)
