# Suites: lists of problems (defined in problems.jl). Standard library only: the orchestrator
# (bench.jl) includes this file without loading the problem definitions. The projector suites time
# forward and back projections; the reconstruction suites (RECONSTRUCTION_SUITES) time a
# reconstruction from the data of the problem, with other libraries (bench.jl).
const SUITES = Dict(
  "quick" => ["cone128", "parallel128"],
  "standard" => ["cone128", "parallel128", "cone256", "helix256", "parallel256"],
  "fdk" => ["fdk128"],
)
const RECONSTRUCTION_SUITES = ["fdk"]
