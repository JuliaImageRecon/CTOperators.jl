# Suites: lists of problems (defined in problems.jl). Standard library only: the orchestrator
# (bench.jl) includes this file without loading the problem definitions.
const SUITES = Dict(
  "quick" => ["cone128", "parallel128"],
  "standard" => ["cone128", "parallel128", "cone256", "helix256", "parallel256"],
)
