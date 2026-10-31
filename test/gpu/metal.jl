using Metal

# Metal has no Float64
arrayTypes = [MtlArray]
floatTypes = [Float32]
include("gpu.jl")
