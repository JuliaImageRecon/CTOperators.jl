# CTOperators.jl: FDK with the plain ramp filter, Float32, the default launch, on the device of
# STORAGE. Included by the adapters ctoperators-fdk(-<device>).

using LinearAlgebra: mul!

library_version() = string(pkgversion(CTOperators))
method_name() = "FDK"
# circular scans only
supports(p) = p.geometry isa Union{ConeBeam,ArcConeBeam,ParallelBeam} &&
              try CTOperators.circular_scan(p.geometry); true catch e; e isa ArgumentError || rethrow(); false end

setup(p) = FDK(p.image, p.geometry; window=:ramp, storage=STORAGE)
reconstruct!(plan, x, y) = mul!(x, plan, y)
