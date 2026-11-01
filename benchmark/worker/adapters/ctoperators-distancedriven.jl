# CTOperators.jl: DistanceDrivenProjector on the CPU (KernelAbstractions' CPU backend), Float32.

using LinearAlgebra: mul!

library_version() = string(pkgversion(CTOperators))
method_name() = "DistanceDriven"
supports(p) = p.geometry isa AbstractDetectorGeometry
measures_error(p) = true

setup(p) = DistanceDrivenProjector(p.image, p.geometry)
forward!(op, y, x) = mul!(y, op, x)
backward!(op, x, y) = mul!(x, adjoint(op), y)
datashape(op, p) = datasize(p.geometry)
