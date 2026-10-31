# CTOperators.jl: JosephProjector on the CPU (KernelAbstractions' CPU backend), Float32.

using LinearAlgebra: mul!

library_version() = string(pkgversion(CTOperators))
method_name() = "Joseph"
supports(p) = true
measures_error(p) = true

setup(p) = JosephProjector(p.image, p.geometry)
forward!(op, y, x) = mul!(y, op, x)
backward!(op, x, y) = mul!(x, adjoint(op), y)
datashape(op, p) = datasize(p.geometry)
