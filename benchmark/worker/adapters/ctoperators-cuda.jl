# CTOperators.jl on an NVIDIA GPU (CUDA.jl), Float32: the projector of PROJECTOR with its geometry on
# the device, image and data as CuArrays. Included by the adapters ctoperators-<method>-cuda.

using CUDA
using LinearAlgebra: mul!

library_version() = string(pkgversion(CTOperators))
method_name() = string(nameof(PROJECTOR))[1:end-length("Projector")]
device_name() = CUDA.name(CUDA.device())
supports(p) = true
measures_error(p) = true

setup(p) = PROJECTOR(p.image, p.geometry; storage=CuArray)
device_array(::AbstractProjector, a) = CuArray(a)
synchronize(::AbstractProjector) = CUDA.synchronize()
forward!(op, y, x) = mul!(y, op, x)
backward!(op, x, y) = mul!(x, adjoint(op), y)
datashape(op, p) = datasize(p.geometry)
