# CTOperators.jl on an AMD GPU (AMDGPU.jl), Float32: the projector of PROJECTOR with its geometry
# on the device, image and data as ROCArrays. Included by the adapters ctoperators-<method>-rocm.

using AMDGPU
using LinearAlgebra: mul!

# the GPU to run on, before any array is created on it
CTOperators.KA.device!(ROCBackend(), DEVICE)

library_version() = string(pkgversion(CTOperators))
method_name() = string(nameof(PROJECTOR))[1:end-length("Projector")]
device_name() = AMDGPU.HIP.name(AMDGPU.device())
# distance driven needs detector cells; the ray-driven projectors take every geometry
supports(p) = PROJECTOR !== DistanceDrivenProjector || p.geometry isa AbstractDetectorGeometry
measures_error(p) = true

setup(p) = PROJECTOR(p.image, p.geometry; storage=ROCArray)
device_array(::AbstractProjector, a) = ROCArray(a)
synchronize(::AbstractProjector) = AMDGPU.synchronize()
forward!(op, y, x) = mul!(y, op, x)
backward!(op, x, y) = mul!(x, adjoint(op), y)
datashape(op, p) = datasize(p.geometry)
