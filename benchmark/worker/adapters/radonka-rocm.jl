# RadonKA.jl on an AMD GPU (AMDGPU.jl), Float32: as the CPU adapter radonka.jl (same geometry, only
# the time is compared), with image, data and angles as ROCArrays. RadonKA requires
# KernelAbstractions 0.9, so its environment resolves an older AMDGPU (2.8) than ours.

using RadonKA, AMDGPU

library_version() = string(pkgversion(RadonKA))
method_name() = "RadonKA (parallel, slice stack)"
device_name() = AMDGPU.HIP.name(AMDGPU.device())
supports(p) = p.geometry isa ParallelBeam
measures_error(p) = false

struct RKState
  angles::ROCVector{Float32}
  geometry::RadonParallelCircle
end

function setup(p)
  n = p.image.dims[1]
  angles = Float32.(range(0, pi, nviews(p.geometry) + 1)[1:nviews(p.geometry)])
  RKState(ROCArray(angles), RadonParallelCircle(n, -(n - 1) ÷ 2:(n - 1) ÷ 2))
end

device_array(::RKState, a) = ROCArray(a)
synchronize(::RKState) = AMDGPU.synchronize()
datashape(s, p) = (length(s.geometry.in_height), length(s.angles), p.image.dims[3])

# RadonKA allocates its results; the copy into the preallocated output is part of its cost here.
forward!(s, y, x) = copyto!(y, RadonKA.radon(x, s.angles; geometry=s.geometry))
backward!(s, x, y) = copyto!(x, RadonKA.backproject(y, s.angles; geometry=s.geometry))
