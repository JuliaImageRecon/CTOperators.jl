# RadonKA.jl: parallel beam as a stack of 2D slices (KernelAbstractions, CPU), Float32. Its geometry
# (pixel units, rotation about pixel N ÷ 2 + 1, circular field of view, first dimensions (y, x)) is
# not expressed in our rays, so only the time is compared, at the same image size, number of
# angles and detector cells per slice (as close as RadonKA allows: N - 1 cells).

using RadonKA

library_version() = string(pkgversion(RadonKA))
method_name() = "RadonKA (parallel, slice stack)"
supports(p) = p.geometry isa ParallelBeam
measures_error(p) = false

struct RKState
  angles::Vector{Float32}
  geometry::RadonParallelCircle
end

function setup(p)
  n = p.image.dims[1]
  angles = Float32.(range(0, pi, nviews(p.geometry) + 1)[1:nviews(p.geometry)])
  RKState(angles, RadonParallelCircle(n, -(n - 1) ÷ 2:(n - 1) ÷ 2))
end

datashape(s, p) = (length(s.geometry.in_height), length(s.angles), p.image.dims[3])

# RadonKA allocates its results; the copy into the preallocated output is part of its cost here.
# Qualified names: `radon` of ImagePhantoms (problems.jl) is in scope too.
forward!(s, y, x) = copyto!(y, RadonKA.radon(x, s.angles; geometry=s.geometry))
backward!(s, x, y) = copyto!(x, RadonKA.backproject(y, s.angles; geometry=s.geometry))
