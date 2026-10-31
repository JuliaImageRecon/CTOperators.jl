# parallelproj (Python, conda-forge): Joseph's method in libparallelproj, CPU build (OpenMP), Float32.
# Its arrays are C-ordered: a Julia array of size (nx, ny, nz) is, without a copy, a C-ordered
# (nz, ny, nx) array, so axes, origin, voxel size and ray coordinates are passed in reversed order.
# The rays are those of our geometry, as start and end points (RayList); lines become segments
# through the whole volume.

using CondaPkg, Libdl
# Inside a Julia process the dynamic loader does not find libparallelproj next to the Python
# extension; load it from the conda environment first.
Libdl.dlopen(joinpath(CondaPkg.envdir(), "lib", "libparallelproj.so.2"), Libdl.RTLD_GLOBAL)
using PythonCall

const np = pyimport("numpy")
const ppc = pyimport("parallelproj_core")

library_version() = pyconvert(String, ppc.__version__)
method_name() = "Joseph"
supports(p) = true
measures_error(p) = true

struct PPState
  starts::Py      # (nrays, 3), coordinates (z, y, x)
  ends::Py
  origin::Py      # world coordinate of voxel [0,0,0] in (z, y, x)
  voxsize::Py
end

# the rays as segments: lines (parallel beam) extended by the volume's diagonal on both sides of
# their point on the detector plane, which lies in the volume's centre plane
segments(g, image) = segments(raytype(g), g, image)
segments(::RaySegment, g, image) = RayList(g)
function segments(::RayLine, g, image)
  reach = sqrt(sum(abs2, image.dims .* image.deltas))
  starts, ends = zeros(3, nrays(g)), zeros(3, nrays(g))
  for i in 1:nrays(g)
    a, b = ray(g, i)
    d = normalize(b - a)
    starts[:, i] = a - reach * d
    ends[:, i] = a + reach * d
  end
  RayList(starts, ends; datasize=datasize(g))
end

function setup(p)
  r = segments(p.geometry, p.image)
  zyx = [3, 2, 1]
  starts = Matrix{Float32}(r.starts[zyx, :])   # 3 × nrays column-major = C-ordered (nrays, 3)
  ends = Matrix{Float32}(r.ends[zyx, :])
  first = [axis(p.image, d)[1] for d in 1:3]
  PPState(np.asarray(starts).T, np.asarray(ends).T,
          np.asarray(Float32.(first[zyx])), np.asarray(Float32.(collect(p.image.deltas)[zyx])))
end

# the C-ordered (nz, ny, nx) view of a Julia (nx, ny, nz) array, and the flat projection data
image_view(x) = np.asarray(x).T
data_view(y) = np.asarray(vec(y))

function forward!(s::PPState, y, x)
  ppc.joseph3d_fwd(s.starts, s.ends, image_view(x), s.origin, s.voxsize, data_view(y))
  y
end

function backward!(s::PPState, x, y)
  fill!(x, 0)   # joseph3d_back adds into the image
  ppc.joseph3d_back(s.starts, s.ends, image_view(x), s.origin, s.voxsize, data_view(y))
  x
end

datashape(s, p) = datasize(p.geometry)
