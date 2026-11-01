# parallelproj (Python, conda-forge) with the CUDA build of libparallelproj, Float32, in its own
# environment (envs/parallelproj-cuda). Rays, image and data live on the GPU as CuPy arrays, so the
# timings exclude transfers between host and device ("direct GPU mode" of the parallelproj paper).
# Layout as in parallelproj.jl: C-ordered arrays, image (nz, ny, nx), coordinates (z, y, x).

using CondaPkg, Libdl
Libdl.dlopen(joinpath(CondaPkg.envdir(), "lib", "libparallelproj.so.2"), Libdl.RTLD_GLOBAL)
using PythonCall
using LinearAlgebra: normalize

const np = pyimport("numpy")
const cp = pyimport("cupy")
const ppc = pyimport("parallelproj_core")

library_version() = pyconvert(String, ppc.__version__)
method_name() = "Joseph"
device_name() = pyconvert(String, cp.cuda.runtime.getDeviceProperties(0)["name"].decode())
supports(p) = true
measures_error(p) = true

# A CuPy array with the shape of the Julia array it was made from, for the worker (`similar`,
# `size`, `Array`): the image in parallelproj's C-ordered (nz, ny, nx) layout, data as a flat vector.
struct CuPyArray
  a::Py
  dims::Tuple
end
Base.size(x::CuPyArray) = x.dims
Base.similar(x::CuPyArray) = CuPyArray(cp.empty_like(x.a), x.dims)
Base.Array(x::CuPyArray) = reshape(pyconvert(Vector{Float32}, cp.asnumpy(x.a).ravel()), x.dims)

struct PPState
  starts::Py      # (nrays, 3) on the GPU, coordinates (z, y, x)
  ends::Py
  origin::Py      # world coordinate of voxel [0,0,0] in (z, y, x), on the host
  voxsize::Py
  imagedims::NTuple{3,Int}   # tells images from (also 3D) projection data in device_array
end

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
  PPState(cp.asarray(np.asarray(starts).T), cp.asarray(np.asarray(ends).T),
          np.asarray(Float32.(first[zyx])), np.asarray(Float32.(collect(p.image.deltas)[zyx])), p.image.dims)
end

# images (nx, ny, nz) become C-ordered (nz, ny, nx) CuPy arrays, projection data flat vectors
function device_array(s::PPState, a::AbstractArray)
  b = Float32.(a)
  size(a) == s.imagedims ? CuPyArray(cp.ascontiguousarray(cp.asarray(np.asarray(b).T)), size(a)) :
                           CuPyArray(cp.asarray(np.asarray(vec(b))), size(a))
end
synchronize(::PPState) = cp.cuda.Device().synchronize()

function forward!(s::PPState, y, x)
  ppc.joseph3d_fwd(s.starts, s.ends, x.a, s.origin, s.voxsize, y.a)
  y
end

function backward!(s::PPState, x, y)
  x.a.fill(0)   # joseph3d_back adds into the image
  ppc.joseph3d_back(s.starts, s.ends, x.a, s.origin, s.voxsize, y.a)
  x
end

datashape(s, p) = datasize(p.geometry)
