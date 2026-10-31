"""
    AbstractProjectionGeometry{T}

Supertype of acquisition geometries: where the rays run and the shape of the projection data.
`T` is the element type of the stored coordinates.

Interface:

* `datasize(g)`: size of the projection data, e.g. `(nu, nv, nviews)`
* `ray(g, i) -> (a, b)`: ray `i` (linear index into the projection data) as two points
  `SVector{3,T}`; callable inside kernels
* `raytype(g)`: [`RaySegment()`](@ref RaySegment) if the ray is the segment from `a` to `b`,
  [`RayLine()`](@ref RayLine) if it is the whole line through `a` and `b`

Geometries are adapted (moved and converted) with `Adapt.adapt`, e.g.
`adapt(CuArray{Float32}, g)`.
"""
abstract type AbstractProjectionGeometry{T} end

"""
    AbstractDetectorGeometry{T} <: AbstractProjectionGeometry{T}

Geometries of a source (or ray direction) and a flat detector per view; the projection data has
size `(nu, nv, nviews)`. Besides rays they describe the detector cells, which footprint methods
need.

Interface (in addition to that of `AbstractProjectionGeometry`):

* `nviews(g)`, `detectorsize(g) = (nu, nv)`
* `subset(g, views)`: the geometry of a subset of the views
"""
abstract type AbstractDetectorGeometry{T} <: AbstractProjectionGeometry{T} end

"""
    RaySegment()

[`raytype`](@ref) of geometries whose ray `i` is the segment between the two points of
`ray(g, i)`, e.g. from the source to a detector pixel.
"""
struct RaySegment end

"""
    RayLine()

[`raytype`](@ref) of geometries whose ray `i` is the whole line through the two points of
`ray(g, i)`, e.g. parallel beams. Only the part inside the volume contributes.
"""
struct RayLine end

"""
    raytype(g::AbstractProjectionGeometry)

[`RaySegment()`](@ref RaySegment) or [`RayLine()`](@ref RayLine).
"""
function raytype end

"""
    datasize(g::AbstractProjectionGeometry)

Size of the projection data.
"""
function datasize end

"""
    ray(g::AbstractProjectionGeometry, i) -> (a, b)

Ray `i` (linear index into the projection data) as two points `SVector{3,T}`, see
[`raytype`](@ref). Callable inside kernels.
"""
function ray end

"""
    nviews(g::AbstractDetectorGeometry)
"""
function nviews end

"""
    detectorsize(g::AbstractDetectorGeometry) -> (nu, nv)
"""
function detectorsize end

"""
    subset(g::AbstractDetectorGeometry, views)

The geometry of the views `views` (indices or a range) of `g`, for ordered-subset methods.
"""
function subset end

"""
    nrays(g::AbstractProjectionGeometry)

Number of rays, `prod(datasize(g))`.
"""
nrays(g::AbstractProjectionGeometry) = prod(datasize(g))

Base.eltype(::AbstractProjectionGeometry{T}) where {T} = T

datasize(g::AbstractDetectorGeometry) = (detectorsize(g)..., nviews(g))

# column k of a 3 × n matrix as an SVector
@inline column(m, k) = @inbounds SVector(m[1, k], m[2, k], m[3, k])

# Detector pixel (iu, iv) and view k of ray i: data order (nu, nv, nviews), u fastest.
@inline function pixel_index(i, nu, nv)
  j = i - 1
  ju, j = rem(j, nu), div(j, nu)
  jv, k = rem(j, nv), div(j, nv)
  ju + 1, jv + 1, k + 1
end

# the position of pixel (iu, iv) on a detector with centre c and cell vectors u, v
@inline function pixel_position(c, u, v, iu, iv, nu, nv)
  T = eltype(c)
  c + (T(iu) - T(nu + 1) / 2) * u + (T(iv) - T(nv + 1) / 2) * v
end

function check_views(name, ms...)
  for m in ms
    size(m, 1) == 3 || throw(DimensionMismatch("$name: per-view vectors must be 3 × nviews, got $(size(m))"))
    size(m, 2) == size(ms[1], 2) || throw(DimensionMismatch("$name: all per-view arrays need the same number of views"))
  end
end

check_detector(nu, nv) = nu > 0 && nv > 0 || throw(ArgumentError("detector size must be positive, got ($nu, $nv)"))

# A new array on the device of the geometry's arrays (storage_array(g): one of them).
geometry_array(g, dims...) = similar(storage_array(g), dims...)

include("raylist.jl")
include("conebeam.jl")
include("parallelbeam.jl")
include("trajectories.jl")
