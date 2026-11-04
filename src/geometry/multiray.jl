# Several rays per detector cell: a detector geometry wrapped with a point strategy (where in the
# cell the rays end) and a weighting strategy. Offsets are in cell units: (0, 0) is the cell
# center, +-1/2 the cell edges. The number of rays K is a type parameter of the point strategy,
# so the loop over the rays has a length known to the compiler.

"""
    AbstractPoints{K}

Supertype of the point strategies of [`MultiRay`](@ref): where the `K` rays of a detector cell end.
A subtype implements [`CTOperators.point_offset`](@ref). If it stores floating-point numbers, it
also implements [`CTOperators.points_eltype`](@ref), so that it follows the element type of the
projector. The type can live in any package; see [Write a Sampling Strategy](@ref).
"""
abstract type AbstractPoints{K} end

"""
    AbstractWeights

Supertype of the weighting strategies of [`MultiRay`](@ref). A subtype implements
[`CTOperators.subray_weight`](@ref), and [`CTOperators.weights_eltype`](@ref) if it stores
floating-point numbers. The type can live in any package; see [Write a Sampling Strategy](@ref).
"""
abstract type AbstractWeights end

"""
    RegularPoints(su, sv=su)

The centers of `su × sv` subcells of each detector cell, `su` along `u` and `sv` along `v`:
`su * sv` rays per cell. Point strategy of [`MultiRay`](@ref).
"""
struct RegularPoints{K} <: AbstractPoints{K}
  su::Int
  sv::Int
end

"""
    JitteredPoints(su, sv=su; seed=0)

The points of [`RegularPoints`](@ref), each moved to a random position within its subcell
("jittered sampling", R. L. Cook, "Stochastic sampling in computer graphics", ACM Transactions on
Graphics 5(1), 51-72, 1986). The offsets are fixed by `seed` (default 0), the detector cell, the view
and the ray, so they are the same in the forward and the back projection, on every device and in
`subset`. Point strategy of [`MultiRay`](@ref).
"""
struct JitteredPoints{K} <: AbstractPoints{K}
  su::Int
  sv::Int
  seed::UInt32
end

"""
    RandomPoints(n; seed=0)

`n` random points, uniformly distributed in each detector cell, fixed like those of
[`JitteredPoints`](@ref). Point strategy of [`MultiRay`](@ref).
"""
struct RandomPoints{K} <: AbstractPoints{K}
  seed::UInt32
end

"""
    GaussianPoints(n, sigma_u, sigma_v=sigma_u; seed=0)

`n` random points per detector cell, normally distributed around the cell center with standard
deviations `sigma_u`, `sigma_v` in cell units: a detector response set by where the rays land, with
all rays counting equally. Points can land outside the cell, which models blur across neighboring
cells. The offsets are fixed like those of [`JitteredPoints`](@ref). Point strategy of
[`MultiRay`](@ref).
"""
struct GaussianPoints{K,T} <: AbstractPoints{K}
  sigma_u::T
  sigma_v::T
  seed::UInt32
end

"""
    Points(offsets)

Given points, the same in every detector cell: column `k` of the `2 × K` matrix `offsets` is the
offset of ray `k` in cell units, `(0, 0)` the cell center and `±1/2` the cell edges. Points
outside the cell are allowed. Point strategy of [`MultiRay`](@ref).
"""
struct Points{K,T} <: AbstractPoints{K}
  offsets::SVector{K,SVector{2,T}}
end

function check_subcells(su, sv)
  su > 0 && sv > 0 || throw(ArgumentError("numbers of subcells must be positive, got ($su, $sv)"))
end

function RegularPoints(su::Integer, sv::Integer=su)
  check_subcells(su, sv)
  RegularPoints{Int(su * sv)}(su, sv)
end

function JitteredPoints(su::Integer, sv::Integer=su; seed::Integer=0)
  check_subcells(su, sv)
  JitteredPoints{Int(su * sv)}(su, sv, UInt32(seed))
end

function RandomPoints(n::Integer; seed::Integer=0)
  n > 0 || throw(ArgumentError("number of points must be positive, got $n"))
  RandomPoints{Int(n)}(UInt32(seed))
end

function GaussianPoints(n::Integer, sigma_u::Real, sigma_v::Real=sigma_u; seed::Integer=0)
  n > 0 || throw(ArgumentError("number of points must be positive, got $n"))
  sigma_u > 0 && sigma_v > 0 || throw(ArgumentError("standard deviations must be positive, got ($sigma_u, $sigma_v)"))
  T = float(promote_type(typeof(sigma_u), typeof(sigma_v)))
  GaussianPoints{Int(n),T}(sigma_u, sigma_v, UInt32(seed))
end

function Points(offsets::AbstractMatrix{<:Real})
  size(offsets, 1) == 2 && size(offsets, 2) > 0 ||
    throw(DimensionMismatch("offsets must be 2 × K with K > 0, got $(size(offsets))"))
  K = size(offsets, 2)
  T = float(eltype(offsets))
  Points{K,T}(SVector{K}(ntuple(k -> SVector{2,T}(offsets[1, k], offsets[2, k]), K)))
end

npoints(::AbstractPoints{K}) where {K} = K

"""
    points_eltype(points, T)

`points` with the floating-point numbers it stores in element type `T`. The default returns `points`
unchanged, for strategies that store none.
"""
points_eltype(p::AbstractPoints, ::Type{T}) where {T} = p
points_eltype(p::GaussianPoints{K}, ::Type{T}) where {K,T} = GaussianPoints{K,T}(p.sigma_u, p.sigma_v, p.seed)
points_eltype(p::Points{K}, ::Type{T}) where {K,T} = Points{K,T}(map(o -> SVector{2,T}(o), p.offsets))

"""
    point_offset(points, T, iu, iv, view, k) -> (du, dv)

The offset of ray `k` (from `1` to `K`) from the center of the detector cell `(iu, iv)` in view
`view`, in element type `T` and in cell units (`±1/2` are the cell edges). `view` is numbered as in
the full geometry, so that subsets draw the same random offsets. Called in kernels: no allocations.
Fields of a strategy must be plain values: they reach the kernels unchanged, so an array field would
stay on the CPU.
"""
function point_offset end

@inline function point_offset(p::RegularPoints, ::Type{T}, iu, iv, view, k) where {T}
  ju, jv = rem(k - 1, p.su), div(k - 1, p.su)
  half = T(1 // 2)
  (ju + half) / p.su - half, (jv + half) / p.sv - half
end

@inline function point_offset(p::JitteredPoints, ::Type{T}, iu, iv, view, k) where {T}
  ju, jv = rem(k - 1, p.su), div(k - 1, p.su)
  ru, rv = random_pair(T, p.seed, iu, iv, view, k)
  half = T(1 // 2)
  (ju + ru) / p.su - half, (jv + rv) / p.sv - half
end

@inline function point_offset(p::RandomPoints, ::Type{T}, iu, iv, view, k) where {T}
  ru, rv = random_pair(T, p.seed, iu, iv, view, k)
  half = T(1 // 2)
  ru - half, rv - half
end

# normally distributed from two uniform numbers (Box-Muller); 1 - ru lies in (0, 1]
@inline function point_offset(p::GaussianPoints, ::Type{T}, iu, iv, view, k) where {T}
  ru, rv = random_pair(T, p.seed, iu, iv, view, k)
  r = sqrt(-2 * log(1 - ru))
  s, c = sincos(2 * T(pi) * rv)
  T(p.sigma_u) * r * c, T(p.sigma_v) * r * s
end

@inline function point_offset(p::Points, ::Type{T}, iu, iv, view, k) where {T}
  o = @inbounds p.offsets[k]
  T(o[1]), T(o[2])
end

"""
    random_pair(T, seed, iu, iv, view, k) -> (ru, rv)

Two uniform random numbers in `[0, 1)`, fixed by the arguments: the same on every device and in
every call, and independent for different arguments. `seed` is a `UInt32`, `(iu, iv)` the detector
cell, `view` the view; `k` is any number, usually the ray of the cell. For random point strategies.
"""
@inline function random_pair(::Type{T}, seed, iu, iv, view, k) where {T}
  # the cell as one number keeps the four inputs distinct for up to 65536 cells per row
  cell = (iu % UInt32) + ((iv % UInt32) << 16)
  hu, hv, _, _ = pcg4d(cell, view % UInt32, k % UInt32, seed)
  unit_interval(T, hu), unit_interval(T, hv)
end

"""
    UniformWeights()

All rays of a detector cell count equally. Weighting strategy of [`MultiRay`](@ref).
"""
struct UniformWeights <: AbstractWeights end

"""
    GaussianWeights(sigma_u, sigma_v=sigma_u)

Subrays weighted by a Gaussian of the offset from the cell center, with standard deviations
`sigma_u`, `sigma_v` in cell units: a detector response that falls off towards the cell edges.
Weighting strategy of [`MultiRay`](@ref).
"""
struct GaussianWeights{T} <: AbstractWeights
  sigma_u::T
  sigma_v::T
  function GaussianWeights{T}(sigma_u, sigma_v) where {T}
    sigma_u > 0 && sigma_v > 0 || throw(ArgumentError("standard deviations must be positive, got ($sigma_u, $sigma_v)"))
    new{T}(sigma_u, sigma_v)
  end
end

GaussianWeights(sigma_u::Real, sigma_v::Real=sigma_u) =
  GaussianWeights{float(promote_type(typeof(sigma_u), typeof(sigma_v)))}(sigma_u, sigma_v)

"""
    weights_eltype(weights, T)

`weights` with the floating-point numbers it stores in element type `T`. The default returns
`weights` unchanged, for strategies that store none.
"""
weights_eltype(w::AbstractWeights, ::Type{T}) where {T} = w
weights_eltype(w::GaussianWeights, ::Type{T}) where {T} = GaussianWeights{T}(w.sigma_u, w.sigma_v)

"""
    subray_weight(weights, du, dv)

The weight of a ray at offset `(du, dv)` from the cell center, not normalized: the kernel divides by
the sum of the weights of the cell, which must not be zero. Called in kernels: no allocations.
"""
function subray_weight end

@inline subray_weight(::UniformWeights, du, dv) = one(du)
@inline subray_weight(w::GaussianWeights, du, dv) = exp(-((du / w.sigma_u)^2 + (dv / w.sigma_v)^2) / 2)

# Whether the rays of a cell move along both detector axes or only along u. For images with one
# voxel layer along z the rays must stay in the image plane, which the projectors ensure with
# `in_plane`.
struct FullCell end
struct InPlane end

@inline restrict(::FullCell, du, dv) = (du, dv)
@inline restrict(::InPlane, du, dv) = (du, zero(dv))

"""
    MultiRay(geometry, points; weights=UniformWeights())

The detector geometry `geometry` (`ConeBeam`, `ArcConeBeam`, `ParallelBeam`) with several rays per
detector cell. The rays of a cell end at the offsets given by the point strategy `points`
([`RegularPoints`](@ref), [`JitteredPoints`](@ref), [`RandomPoints`](@ref),
[`GaussianPoints`](@ref), [`Points`](@ref), or a custom [`AbstractPoints`](@ref)) and
are averaged with the weights of `weights` ([`UniformWeights`](@ref),
[`GaussianWeights`](@ref), or a custom [`AbstractWeights`](@ref)), normalized to sum 1 per cell. A ray-driven projector built on it
computes

    y[i] = sum_k w_k * (line integral along ray k of data element i)

and the exact adjoint. This averages the line integrals over the cell (closer to what a detector
cell measures than one line through its center) and reduces the gaps between rays that occur when
detector cells are large compared to the voxels. The data have the size of `geometry`.

For images with one layer of voxels along `z` (2D images, or 3D images with `nz = 1`) the projectors
keep the rays in the image plane: they are spread along `u` only, and point strategies with several
subcells along `v` are rejected.

A `MultiRay` has no single ray per data element (no [`ray`](@ref)): a projector that traces only one
ray per cell fails with a `MethodError` when applied, instead of ignoring the other rays. `subset` and `adapt` keep the points and weights; random offsets stay the
same for the views of a subset.

# Example
```julia
g = MultiRay(circular_cone_beam(; angles, dso, dsd, nu, nv, du), JitteredPoints(2, 2; seed=1))
op = JosephProjector(image, g)
```
"""
struct MultiRay{T,G<:AbstractDetectorGeometry{T},P,W,V,S} <: AbstractProjectionGeometry{T}
  geometry::G
  points::P
  weights::W
  # nothing, or the numbers of the views in the geometry this one is a subset of, stored like the
  # other arrays of the geometry (element type T: exact for up to 2^24 views in Float32)
  views::V
  span::S   # FullCell() or InPlane()
end

function MultiRay(geometry, points::AbstractPoints; weights::AbstractWeights=UniformWeights())
  g = projection_geometry(geometry)
  g isa AbstractDetectorGeometry ||
    throw(ArgumentError("MultiRay needs a detector geometry (cells to sample), got $(nameof(typeof(g)))"))
  multiray(g, points, weights, nothing, FullCell())
end

function multiray(g::AbstractDetectorGeometry{T}, points, weights, views, span) where {T}
  P, W = points_eltype(points, T), weights_eltype(weights, T)
  MultiRay{T,typeof(g),typeof(P),typeof(W),typeof(views),typeof(span)}(g, P, W, views, span)
end

nsubrays(g::MultiRay) = npoints(g.points)
datasize(g::MultiRay) = datasize(g.geometry)
raytype(g::MultiRay) = raytype(g.geometry)
nviews(g::MultiRay) = nviews(g.geometry)
detectorsize(g::MultiRay) = detectorsize(g.geometry)
storage_array(g::MultiRay) = storage_array(g.geometry)

function subset(g::MultiRay{T}, views) where {T}
  numbers = isnothing(g.views) ? collect(views) : Array(g.views)[views]
  inner = subset(g.geometry, views)
  multiray(inner, g.points, g.weights, copyto!(geometry_array(inner, length(numbers)), T.(numbers)), g.span)
end

Adapt.adapt_structure(to, g::MultiRay) =
  multiray(adapt(to, g.geometry), g.points, g.weights, adapt(to, g.views), g.span)

"""
    in_plane(geometry)

`geometry` with the rays of a [`MultiRay`](@ref) in the image plane (offsets along `u` only); other
geometries unchanged. The constructor of a projector calls it for images with one layer of voxels
along `z`, see [`CTOperators.one_layer`](@ref).
"""
in_plane(g) = g
function in_plane(g::MultiRay)
  check_in_plane(g.points)
  multiray(g.geometry, g.points, g.weights, g.views, InPlane())
end

# subcells along v would all fall onto v = 0 and trace the same rays several times
"""
    one_layer(image::ImageGeom)

Whether `image` has one layer of voxels along `z`: a 2D image, or a 3D image with `nz = 1`. Rays that
leave the center of the layer see too little of it, so the rays of a cell stay in its plane.
"""
one_layer(image::ImageGeom) = length(image.dims) == 2 || image.dims[3] == 1

check_in_plane(p::Union{RegularPoints,JitteredPoints}) =
  p.sv == 1 || throw(ArgumentError("an image with one layer of voxels needs one row of rays: use $(nameof(typeof(p)))(su, 1)"))
check_in_plane(p) = nothing

function Base.show(io::IO, ::MIME"text/plain", g::MultiRay)
  print(io, "MultiRay with ", nsubrays(g), " rays per cell (", nameof(typeof(g.points)), ", ",
        nameof(typeof(g.weights)), ") of ")
  show(io, MIME("text/plain"), g.geometry)
end

# the view number in the full geometry
@inline full_view(::Nothing, view) = view
@inline full_view(views, view) = @inbounds unsafe_trunc(Int, views[view])

# the cell of data element i, once per element: its detector indices, the view and the view number
# in the full geometry
@inline function element_cell(g::MultiRay, i)
  nu, nv = detectorsize(g.geometry)
  iu, iv, view = pixel_index(i, nu, nv)
  iu, iv, view, full_view(g.views, view)
end

# the offset of ray k in the cell
@inline function subray_offset(g::MultiRay{T}, cell, k) where {T}
  iu, iv, _, number = cell
  restrict(g.span, point_offset(g.points, T, iu, iv, number, k)...)
end

# the weight (not normalized) of ray k in the cell
@inline function cell_weight(g::MultiRay, cell, k)
  du, dv = subray_offset(g, cell, k)
  subray_weight(g.weights, du, dv)
end

# ray k in the cell
@inline function cell_subray(g::MultiRay{T}, cell, k) where {T}
  iu, iv, view, _ = cell
  du, dv = subray_offset(g, cell, k)
  a, b = cell_ray(g.geometry, T(iu) + du, T(iv) + dv, view)
  a, b, subray_weight(g.weights, du, dv)
end

"""
    subray(g::MultiRay, i, k) -> (a, b, w)

Ray `k` of data element `i` as two points like [`ray`](@ref), and its weight (not normalized).
Callable inside kernels.
"""
@inline subray(g::MultiRay, i, k) = cell_subray(g, element_cell(g, i), k)
