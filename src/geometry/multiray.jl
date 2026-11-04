# Several rays per detector cell: a detector geometry wrapped with a point strategy (where in the
# cell the subrays end) and a weighting strategy. Offsets are in cell units: (0, 0) is the cell
# center, +-1/2 the cell edges. The number of subrays K is a type parameter of the point strategy,
# so the loop over the subrays has a length known to the compiler.

"""
    RegularPoints(su, sv=su)

The centers of `su × sv` subcells of each detector cell: `su` along `u`, `sv` along `v`. Point
strategy of [`MultiRay`](@ref).
"""
struct RegularPoints{K}
  su::Int
  sv::Int
end

"""
    JitteredPoints(su, sv=su; seed=0)

The points of [`RegularPoints`](@ref), each moved to a random position within its subcell
("jittered sampling", R. L. Cook, "Stochastic sampling in computer graphics", ACM Transactions on
Graphics 5(1), 51-72, 1986). The positions are fixed by `seed`, the detector cell, the view and the
subray, so they are the same in the forward and the back projection, on every device and in
`subset`. Point strategy of [`MultiRay`](@ref).
"""
struct JitteredPoints{K}
  su::Int
  sv::Int
  seed::UInt32
end

"""
    RandomPoints(n; seed=0)

`n` random points in each detector cell, fixed like those of [`JitteredPoints`](@ref). Point
strategy of [`MultiRay`](@ref).
"""
struct RandomPoints{K}
  seed::UInt32
end

"""
    GaussianPoints(n, sigma_u, sigma_v=sigma_u; seed=0)

`n` random points per detector cell, normally distributed around the cell center with standard
deviations `sigma_u`, `sigma_v` in cell units: a detector response set by where the rays land, with
all rays counting equally. Points can land outside the cell, which models blur across neighboring
cells. The positions are fixed like those of [`JitteredPoints`](@ref). Point strategy of
[`MultiRay`](@ref).
"""
struct GaussianPoints{K,T}
  sigma_u::T
  sigma_v::T
  seed::UInt32
end

"""
    Points(offsets)

Given points, the same in every detector cell: column `k` of the `2 × K` matrix `offsets` is the
position of subray `k` in cell units, `(0, 0)` the cell center and `±1/2` the cell edges. Points
outside the cell are allowed. Point strategy of [`MultiRay`](@ref).
"""
struct Points{K,T}
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

npoints(::Union{RegularPoints{K},JitteredPoints{K},RandomPoints{K},GaussianPoints{K},Points{K}}) where {K} = K

# the points in element type T (given and Gaussian points store numbers)
points_eltype(p, ::Type{T}) where {T} = p
points_eltype(p::GaussianPoints{K}, ::Type{T}) where {K,T} = GaussianPoints{K,T}(p.sigma_u, p.sigma_v, p.seed)
points_eltype(p::Points{K}, ::Type{T}) where {K,T} = Points{K,T}(map(o -> SVector{2,T}(o), p.offsets))

# The offset (du, dv) of subray k in the cell (iu, iv) of view `view` (numbered as in the full
# geometry, so that subsets draw the same random positions).
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

# two numbers in [0, 1) from the hash of (cell, view, subray, seed); the cell as one number
# (iu + 65536 iv) keeps the four inputs distinct for detectors of up to 65536 cells per row
@inline function random_pair(::Type{T}, seed, iu, iv, view, k) where {T}
  cell = (iu % UInt32) + ((iv % UInt32) << 16)
  hu, hv, _, _ = pcg4d(cell, view % UInt32, k % UInt32, seed)
  unit_interval(T, hu), unit_interval(T, hv)
end

"""
    UniformWeights()

All subrays of a detector cell count equally. Weighting strategy of [`MultiRay`](@ref).
"""
struct UniformWeights end

"""
    GaussianWeights(sigma_u, sigma_v=sigma_u)

Subrays weighted by a Gaussian of the offset from the cell center, with standard deviations
`sigma_u`, `sigma_v` in cell units: a detector response that falls off towards the cell edges.
Weighting strategy of [`MultiRay`](@ref).
"""
struct GaussianWeights{T}
  sigma_u::T
  sigma_v::T
  function GaussianWeights{T}(sigma_u, sigma_v) where {T}
    sigma_u > 0 && sigma_v > 0 || throw(ArgumentError("standard deviations must be positive, got ($sigma_u, $sigma_v)"))
    new{T}(sigma_u, sigma_v)
  end
end

GaussianWeights(sigma_u::Real, sigma_v::Real=sigma_u) =
  GaussianWeights{float(promote_type(typeof(sigma_u), typeof(sigma_v)))}(sigma_u, sigma_v)

weights_eltype(w, ::Type{T}) where {T} = w
weights_eltype(w::GaussianWeights, ::Type{T}) where {T} = GaussianWeights{T}(w.sigma_u, w.sigma_v)

# the (unnormalized) weight of a subray at offset (du, dv)
@inline subray_weight(::UniformWeights, du, dv) = one(du)
@inline subray_weight(w::GaussianWeights, du, dv) = exp(-((du / w.sigma_u)^2 + (dv / w.sigma_v)^2) / 2)

# Whether the subrays of a cell move along both detector axes or only along u. For 2D images (one
# voxel layer along z) the rays must stay in the image plane, which the projectors ensure with
# `in_plane`.
struct FullCell end
struct InPlane end

@inline restrict(::FullCell, du, dv) = (du, dv)
@inline restrict(::InPlane, du, dv) = (du, zero(dv))

"""
    MultiRay(geometry, points; weights=UniformWeights())

The detector geometry `geometry` (`ConeBeam`, `ArcConeBeam`, `ParallelBeam`) with several rays per
detector cell. The rays of a cell end at the positions given by the point strategy `points`
([`RegularPoints`](@ref), [`JitteredPoints`](@ref), [`RandomPoints`](@ref),
[`GaussianPoints`](@ref), [`Points`](@ref)) and
are averaged with the weights of `weights` ([`UniformWeights`](@ref),
[`GaussianWeights`](@ref)), normalized to sum 1 per cell. A ray-driven projector built on it
computes

    y[i] = sum_k w_k * (line integral along subray k of data element i)

and the exact adjoint. This averages the line integrals over the cell (closer to what a detector
cell measures than one line through its center) and removes the gaps between rays that occur when
detector cells are large against the voxels. The data has the size of `geometry`.

For 2D images (one voxel layer along `z`) the projectors keep the subrays in the image plane: they
are spread along `u` only, and point strategies with several subcells along `v` are rejected.

A `MultiRay` has no single ray per data element (no [`ray`](@ref)); projectors that trace only one
ray per cell reject it. `subset` and `adapt` keep the points and weights; random positions stay the
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

function MultiRay(geometry, points; weights=UniformWeights())
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

# the geometry with its subrays in the plane v = 0, for 2D images (plain geometries are unchanged)
in_plane(g) = g
function in_plane(g::MultiRay)
  check_in_plane(g.points)
  multiray(g.geometry, g.points, g.weights, g.views, InPlane())
end

# subcells along v would all fall onto v = 0 and trace the same rays several times
check_in_plane(p::Union{RegularPoints,JitteredPoints}) =
  p.sv == 1 || throw(ArgumentError("a 2D image has one row of rays per view: use $(nameof(typeof(p)))(su, 1)"))
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

# the offset of subray k in the cell
@inline function subray_offset(g::MultiRay{T}, cell, k) where {T}
  iu, iv, _, number = cell
  restrict(g.span, point_offset(g.points, T, iu, iv, number, k)...)
end

# the weight (not normalized) of subray k in the cell
@inline function subray_weight(g::MultiRay, cell, k)
  du, dv = subray_offset(g, cell, k)
  subray_weight(g.weights, du, dv)
end

# subray k in the cell
@inline function cell_subray(g::MultiRay{T}, cell, k) where {T}
  iu, iv, view, _ = cell
  du, dv = subray_offset(g, cell, k)
  a, b = cell_ray(g.geometry, T(iu) + du, T(iv) + dv, view)
  a, b, subray_weight(g.weights, du, dv)
end

"""
    subray(g::MultiRay, i, k) -> (a, b, w)

Subray `k` of data element `i` as two points like [`ray`](@ref), and its weight (not normalized).
Callable inside kernels.
"""
@inline subray(g::MultiRay, i, k) = cell_subray(g, element_cell(g, i), k)
