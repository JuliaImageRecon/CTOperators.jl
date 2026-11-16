# What a measurement averages over, besides the line integral along one ray: the samples of the
# focal spot (where the rays start) and of the detector cell (where they end), each a point
# strategy with a weighting strategy. `sampled` (operators/sampled.jl) turns them into operators.

"""
    AbstractSampling

Supertype of the sampling descriptions of [`sampled`](@ref) and [`stacked`](@ref):
[`FocalSpot`](@ref) (the source side) and [`DetectorSampling`](@ref) (the detector side).
"""
abstract type AbstractSampling end

"""
    FocalSpot(fu, fv=fu; points, weights=UniformWeights())

A focal spot of size `fu × fv` (in the length unit of the image): a flat patch through the source,
parallel to a flat detector ([`ConeBeam`](@ref)) or perpendicular to the central ray for a detector
on an arc ([`ArcConeBeam`](@ref)), with `fu` along the detector rows (`u`) and `fv` along the columns
(`v`). The rays of a view start at the points of `points` (offsets in units of the spot size, `±1/2`
at its edges: [`RegularPoints`](@ref) for a rectangle, [`DiskPoints`](@ref) for a circle or ellipse,
or any [`AbstractPoints`](@ref) evaluated once per view), weighted with `weights`
([`UniformWeights`](@ref), [`GaussianWeights`](@ref) in the same units, or any
[`AbstractWeights`](@ref)), normalized to sum 1 per view.

The keyword `points` has no default: one point at the center would be a point source.

For images one voxel thick in `z` the spot is sampled along `u` only: `RegularPoints` and
`JitteredPoints` need one row (`RegularPoints(n, 1)`), `DiskPoints` are rejected, other strategies
get offsets along `v` of 0.
"""
struct FocalSpot{P<:AbstractPoints,W<:AbstractWeights} <: AbstractSampling
  fu::Float64
  fv::Float64
  points::P
  weights::W
end

function FocalSpot(fu::Real, fv::Real=fu; points::AbstractPoints, weights::AbstractWeights=UniformWeights())
  fu >= 0 && fv >= 0 || throw(ArgumentError("the size of the focal spot must not be negative, got ($fu, $fv)"))
  FocalSpot(Float64(fu), Float64(fv), points, weights)
end

"""
    DetectorSampling(points; weights=UniformWeights())

Several rays per detector cell, ending at the offsets of `points` (in cell units, `±1/2` at the cell
edges) and averaged with the ray weights of `weights`, normalized to sum 1 per cell.
"""
struct DetectorSampling{P<:AbstractPoints,W<:AbstractWeights} <: AbstractSampling
  points::P
  weights::W
end

DetectorSampling(points::AbstractPoints; weights::AbstractWeights=UniformWeights()) = DetectorSampling(points, weights)

npoints(s::AbstractSampling) = npoints(s.points)

function Base.show(io::IO, s::FocalSpot)
  print(io, "FocalSpot(", s.fu, ", ", s.fv, "; ", npoints(s), " points, ", nameof(typeof(s.points)), ", ",
        nameof(typeof(s.weights)), ")")
end

function Base.show(io::IO, s::DetectorSampling)
  print(io, "DetectorSampling(", npoints(s), " points, ", nameof(typeof(s.points)), ", ", nameof(typeof(s.weights)), ")")
end

# The offsets of sample k of a sampling in every view (in point units, restricted to u for an image
# one voxel thick in z) and their weights normalized over the samples of the view: a 2 x nviews
# matrix per sample and a nviews x K matrix of weights, on the host in Float64. `views` numbers the
# views as in the full geometry, so that random offsets do not depend on subsets.
function sample_offsets(s::AbstractSampling, views, span)
  K, n = npoints(s), length(views)
  offsets = [zeros(2, n) for _ in 1:K]
  weights = zeros(n, K)
  for (j, view) in enumerate(views), k in 1:K
    du, dv = restrict(span, point_offset(s.points, Float64, 0, 0, view, k)...)
    offsets[k][:, j] .= (du, dv)
    weights[j, k] = subray_weight(s.weights, du, dv)
  end
  weights ./= sum(weights; dims=2)
  offsets, weights
end

# The frame of the focal spot in view k of a host geometry (Float64): unit vectors along which fu
# and fv extend. A flat detector: along its rows and columns, so that the spot is parallel to it. An
# arc: perpendicular to the central ray, along the row tangent and the column direction.
function spot_frame(g::ConeBeam, k)
  normalize(column(g.u, k)), normalize(column(g.v, k))
end

function spot_frame(g::ArcConeBeam, k)
  w = normalize(column(g.center, k) - column(g.source, k))
  u = column(g.u, k)
  eu = normalize(u - dot(u, w) * w)
  ev = cross(w, eu)
  eu, dot(ev, column(g.v, k)) < 0 ? -ev : ev
end

# the source offsets (3 x nviews, lengths) of the point offsets `o` (2 x nviews, point units)
function spot_offsets(spot::FocalSpot, g, o)
  m = zeros(3, nviews(g))
  for k in 1:nviews(g)
    eu, ev = spot_frame(g, k)
    m[:, k] = spot.fu * o[1, k] * eu + spot.fv * o[2, k] * ev
  end
  m
end
