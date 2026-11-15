# Redundancy weighting of FDK: how often a scan measures each line, and how the measurements of a line
# are combined. It decides which scans FDK accepts, weights every data element before the filter, and
# gives the factor of every view in the back projection.

"""
    AbstractRedundancy

How [`FDK`](@ref) combines the measurements of lines that a scan measures more than once (keyword
`redundancy`). A full scan measures every line twice, from opposite sides; other scans measure some
lines once and some twice, and need weights that depend on the ray. Subtypes: [`FullScan`](@ref)
(the default), [`ParkerWeights`](@ref) (short scans) and [`ArrayRedundancy`](@ref) (weights given
as arrays).

A redundancy weighting checks whether it can reconstruct a scan, and gives two kinds of weights:
a weight per data element (cell and view), applied before the ramp filter, and a weight per view,
applied in the back projection with the angle step. Interface:

* [`redundancy_weights`](@ref CTOperators.redundancy_weights)`(r, scan, geometry) -> (weight,
  views)`, on the host: `weight` is the object that the kernel passes to
  [`redundancy_weight`](@ref CTOperators.redundancy_weight) (`isbits`, or with arrays and an
  `Adapt.adapt_structure` method; it is moved to the device with `adapt(storage, weight)`), and
  `views` a vector with the weight of each view. Throws an `ArgumentError` for scans it cannot
  handle.
* [`redundancy_weight`](@ref CTOperators.redundancy_weight)`(weight, geometry, iu, iv, k)`: the
  weight of cell `(iu, iv)` in view `k`; called in kernels.

FDK back projects view `k` (the data times the cosine weight and `weight`, filtered with the ramp)
with the factor `views[k] · dso_k / dsd_k` times the squared magnification of the voxel. The weights
must split every line among its measurements: the product of `weight` and `views[k]`, divided by
the angle step of the view, must sum to 1 over the measurements of each line. A full scan measures
every line twice and weights each view with half its angle step; a parallel-beam scan over half the
circle measures every line once and weights each view with its full angle step.
"""
abstract type AbstractRedundancy end

"""
    redundancy_weights(r::AbstractRedundancy, scan::CircularScan, geometry) -> (weight, views)

The weights of the redundancy weighting `r` for the scan `scan` of `geometry`, see
[`AbstractRedundancy`](@ref). Runs on the host; `geometry` is that of the plan, in its element type
and on its device, `scan` is computed from it on the host in `Float64`.
"""
function redundancy_weights end

"""
    redundancy_weight(weight, geometry, iu, iv, k)

The weight of cell `(iu, iv)` in view `k` before the ramp filter, in the element type of
`geometry`; `weight` is the first result of [`redundancy_weights`](@ref
CTOperators.redundancy_weights). Called in kernels.
"""
function redundancy_weight end

"""
    FullScan()

The redundancy weighting of a full scan (the default of [`FDK`](@ref)): the views go once around the
circle (parallel beams: half or once around), so every line is measured equally often. The angle
step of a view is half the angle between its two neighbors, the first and the last view taking the
gap that closes the circle as the missing neighbor; each view is weighted with half its angle step
(every line measured twice), parallel beams over half the circle with the full step (every line
measured once). Uneven angle steps are allowed; the gap must be at most 1.5 times the largest step.
All data elements have the same weight.
"""
struct FullScan <: AbstractRedundancy end

# the kernel side of FullScan: weight 1
struct UniformRedundancy end

@inline redundancy_weight(::UniformRedundancy, geometry::AbstractDetectorGeometry{T}, iu, iv, k) where {T} = one(T)

function redundancy_weights(::FullScan, scan::CircularScan, geometry)
  span = angular_span(scan)
  coverage = geometry isa ParallelBeam && span < 1.5pi ? 1.0pi : 2.0pi
  d = abs.(diff(scan.angles))
  gap = coverage - span
  -1e-6 * coverage <= gap <= 1.5 * maximum(d) ||
    throw(ArgumentError("FDK with FullScan() needs views around the whole circle ($(coverage == 2pi ? "2pi" : "pi or " *
                        "2pi for parallel beams")); they cover $(round(span; sigdigits=4)) with steps up to " *
                        "$(round(maximum(d); sigdigits=4)). Shorter scans need redundancy=ParkerWeights()"))
  gap = max(gap, 0.0)
  steps = [(k == 1 ? gap : d[k-1]) / 2 + (k == length(d) + 1 ? gap : d[k]) / 2 for k in eachindex(scan.angles)]
  UniformRedundancy(), (pi / coverage) .* steps
end

"""
    ArrayRedundancy(weights, views)

A redundancy weighting given by arrays, for weights computed elsewhere: `weights` the weight of
each data element, an array of size `(nu, nv, nviews)` (the data size of the geometry), where a
dimension of length 1 stands for all cells or views along it (as in broadcasting; `(nu, 1, nviews)`:
the same weight for every row), and `views` a vector with the weight of each view (see
[`AbstractRedundancy`](@ref) for how both enter, and for the condition they must meet). Only the
sizes are checked: the weights must suit the scan.

`weights` is moved to the device of the plan with `adapt(storage, weights)` and keeps its element
type (on devices without `Float64`, pass `Float32` weights); the kernel converts each weight to the
element type of the plan.

# Example

```julia
# a full scan with 360 equal steps as arrays: weight 1 for all data elements, half the angle step per view
geometry = circular_cone_beam(; angles=range(0, 2pi, 361)[1:360], dso=500, dsd=1000, nu=256, nv=128)
plan = FDK(image, geometry; redundancy=ArrayRedundancy(ones(1, 1, 1), fill(pi / 360, 360)))
```
"""
struct ArrayRedundancy{A<:AbstractArray{<:Real,3},V<:AbstractVector{<:Real}} <: AbstractRedundancy
  weights::A
  views::V
end

# the kernel side of ArrayRedundancy: the array of weights on the device
struct ArrayWeight{A}
  weights::A
end

Adapt.@adapt_structure ArrayWeight

@inline function redundancy_weight(w::ArrayWeight, geometry::AbstractDetectorGeometry{T}, iu, iv, k) where {T}
  a = w.weights
  T(a[ifelse(size(a, 1) == 1, 1, iu), ifelse(size(a, 2) == 1, 1, iv), ifelse(size(a, 3) == 1, 1, k)])
end

function redundancy_weights(r::ArrayRedundancy, scan::CircularScan, geometry)
  dims = (detectorsize(geometry)..., nviews(geometry))
  all(size(r.weights, i) in (1, dims[i]) for i in 1:3) ||
    throw(DimensionMismatch("ArrayRedundancy: weights of size $(size(r.weights)) do not fit the data size $dims " *
                            "(each dimension must have that length or 1)"))
  length(r.views) == dims[3] ||
    throw(DimensionMismatch("ArrayRedundancy: $(length(r.views)) view weights for $(dims[3]) views"))
  ArrayWeight(r.weights), Vector{Float64}(Array(r.views))
end

"""
    ParkerWeights()

The redundancy weighting of a short scan (Parker 1982): views over at least half the circle plus the
fan angle, in order, with angle steps that may differ. Lines near the start and the end of the scan
are measured twice, from views about half a circle apart; their two data elements get weights that
add up to 1 and fall smoothly to 0 at the first and the last view. A data element at the angle `β` of
its view (from the first view, along the orbit) and the fan angle `γ` of its cell (the angle between
its ray and the central ray, within the plane of the orbit) gets the weight

* `sin²(π β / (4 (Δ - γ)))` for `0 ≤ β < 2 (Δ - γ)`,
* `1` up to `β = π - 2γ`,
* `sin²(π (π + 2Δ - β) / (4 (Δ + γ)))` up to `β = π + 2Δ`,

where `π + 2Δ` is the angle between the first and the last view. Parker's weights take `Δ` as half
the fan angle, the shortest scan; for longer scans, up to one turn, `Δ = (span - π) / 2` keeps the
weights of a line adding up to 1 (Wesarg et al. 2002, Eq. 15). Each view is weighted with its angle
step in the back projection. A scan must cover at least `π` plus the fan angle of the detector
(twice the largest `|γ|` of its cells); a parallel-beam scan at least `π` (no fan: `γ = 0`).

In a cone beam the weights are those of the fan in the plane of the orbit, for every row; Wesarg et
al. (2002) state that the weights carry over to the circular cone beam with Feldkamp's
approximation. With a detector offset, the columns beyond the symmetric part of the fan see lines
whose second measurement falls off the detector; their weights assume both measurements. With data
that are not smooth at the first and the last view (an object that reaches out of the field of view),
the reconstruction shows streaks; a slightly longer scan avoids the jump of the weights at the edges
of the fan (Wesarg et al. 2002).

References:
* D. L. Parker, "Optimal short scan convolution reconstruction for fanbeam CT", Med. Phys. 9(2),
  254–257, 1982 (Eq. 12; the denominator of its third case must read `δ + α`, as Eq. 9 requires).
* S. Wesarg, M. Ebert, T. Bortfeld, "Parker weights revisited", Med. Phys. 29(3), 372–378, 2002.
"""
struct ParkerWeights <: AbstractRedundancy end

# The weight of Parker (1982) in the form of Wesarg et al. (2002, Eq. 15): view angle beta from the
# first view, fan angle gamma, the scan from 0 to pi + 2 delta. The branches exclude the points where
# a denominator is 0 (gamma = +-delta at the ends of the scan).
@inline function parker_weight(beta::T, gamma::T, delta::T) where {T}
  if beta < 2 * (delta - gamma)
    sinpi(beta / (4 * (delta - gamma)))^2
  elseif beta <= T(pi) - 2 * gamma
    one(T)
  elseif beta <= T(pi) + 2 * delta
    sinpi((T(pi) + 2 * delta - beta) / (4 * (delta + gamma)))^2
  else
    zero(T)
  end
end

# the kernel side of ParkerWeights: per view the angle from the first view and the central ray, the
# rotation axis, the direction of rotation (+1: angles increasing about the axis) and delta
struct ParkerWeight{T,V<:AbstractVector{T},C}
  beta::V
  central::C        # 3 x nviews, nothing for parallel beams
  axis::SVector{3,T}
  orientation::T
  delta::T
end

Adapt.@adapt_structure ParkerWeight

# The fan angle of cell (iu, iv) in view k: the angle from the central ray to the ray of the cell,
# within the plane of the orbit, positive about the rotation axis; 0 for parallel beams.
@inline fan_angle(::ParallelBeam{T}, w::ParkerWeight, iu, iv, k) where {T} = zero(T)

@inline function fan_angle(g::AbstractDetectorGeometry{T}, w::ParkerWeight, iu, iv, k) where {T}
  a, b = cell_ray(g, T(iu), T(iv), k)
  d = b - a
  d = d - dot(d, w.axis) * w.axis
  c = column(w.central, k)
  atan(dot(cross(c, d), w.axis), dot(c, d))
end

@inline function redundancy_weight(w::ParkerWeight, geometry::AbstractDetectorGeometry{T}, iu, iv, k) where {T}
  gamma = w.orientation * fan_angle(geometry, w, iu, iv, k)
  parker_weight(T(w.beta[k]), T(gamma), T(w.delta))
end

function redundancy_weights(::ParkerWeights, scan::CircularScan, geometry::AbstractDetectorGeometry{T}) where {T}
  h = adapt(Array{Float64}, geometry)
  span = angular_span(scan)
  span <= 2pi * (1 + 1e-9) ||
    throw(ArgumentError("FDK with ParkerWeights() needs views within one turn; they cover $(round(span; sigdigits=4))"))
  orientation = scan.angles[end] > scan.angles[1] ? 1.0 : -1.0
  beta = abs.(scan.angles .- scan.angles[1])
  central = h isa ParallelBeam ? nothing : scan.central
  host = ParkerWeight(beta, central, scan.axis, orientation, 0.0)
  # half the fan angle: the largest |gamma| of the outermost columns in all views
  nu = detectorsize(h)[1]
  fan = maximum(abs(fan_angle(h, host, iu, 1, k)) for iu in (1, nu), k in 1:nviews(h))
  delta = (span - pi) / 2
  # rtol as in the checks of the scan: the geometry may be in Float32
  delta >= fan * (1 - 1e-4) ||
    throw(ArgumentError("FDK with ParkerWeights() needs views over at least pi plus the fan angle " *
                        "($(round(pi + 2 * fan; sigdigits=4))); they cover $(round(span; sigdigits=4))"))
  d = abs.(diff(scan.angles))
  steps = [((k == 1 ? 0.0 : d[k-1]) + (k == length(d) + 1 ? 0.0 : d[k])) / 2 for k in eachindex(scan.angles)]
  backend = KA.get_backend(storage_array(geometry))
  device(a) = isnothing(a) ? nothing : copyto!(KA.allocate(backend, T, size(a)), T.(a))
  ParkerWeight(device(beta), device(central), SVector{3,T}(scan.axis), T(orientation), T(delta)), steps
end
