# Redundancy weighting of FDK: how often a scan measures each line, and how the measurements of a line
# are combined. It decides which scans FDK accepts, weights every data value before the filter, and
# gives the factor of every view in the back projection.

"""
    AbstractRedundancy

How [`FDK`](@ref) combines the measurements of lines that a scan measures more than once (keyword
`redundancy`). A full scan measures every line twice, from opposite sides; other scans measure some
lines once and some twice, and need weights that depend on the ray. Subtypes: [`FullScan`](@ref)
(the default) and [`ArrayRedundancy`](@ref) (weights given as arrays).

A redundancy weighting checks whether it can reconstruct a scan, and gives two kinds of weights:
a weight per data value (cell and view), applied before the ramp filter, and a weight per view,
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

The weights of the redundancy weighting `r` for the scan `scan` of `geometry` (on the host), see
[`AbstractRedundancy`](@ref).
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
circle (parallel beams: half or once around), so every line is measured equally often, and each view
is weighted with its angle step: half the angle between its two neighbors, the first and the last
view taking the gap that closes the circle as the missing neighbor. Uneven angle steps are allowed;
the gap must be at most 1.5 times the largest step. All data values have the same weight.
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
                        "$(round(maximum(d); sigdigits=4)). Shorter scans need a redundancy weighting for them (keyword redundancy)"))
  gap = max(gap, 0.0)
  steps = [(k == 1 ? gap : d[k-1]) / 2 + (k == length(d) + 1 ? gap : d[k]) / 2 for k in eachindex(scan.angles)]
  UniformRedundancy(), (pi / coverage) .* steps
end

"""
    ArrayRedundancy(weights, views)

A redundancy weighting given by arrays, for weights computed elsewhere: `weights` the weight of
each data value, an array of size `(nu, nv, nviews)` (the data size of the geometry), where a
dimension of length 1 stands for all cells or views along it (as in broadcasting; `(nu, 1, nviews)`:
the same weight for every row), and `views` a vector with the weight of each view (see
[`AbstractRedundancy`](@ref) for how both enter, and for the condition they must meet). Only the
sizes are checked: the weights must suit the scan.

`weights` is moved to the device of the plan with `adapt(storage, weights)` and keeps its element
type (on devices without `Float64`, pass `Float32` weights); the kernel converts each weight to the
element type of the plan.

# Example

```julia
# a full scan with 360 equal steps as arrays: weight 1 for all data values, half the angle step per view
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
