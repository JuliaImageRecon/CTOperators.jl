# The sampling of the beam (geometry/sampling.jl) as operators. A projector either samples in its
# kernel ("fused": the detector sampling of the ray-driven projectors, through the internal
# MultiRay geometry) or the operator sums projectors on shifted geometries, one per sample
# ("sequential"). Which projector can do what is asked by dispatch (`supports_sampling`,
# `fuses_sampling`), so that projectors of other packages opt in with a method.

# the outputs of a sampling: the weighted sum of the samples, or the samples side by side
struct Summed end
struct Stacked end

"""
    supports_sampling(op::AbstractProjector, geometry, sampling) -> Bool

Whether the projector `op` on `geometry` can sample `sampling` (a [`FocalSpot`](@ref) or a
[`DetectorSampling`](@ref)) at all, fused or sequentially. `false` by default; `true` for focal
spots on `ConeBeam` and `ArcConeBeam` and for detector sampling on detector geometries, except for
the distance-driven projector, whose footprint already covers the cell. A projector of another
package refuses a sampling with a method returning `false`.
"""
supports_sampling(op::AbstractProjector, g::AbstractProjectionGeometry, s::AbstractSampling) = false
supports_sampling(op::AbstractProjector, g::Union{ConeBeam,ArcConeBeam}, ::FocalSpot) = true
supports_sampling(op::AbstractProjector, g::AbstractDetectorGeometry, ::DetectorSampling) = true
supports_sampling(op::DistanceDrivenProjector, g::AbstractDetectorGeometry, ::DetectorSampling) = false

"""
    fuses_sampling(op::AbstractProjector, geometry, sampling, output) -> Bool

Whether the kernel of `op` samples `sampling` itself, in one pass, for the output `output`
(`CTOperators.Summed()`: the weighted mean, as [`sampled`](@ref) computes it). `false` by default;
`true` for the detector sampling of [`SiddonProjector`](@ref) and [`JosephProjector`](@ref) on
detector geometries. A projector that traces through [`CTOperators.trace_element`](@ref) declares it
with `fuses_sampling(::MyProjector, ::AbstractDetectorGeometry, ::DetectorSampling, ::CTOperators.Summed) = true`.
"""
fuses_sampling(op, g, s, output) = false
fuses_sampling(::Union{SiddonProjector,JosephProjector}, ::AbstractDetectorGeometry, ::DetectorSampling, ::Summed) = true

# The projectors and the per-view weights of the terms of a sampled operator: one term per
# combination of a source sample and a detector sample run sequentially (source outside, detector
# inside); the detector samples of a projector that fuses them stay in one term.
function sample_terms(op::AbstractProjector, source, detector, execution::Symbol, output)
  execution in (:auto, :fused, :sequential) ||
    throw(ArgumentError("execution must be :auto, :fused or :sequential, got :$execution"))
  g = geometry(op)
  name = nameof(typeof(op))
  for s in (source, detector)
    isnothing(s) || s isa AbstractSampling || throw(ArgumentError("a sampling is a FocalSpot or a DetectorSampling, got $(typeof(s))"))
    isnothing(s) || supports_sampling(op, g, s) ||
      throw(ArgumentError("$name on a $(nameof(typeof(g))) does not support $(nameof(typeof(s)))"))
  end
  flat = one_layer(image_geometry(op))
  span = flat ? InPlane() : FullCell()
  for s in (source, detector)
    flat && !isnothing(s) && check_in_plane(s.points)
  end
  fused = !isnothing(detector) && execution !== :sequential && fuses_sampling(op, g, detector, output)
  if execution === :fused
    isnothing(source) ||
      throw(ArgumentError("no projector samples a FocalSpot in its kernel: use execution=:auto or :sequential"))
    isnothing(detector) || fused ||
      throw(ArgumentError("$name does not sample DetectorSampling in its kernel: use execution=:auto or :sequential"))
  end
  if !isnothing(detector) && !fused && varies_per_cell(detector.points)
    throw(ArgumentError("$(nameof(typeof(detector.points))) varies per cell and needs a projector that samples it in its " *
                        "kernel (SiddonProjector, JosephProjector); $name samples sequentially, one geometry per offset"))
  end
  T = eltype(op)
  n = nviews(g)
  device(m) = copyto!(geometry_array(g, size(m)...), T.(m))
  sources = if isnothing(source)
    [(nothing, ones(n))]
  else
    host = adapt(Array{Float64}, g)
    offsets, weights = sample_offsets(source, 1:n, span)
    [(device(spot_offsets(source, host, offsets[k])), weights[:, k]) for k in 1:npoints(source)]
  end
  detectors = if isnothing(detector) || fused
    [(nothing, ones(n))]
  else
    offsets, weights = sample_offsets(detector, 1:n, span)
    [(device(offsets[l]), weights[:, l]) for l in 1:npoints(detector)]
  end
  terms = []
  for (delta, ws) in sources, (epsilon, wd) in detectors
    gs = shift(g; source=delta, detector=epsilon)
    gk = fused ? multiray(gs, detector.points, detector.weights, nothing, FullCell()) : gs
    push!(terms, (similar_projector(op, gk), ws .* wd))
  end
  [t[1] for t in terms], [t[2] for t in terms]
end

"""
    SampleSum

The operator of [`sampled`](@ref) when it sums several projectors: `y = sum_k w_k .* (A_k x)` with
a weight per view for every projector `A_k`, and the exact adjoint. Holds a data and an image
buffer, so it must not be applied by several tasks at once.
"""
struct SampleSum{T,O<:AbstractCTOperator{T},V,D,I} <: AbstractCTOperator{T}
  ops::Vector{O}
  weights::Vector{V}   # per projector, one weight per view, on the device
  data::D
  image::I
end

"""
    SampleStack

The operator of [`stacked`](@ref): the projections `A_k x` of the samples side by side, data of size
`(datasize(op)..., K)`, and the exact adjoint `sum_k A_kᵀ y[…, k]`. The weights of the samples are
[`sample_weights`](@ref); the product does not apply them. Holds buffers, so it must not be applied
by several tasks at once.
"""
struct SampleStack{T,O<:AbstractCTOperator{T},V,D,I} <: AbstractCTOperator{T}
  ops::Vector{O}
  weights::Vector{V}
  data::D
  image::I
end

# the operator of `ops` with the per-view weights `weights` (host vectors), with buffers on the
# device of the projectors
function sample_operator(::Type{S}, ops, weights) where {S}
  op = first(ops)
  T, backend = eltype(op), KA.get_backend(op)
  w = [copyto!(KA.allocate(backend, T, length(v)), T.(v)) for v in weights]
  S(collect(ops), w, KA.allocate(backend, T, datasize(op)), KA.allocate(backend, T, imagesize(op)))
end

const SampledOperator = Union{SampleSum,SampleStack}

"""
    sampled(op::AbstractProjector; source=nothing, detector=nothing, execution=:auto)

The projector `op` averaged over the samples of a focal spot (`source`, a [`FocalSpot`](@ref)) and of
the detector cells (`detector`, a [`DetectorSampling`](@ref)): the weighted mean of the line
integrals along the rays from every source sample to every detector sample, `sum_k w_k A_k x`, with
the data size of `op` and the exact adjoint. `op` keeps its element type, device, launch, image mask
and ray mask.

* `execution`: `:auto` (the default) samples in the projector's kernel where it can (detector
  sampling of Siddon and Joseph, see [`CTOperators.fuses_sampling`](@ref)) and sequentially
  otherwise, one projector per sample on a shifted geometry; `:fused` requires the kernel,
  `:sequential` forces one projector per sample.
* Sequential detector sampling needs a point strategy whose offsets are the same in every cell
  ([`CTOperators.varies_per_cell`](@ref)).

Returns `op` itself, a projector on the sampled geometry, or a [`SampleSum`](@ref).

# Example
```julia
op = JosephProjector(image, geometry)
spot = FocalSpot(0.4; points=DiskPoints(2, 6), weights=GaussianWeights(0.25))
A = sampled(op; source=spot, detector=DetectorSampling(RegularPoints(2)))
y = A * x
```
"""
function sampled(op::AbstractProjector; source=nothing, detector=nothing, execution::Symbol=:auto)
  ops, weights = sample_terms(op, source, detector, execution, Summed())
  length(ops) == 1 && all(isone, only(weights)) && return only(ops)
  sample_operator(SampleSum, ops, weights)
end

"""
    stacked(op::AbstractProjector; source=nothing, detector=nothing, execution=:auto)

The projections of `op` for every sample of `source` and `detector` (see [`sampled`](@ref)) side by
side, a [`SampleStack`](@ref) with data of size `(datasize(op)..., K)`: `K` the number of source
samples times that of the detector samples, or that of the source samples alone for a projector
that samples the detector in its kernel. Their weights are [`sample_weights`](@ref).
"""
function stacked(op::AbstractProjector; source=nothing, detector=nothing, execution::Symbol=:auto)
  ops, weights = sample_terms(op, source, detector, execution, Stacked())
  sample_operator(SampleStack, ops, weights)
end

"""
    sample_weights(B::SampleStack)

The weights of the samples of `B`, one vector per sample with a weight per view; over the samples
they sum to 1 in every view.
"""
sample_weights(B::SampleStack) = B.weights

datasize(S::SampleSum) = datasize(first(S.ops))
datasize(S::SampleStack) = (datasize(first(S.ops))..., length(S.ops))
imagesize(S::SampledOperator) = imagesize(first(S.ops))
KA.get_backend(S::SampledOperator) = KA.get_backend(first(S.ops))

# the per-view weights as an array that broadcasts over the data of one projector
view_weights(w) = reshape(w, 1, 1, :)

function forward!(y, S::SampleSum, x)
  fill!(y, zero(eltype(y)))
  for (op, w) in zip(S.ops, S.weights)
    apply!(S.data, op, x)
    y .+= view_weights(w) .* S.data
  end
  y
end

function backward!(x, S::SampleSum, y)
  fill!(x, zero(eltype(x)))
  for (op, w) in zip(S.ops, S.weights)
    S.data .= view_weights(w) .* y
    apply!(S.image, adjoint(op), S.data)
    x .+= S.image
  end
  x
end

function forward!(y, B::SampleStack, x)
  for (k, op) in enumerate(B.ops)
    apply!(B.data, op, x)
    selectdim(y, ndims(y), k) .= B.data
  end
  y
end

function backward!(x, B::SampleStack, y)
  fill!(x, zero(eltype(x)))
  for (k, op) in enumerate(B.ops)
    B.data .= selectdim(y, ndims(y), k)
    apply!(B.image, adjoint(op), B.data)
    x .+= B.image
  end
  x
end

subset(S::SampleSum, views) = sample_operator(SampleSum, [subset(op, views) for op in S.ops], [Array(w)[views] for w in S.weights])
subset(B::SampleStack, views) =
  sample_operator(SampleStack, [subset(op, views) for op in B.ops], [Array(w)[views] for w in B.weights])

Adapt.adapt_structure(to, S::SampleSum) = sample_operator(SampleSum, [adapt(to, op) for op in S.ops], [Array(w) for w in S.weights])
Adapt.adapt_structure(to, B::SampleStack) =
  sample_operator(SampleStack, [adapt(to, op) for op in B.ops], [Array(w) for w in B.weights])

function Base.show(io::IO, ::MIME"text/plain", S::SampledOperator)
  print(io, nameof(typeof(S)), " of ", length(S.ops), " ", nameof(typeof(first(S.ops))), "s, data ", datasize(S),
        ", image ", imagesize(S))
end
