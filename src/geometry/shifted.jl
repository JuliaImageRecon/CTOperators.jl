# A detector geometry with the source and the detector positions shifted per view: one sample of a
# focal spot (source offsets, lengths) or of the detector cells (offsets in cell units), as
# `sampled` builds them. The detector itself stays where it is: a source offset moves only the start
# of the rays. A flat detector and a parallel beam have no tie between source and detector, so
# `shift` returns a plain geometry for them; a detector on an arc is a cylinder about the source,
# which a shifted source must not move, hence the wrapper.

"""
    Shifted(geometry, source, detector)

`geometry` (an [`ArcConeBeam`](@ref)) with the rays starting at `source[:, k]` added to its source
in view `k` (a `3 × nviews` matrix of lengths, or `nothing`) and ending at the detector position
shifted by `detector[:, k]` cells along `u` and `v` (a `2 × nviews` matrix, or `nothing`). The
detector does not move with the source. Built by [`sampled`](@ref) for the samples of a focal spot
or of the detector cells; for `ConeBeam` and `ParallelBeam` it builds plain geometries instead.
"""
struct Shifted{T,G<:AbstractDetectorGeometry{T},S,D} <: AbstractDetectorGeometry{T}
  geometry::G
  source::S     # nothing, or 3 x nviews: offsets of the source
  detector::D   # nothing, or 2 x nviews: offsets on the detector, in cells
end

nviews(g::Shifted) = nviews(g.geometry)
detectorsize(g::Shifted) = detectorsize(g.geometry)
raytype(g::Shifted) = raytype(g.geometry)
storage_array(g::Shifted) = storage_array(g.geometry)

columns_of(::Nothing, views) = nothing
columns_of(m, views) = m[:, views]
subset(g::Shifted, views) = Shifted(subset(g.geometry, views), columns_of(g.source, views), columns_of(g.detector, views))

Adapt.adapt_structure(to, g::Shifted) = Shifted(adapt(to, g.geometry), adapt(to, g.source), adapt(to, g.detector))

function Base.show(io::IO, ::MIME"text/plain", g::Shifted)
  print(io, "Shifted (", isnothing(g.source) ? "" : "source", isnothing(g.source) || isnothing(g.detector) ? "" : ", ",
        isnothing(g.detector) ? "" : "detector", ") ")
  show(io, MIME("text/plain"), g.geometry)
end

@inline source_offset(::Nothing, k, ::Type{T}) where {T} = zero(SVector{3,T})
@inline source_offset(m, k, ::Type{T}) where {T} = column(m, k)
@inline detector_offset(::Nothing, k, ::Type{T}) where {T} = (zero(T), zero(T))
@inline detector_offset(m, k, ::Type{T}) where {T} = @inbounds (m[1, k], m[2, k])

@inline function cell_ray(g::Shifted{T}, cu, cv, k) where {T}
  eu, ev = detector_offset(g.detector, k, T)
  a, b = cell_ray(g.geometry, cu + eu, cv + ev, k)
  a + source_offset(g.source, k, T), b
end

@inline function ray(g::Shifted{T}, i) where {T}
  nu, nv = detectorsize(g)
  iu, iv, k = pixel_index(i, nu, nv)
  cell_ray(g, T(iu), T(iv), k)
end

# Where the ray from the shifted source through p meets the cylinder of the detector (the axis
# through the unshifted source s along v, radius r): `s' + t d` with `d = p - s'` solves
# |a + t d|^2 = r^2 in the plane perpendicular to the axis (a = s' - s), the root with t > 0. The
# cell coordinates come from the hit as for the unshifted arc, minus the detector offset. The
# density (R/r)^2 / (a cos phi) has cos phi = |d . n| / |d| with n the radial normal at the hit.
@inline function project_point(g::Shifted{T,<:ArcConeBeam}, p, k) where {T}
  h = g.geometry
  s = column(h.source, k)
  w = column(h.center, k) - s
  u = column(h.u, k)
  v = column(h.v, k)
  du, dv = norm(u), norm(v)
  ev = v / dv
  hv = dot(w, ev)
  radial = w - hv * ev
  r = norm(radial)
  a = source_offset(g.source, k, T)
  d = p - (s + a)
  aperp = a - dot(a, ev) * ev
  dperp = d - dot(d, ev) * ev
  qa, qb, qc = dot(dperp, dperp), 2 * dot(aperp, dperp), dot(aperp, aperp) - r * r
  disc = qb * qb - 4 * qa * qc
  t = (-qb + sqrt(max(disc, zero(T)))) / (2 * qa)
  q = a + t * d                                # the hit relative to s
  qperp = q - dot(q, ev) * ev
  x, y = dot(q, radial) / r, dot(q, u) / du
  eu, evo = detector_offset(g.detector, k, T)
  n = qperp / norm(qperp)
  DetectorHit(atan(y, x) * r / du + T(h.nu + 1) / 2 - eu, (dot(q, ev) - hv) / dv + T(h.nv + 1) / 2 - evo,
              t * t * norm(d) / (abs(dot(d, n)) * du * dv), t, dv, disc >= 0 && t > 0 && isfinite(t) && x > 0)
end

"""
    shift(geometry; source=nothing, detector=nothing)

`geometry` with its source shifted by `source[:, k]` (lengths) and its detector positions by
`detector[:, k]` cells along `u` and `v` in view `k`, both matrices on the device of the geometry,
or `nothing`. A `ConeBeam` and a `ParallelBeam` become plain geometries of their type (the center
of the detector moves by the detector offset); an `ArcConeBeam` becomes a [`Shifted`](@ref)
geometry, since its detector must not move with the source.
"""
function shift end

shift(g::AbstractDetectorGeometry; source=nothing, detector=nothing) =
  isnothing(source) && isnothing(detector) ? g : Shifted(g, source, detector)

moved_center(c, u, v, ::Nothing) = c
moved_center(c, u, v, e) = c .+ e[1:1, :] .* u .+ e[2:2, :] .* v

function shift(g::ConeBeam; source=nothing, detector=nothing)
  isnothing(source) && isnothing(detector) && return g
  ConeBeam(isnothing(source) ? g.source : g.source .+ source, moved_center(g.center, g.u, g.v, detector), g.u, g.v;
           nu=g.nu, nv=g.nv)
end

function shift(g::ParallelBeam; source=nothing, detector=nothing)
  isnothing(source) || throw(ArgumentError("a ParallelBeam has no source to shift"))
  isnothing(detector) && return g
  ParallelBeam(g.direction, moved_center(g.center, g.u, g.v, detector), g.u, g.v; nu=g.nu, nv=g.nv)
end
