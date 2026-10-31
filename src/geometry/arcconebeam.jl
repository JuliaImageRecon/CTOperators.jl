"""
    ArcConeBeam(source, centre, u, v; nu, nv)

Cone beam with a curved detector of `nu × nv` cells on a cylinder about the source, as in clinical
CT scanners (third-generation geometry), and an arbitrary trajectory, given per view `k` (columns
of `3 × nviews` matrices):

* `source[:, k]`: position of the source, on the axis of the detector cylinder (the axis runs
  along `v`)
* `centre[:, k]`: centre of the detector; its distance from the axis is the radius of the
  cylinder
* `u[:, k]`: the direction of the detector rows at the centre (tangent to the arc); its length is
  the cell size along the arc
* `v[:, k]`: vector from one detector row to the next along the cylinder axis; its length is the
  cell size

Cell `(iu, iv)` sits at the fan angle `γ = (iu − (nu + 1)/2) |u| / r` from the centre, with `r` the
radius, and at the height `(iv − (nv + 1)/2) v` from it; ray `(iu, iv, k)` runs from the source to
it. The projection data has size `(nu, nv, nviews)`. `u` must be orthogonal to `v` and to
`centre − source`.

[`circular_cone_beam`](@ref) and [`helical_cone_beam`](@ref) with `detector=:arc` construct the
common trajectories.
"""
struct ArcConeBeam{T,M<:AbstractMatrix{T}} <: AbstractDetectorGeometry{T}
  source::M
  centre::M
  u::M
  v::M
  nu::Int
  nv::Int
  function ArcConeBeam{T,M}(source::M, centre::M, u::M, v::M, nu::Int, nv::Int) where {T,M<:AbstractMatrix{T}}
    check_views("ArcConeBeam", source, centre, u, v)
    check_detector(nu, nv)
    new{T,M}(source, centre, u, v, nu, nv)
  end
end

function ArcConeBeam(source::AbstractMatrix, centre::AbstractMatrix, u::AbstractMatrix, v::AbstractMatrix;
                     nu::Integer, nv::Integer)
  ms = promote(source, centre, u, v)
  check_arc_frames(ms...)
  ArcConeBeam{eltype(ms[1]),typeof(ms[1])}(ms..., Int(nu), Int(nv))
end

# The arc model needs the row tangent orthogonal to the central ray and to the column direction
# (checked on the host, where the geometry is built; device arrays are not checked).
function check_arc_frames(source, centre, u, v)
  source isa Array || return nothing
  for k in axes(source, 2)
    w = centre[:, k] - source[:, k]
    tol = 1e-6 * norm(w) * max(norm(u[:, k]), norm(v[:, k]))
    (abs(dot(w, u[:, k])) <= tol && abs(dot(u[:, k], v[:, k])) <= 1e-6 * norm(u[:, k]) * norm(v[:, k])) ||
      throw(ArgumentError("ArcConeBeam view $k: u must be orthogonal to v and to centre - source"))
  end
end

nviews(g::ArcConeBeam) = size(g.source, 2)
detectorsize(g::ArcConeBeam) = (g.nu, g.nv)
raytype(::ArcConeBeam) = RaySegment()
subset(g::ArcConeBeam, views) =
  ArcConeBeam(g.source[:, views], g.centre[:, views], g.u[:, views], g.v[:, views]; nu=g.nu, nv=g.nv)
storage_array(g::ArcConeBeam) = g.source

@inline function ray(g::ArcConeBeam, i)
  iu, iv, k = pixel_index(i, g.nu, g.nv)
  s = column(g.source, k)
  w = column(g.centre, k) - s
  u = column(g.u, k)
  v = column(g.v, k)
  T = eltype(s)
  # the central ray split into its part along the cylinder axis (h) and the radius (w - h)
  h = (dot(w, v) / dot(v, v)) * v
  radial = w - h
  r = norm(radial)
  du = norm(u)
  sg, cg = sincos((T(iu) - T(g.nu + 1) / 2) * du / r)
  b = s + h + cg * radial + (sg * r / du) * u + (T(iv) - T(g.nv + 1) / 2) * v
  s, b
end

Adapt.adapt_structure(to, g::ArcConeBeam) =
  ArcConeBeam(adapt(to, g.source), adapt(to, g.centre), adapt(to, g.u), adapt(to, g.v); nu=g.nu, nv=g.nv)
