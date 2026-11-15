"""
    ConeBeam(source, center, u, v; nu, nv)

Cone beam with a flat detector of `nu × nv` cells and an arbitrary trajectory, given per view
`k` (columns of `3 × nviews` matrices):

* `source[:, k]`: position of the source
* `center[:, k]`: center of the detector
* `u[:, k]`, `v[:, k]`: vectors from one detector cell to the next along the rows (`u`) and the
  columns (`v`); their lengths are the cell sizes

The center of cell `(iu, iv)` is `center + (iu − (nu + 1)/2) u + (iv − (nv + 1)/2) v`, and ray
`(iu, iv, k)` runs from the source to this point. The projection data has size
`(nu, nv, nviews)`.

[`circular_cone_beam`](@ref) and [`helical_cone_beam`](@ref) construct the common trajectories.
"""
struct ConeBeam{T,M<:AbstractMatrix{T}} <: AbstractDetectorGeometry{T}
  source::M
  center::M
  u::M
  v::M
  nu::Int
  nv::Int
  function ConeBeam{T,M}(source::M, center::M, u::M, v::M, nu::Int, nv::Int) where {T,M<:AbstractMatrix{T}}
    check_views("ConeBeam", source, center, u, v)
    check_detector(nu, nv)
    new{T,M}(source, center, u, v, nu, nv)
  end
end

function ConeBeam(source::AbstractMatrix, center::AbstractMatrix, u::AbstractMatrix, v::AbstractMatrix;
                  nu::Integer, nv::Integer)
  ms = promote(source, center, u, v)
  ConeBeam{eltype(ms[1]),typeof(ms[1])}(ms..., Int(nu), Int(nv))
end

nviews(g::ConeBeam) = size(g.source, 2)
detectorsize(g::ConeBeam) = (g.nu, g.nv)
raytype(::ConeBeam) = RaySegment()
subset(g::ConeBeam, views) =
  ConeBeam(g.source[:, views], g.center[:, views], g.u[:, views], g.v[:, views]; nu=g.nu, nv=g.nv)
storage_array(g::ConeBeam) = g.source

@inline function ray(g::ConeBeam, i)
  iu, iv, k = pixel_index(i, g.nu, g.nv)
  T = eltype(g)
  cell_ray(g, T(iu), T(iv), k)
end

@inline cell_ray(g::ConeBeam, cu, cv, k) =
  column(g.source, k), cell_position(column(g.center, k), column(g.u, k), column(g.v, k), cu, cv, g.nu, g.nv)

@inline function project_point(g::ConeBeam, p, k)
  s, c, u, v = column(g.source, k), column(g.center, k), column(g.u, k), column(g.v, k)
  T = eltype(s)
  n = cross(u, v)
  d = p - s
  dn = dot(d, n)
  t = dot(c - s, n) / dn                      # R / r: the detector point is s + t d
  alpha, beta = plane_coordinates(s + t * d - c, u, v)
  # (R/r)^2 / (a cos phi) with a = |n| and cos phi = |dn| / (|d| |n|)
  DetectorHit(alpha + T(g.nu + 1) / 2, beta + T(g.nv + 1) / 2, t * t * norm(d) / abs(dn), t, norm(v),
              t > 0 && isfinite(t))
end

Adapt.adapt_structure(to, g::ConeBeam) =
  ConeBeam(adapt(to, g.source), adapt(to, g.center), adapt(to, g.u), adapt(to, g.v); nu=g.nu, nv=g.nv)
