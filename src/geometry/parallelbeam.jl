"""
    ParallelBeam(direction, center, u, v; nu, nv)

Parallel beam with a flat detector of `nu × nv` cells, given per view `k` (columns of
`3 × nviews` matrices):

* `direction[:, k]`: direction of the rays (any nonzero length)
* `center[:, k]`: the center of the detector; since the rays are whole lines, its position along
  `direction` does not matter
* `u[:, k]`, `v[:, k]`: vectors from one detector cell to the next along the rows (`u`) and the
  columns (`v`); their lengths are the cell sizes

Ray `(iu, iv, k)` is the whole line through the center of cell `(iu, iv)` (see [`ConeBeam`](@ref))
along `direction[:, k]` ([`RayLine`](@ref)). The projection data has size `(nu, nv, nviews)`.

[`parallel_beam`](@ref) constructs the common rotation about the `z` axis.
"""
struct ParallelBeam{T,M<:AbstractMatrix{T}} <: AbstractDetectorGeometry{T}
  direction::M
  center::M
  u::M
  v::M
  nu::Int
  nv::Int
  function ParallelBeam{T,M}(direction::M, center::M, u::M, v::M, nu::Int, nv::Int) where {T,M<:AbstractMatrix{T}}
    check_views("ParallelBeam", direction, center, u, v)
    check_detector(nu, nv)
    new{T,M}(direction, center, u, v, nu, nv)
  end
end

function ParallelBeam(direction::AbstractMatrix, center::AbstractMatrix, u::AbstractMatrix, v::AbstractMatrix;
                      nu::Integer, nv::Integer)
  ms = promote(direction, center, u, v)
  ParallelBeam{eltype(ms[1]),typeof(ms[1])}(ms..., Int(nu), Int(nv))
end

nviews(g::ParallelBeam) = size(g.direction, 2)
detectorsize(g::ParallelBeam) = (g.nu, g.nv)
raytype(::ParallelBeam) = RayLine()
subset(g::ParallelBeam, views) =
  ParallelBeam(g.direction[:, views], g.center[:, views], g.u[:, views], g.v[:, views]; nu=g.nu, nv=g.nv)
storage_array(g::ParallelBeam) = g.direction

@inline function ray(g::ParallelBeam, i)
  iu, iv, k = pixel_index(i, g.nu, g.nv)
  a = pixel_position(column(g.center, k), column(g.u, k), column(g.v, k), iu, iv, g.nu, g.nv)
  a, a + column(g.direction, k)
end

Adapt.adapt_structure(to, g::ParallelBeam) =
  ParallelBeam(adapt(to, g.direction), adapt(to, g.center), adapt(to, g.u), adapt(to, g.v); nu=g.nu, nv=g.nv)
