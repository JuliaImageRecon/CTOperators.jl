# Constructors of the common trajectories. Angles in radians. Conventions of Fessler's codes
# (Sinograms.jl, Long, Fessler & Balter 2010): the source sits at angle β counter-clockwise from
# the y axis, the detector opposite; at β = 0 the detector rows (u) run along +x, the columns (v)
# along +z. Detector offsets are in cells, with the sign of ImageGeoms' offsets: the cell
# positions along u are ((0:nu-1) .- (nu - 1)/2 .- offset_u) .* du.

"""
    circular_cone_beam(; angles, dso, dsd, nu, nv, du, dv=du, offset_u=0, offset_v=0, source_z=0,
                       detector=:flat)

Cone-beam geometry of a source on a circle of radius `dso` about the `z` axis and a detector at
distance `dsd` from the source, `nu × nv` cells of size `du × dv`, one view per angle in `angles`
(radians, source counter-clockwise from the `y` axis).

`detector=:flat` gives a flat detector ([`ConeBeam`](@ref)), `detector=:arc` a detector curved on
a cylinder about the source ([`ArcConeBeam`](@ref)), with `du` the cell size along the arc at
distance `dsd` (an angle of `du / dsd` per cell) and `offset_u` shifting the cells along the arc.

`source_z` is the height of source and detector centre, a number or one value per view; with
`source_z` varying the trajectory is no longer circular (see [`helical_cone_beam`](@ref)).
"""
function circular_cone_beam(; angles::AbstractVector{<:Real}, dso::Real, dsd::Real, nu::Integer, nv::Integer,
                            du::Real, dv::Real=du, offset_u::Real=0, offset_v::Real=0,
                            source_z::Union{Real,AbstractVector{<:Real}}=0, detector::Symbol=:flat)
  detector in (:flat, :arc) || throw(ArgumentError("detector must be :flat or :arc, got :$detector"))
  0 < dso < dsd || throw(ArgumentError("need 0 < dso < dsd, got dso = $dso, dsd = $dsd"))
  source_z isa AbstractVector && length(source_z) != length(angles) &&
    throw(DimensionMismatch("$(length(source_z)) source heights for $(length(angles)) angles"))
  n = length(angles)
  source, centre, u, v = (zeros(3, n) for _ in 1:4)
  dod = dsd - dso
  for (k, beta) in enumerate(angles)
    s, c = sincos(beta)
    z = source_z isa Real ? source_z : source_z[k]
    eu = SVector(c, s, 0.0)
    ev = SVector(0.0, 0.0, 1.0)
    src = SVector(-dso * s, dso * c, z)
    source[:, k] = src
    if detector === :flat
      centre[:, k] = SVector(dod * s, -dod * c, z) - offset_u * du * eu - offset_v * dv * ev
      u[:, k] = du * eu
    else
      # the centre turned on the arc by the offset, about the source; the row direction with it
      ew = SVector(s, -c, 0.0)          # from the source towards the rotation axis
      sg, cg = sincos(-offset_u * du / dsd)
      centre[:, k] = src + dsd * (cg * ew + sg * eu) - offset_v * dv * ev
      u[:, k] = du * (cg * eu - sg * ew)
    end
    v[:, k] = dv * ev
  end
  detector === :flat ? ConeBeam(source, centre, u, v; nu, nv) : ArcConeBeam(source, centre, u, v; nu, nv)
end

"""
    helical_cone_beam(; angles, pitch, z0=0, kwargs...)

[`ConeBeam`](@ref) geometry of a helical trajectory: as [`circular_cone_beam`](@ref) (same
keywords), with source and detector moving along `z` by `pitch` (a length) per turn,
`z = z0 + pitch (β − β₁)/2π` for view angle `β` (`β₁ = angles[1]`).

An object that translates along the rotation axis with speed `s` during a circular scan sees this
geometry with `pitch = −s · (time per turn)`.
"""
function helical_cone_beam(; angles::AbstractVector{<:Real}, pitch::Real, z0::Real=0, kwargs...)
  haskey(kwargs, :source_z) && throw(ArgumentError("helical_cone_beam sets source_z from pitch and z0"))
  beta1 = isempty(angles) ? 0.0 : float(first(angles))
  circular_cone_beam(; angles, source_z=[z0 + pitch * (beta - beta1) / 2pi for beta in angles], kwargs...)
end

"""
    parallel_beam(; angles, nu, nv, du, dv=du, offset_u=0, offset_v=0)

[`ParallelBeam`](@ref) geometry rotating about the `z` axis, `nu × nv` cells of size `du × dv`,
one view per angle in `angles` (radians). At angle `ϕ` the rays run along
`(−sin ϕ, cos ϕ, 0)` and the detector rows along `(cos ϕ, sin ϕ, 0)`: the coordinates `(u, v, ϕ)`
of ImagePhantoms.jl's `radon` with polar angle 0.
"""
function parallel_beam(; angles::AbstractVector{<:Real}, nu::Integer, nv::Integer, du::Real, dv::Real=du,
                       offset_u::Real=0, offset_v::Real=0)
  n = length(angles)
  direction, centre, u, v = (zeros(3, n) for _ in 1:4)
  for (k, phi) in enumerate(angles)
    s, c = sincos(phi)
    eu = SVector(c, s, 0.0)
    ev = SVector(0.0, 0.0, 1.0)
    direction[:, k] = SVector(-s, c, 0.0)
    centre[:, k] = -offset_u * du * eu - offset_v * dv * ev
    u[:, k] = du * eu
    v[:, k] = dv * ev
  end
  ParallelBeam(direction, centre, u, v; nu, nv)
end
