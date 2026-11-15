# The circular scan of a detector geometry, as FDK needs it. Computed on the host in Float64 from the
# geometry's vectors, with checks that the geometry is a circular scan; violations throw an
# ArgumentError rather than give a wrong image. Whether the views cover enough of the circle is up to
# the redundancy weighting (redundancy.jl).

"""
    CircularScan

The circular scan of a geometry as [`FDK`](@ref) sees it, the input of
[`redundancy_weights`](@ref CTOperators.redundancy_weights). Fields (lengths in the unit of the
geometry, angles in radians):

* `axis`: unit vector along the rotation axis (the direction of the detector columns)
* `angles`: the angle of each view about the axis, unwrapped: strictly increasing or decreasing,
  the first in `(-π, π]`. For cone beams the angle of the source, for parallel beams that of the
  ray direction.
* `dso`, `dsd`: per view the distance from the source to the axis and to the detector along the
  central ray (both 1 for parallel beams)
* `central`: `3 × nviews`, the unit vector of the central ray of each view, from the source
  toward the axis and perpendicular to it (for parallel beams the ray direction)
"""
struct CircularScan
  axis::SVector{3,Float64}
  angles::Vector{Float64}
  dso::Vector{Float64}
  dsd::Vector{Float64}
  central::Matrix{Float64}
end

unitvec(a) = a / norm(a)
columns(m) = [SVector{3,Float64}(m[1, k], m[2, k], m[3, k]) for k in axes(m, 2)]

# two unit vectors that complete `axis` to a right-handed orthonormal frame
function plane_frame(axis)
  e1 = unitvec(cross(axis, abs(axis[1]) < 0.9 ? SVector(1.0, 0.0, 0.0) : SVector(0.0, 1.0, 0.0)))
  e1, cross(axis, e1)
end

# Angles are checked by their sine (or cosine for perpendicular vectors) against rtol: a tolerance of
# about rtol radians for every check.
parallel(a, b, rtol) = norm(cross(unitvec(a), unitvec(b))) <= rtol
perpendicular(a, b, rtol) = abs(dot(unitvec(a), unitvec(b))) <= rtol

# the cell spacing along the rows (|u|) must be the same in all views: the filter is built for one
function check_spacing(us, rtol)
  du = norm(us[1])
  all(u -> abs(norm(u) - du) <= rtol * du, us) ||
    throw(ArgumentError("FDK: the detector cells must have the same width (|u|) in all views"))
end

# the common direction of the detector columns: the rotation axis
function rotation_axis(vs, rtol)
  axis = unitvec(vs[1])
  all(v -> dot(v, axis) > 0 && parallel(v, axis, rtol), vs) ||
    throw(ArgumentError("FDK: the detector columns (v) must have the same direction in all views, the rotation axis"))
  axis
end

# Center and radius of the circle through the points (x_k, y_k) by least squares (Kasa's fit):
# x^2 + y^2 = 2 a x + 2 b y + c with radius^2 = c + a^2 + b^2.
function fit_circle(xs, ys)
  m = hcat(2 .* xs, 2 .* ys, ones(length(xs)))
  a, b, c = m \ (xs .^ 2 .+ ys .^ 2)
  a, b, sqrt(c + a^2 + b^2)
end

function circular_scan(g::Union{ConeBeam,ArcConeBeam}; rtol=1e-4)
  h = adapt(Array{Float64}, g)
  n = nviews(h)
  n >= 3 || throw(ArgumentError("FDK needs at least 3 views, got $n"))
  sources, centers, us, vs = columns(h.source), columns(h.center), columns(h.u), columns(h.v)
  axis = rotation_axis(vs, rtol)
  e1, e2 = plane_frame(axis)
  xs, ys, zs = dot.(sources, Ref(e1)), dot.(sources, Ref(e2)), dot.(sources, Ref(axis))
  a, b, radius = fit_circle(xs, ys)
  dso = hypot.(xs .- a, ys .- b)
  maximum(abs.(dso .- radius)) <= rtol * radius && maximum(abs.(zs .- zs[1])) <= rtol * radius ||
    throw(ArgumentError("FDK: the sources must lie on a circle about the rotation axis, in a plane perpendicular " *
                        "to it (helical and other trajectories are not supported)"))
  central = [unitvec((a - xs[k]) * e1 + (b - ys[k]) * e2) for k in 1:n]
  check_spacing(us, rtol)
  dsd = similar(dso)
  for k in 1:n
    w = central[k]
    perpendicular(us[k], axis, rtol) ||
      throw(ArgumentError("FDK: the detector rows (u) must be perpendicular to the rotation axis (view $k)"))
    if g isa ConeBeam
      normal = unitvec(cross(us[k], vs[k]))
      parallel(normal, w, rtol) ||
        throw(ArgumentError("FDK: the flat detector must be perpendicular to the central ray, the ray from the " *
                            "source to the rotation axis (view $k)"))
      dsd[k] = abs(dot(centers[k] - sources[k], normal))
    else
      # the radius of the detector cylinder (see project_point of ArcConeBeam)
      c = centers[k] - sources[k]
      dsd[k] = norm(c - dot(c, axis) * axis)
    end
  end
  g isa ArcConeBeam && maximum(abs.(dsd .- dsd[1])) > rtol * dsd[1] &&
    throw(ArgumentError("FDK: the detector arc must have the same radius in all views"))
  CircularScan(axis, ordered_angles(atan.(ys .- b, xs .- a)), dso, dsd, reduce(hcat, central))
end

function circular_scan(g::ParallelBeam; rtol=1e-4)
  h = adapt(Array{Float64}, g)
  n = nviews(h)
  n >= 2 || throw(ArgumentError("FDK of a parallel beam needs at least 2 views, got $n"))
  directions, us, vs = unitvec.(columns(h.direction)), columns(h.u), columns(h.v)
  axis = rotation_axis(vs, rtol)
  e1, e2 = plane_frame(axis)
  check_spacing(us, rtol)
  for k in 1:n
    perpendicular(directions[k], axis, rtol) && perpendicular(us[k], directions[k], rtol) &&
      perpendicular(us[k], axis, rtol) ||
      throw(ArgumentError("FDK: the rays must be perpendicular to the rotation axis (v), and the detector rows " *
                          "(u) perpendicular to the rays and the axis (view $k)"))
  end
  angles = atan.(dot.(directions, Ref(e2)), dot.(directions, Ref(e1)))
  CircularScan(axis, ordered_angles(angles), ones(n), ones(n), reduce(hcat, directions))
end

# differences of consecutive angles, wrapped to (-pi, pi]
wrapped_differences(angles) = [rem2pi(angles[k+1] - angles[k], RoundNearest) for k in 1:length(angles)-1]

# the angles unwrapped, checked to run in one direction
function ordered_angles(angles)
  d = wrapped_differences(angles)
  (all(>(0), d) || all(<(0), d)) ||
    throw(ArgumentError("FDK: the views must be ordered along the orbit (angles strictly increasing or decreasing)"))
  cumsum([angles[1]; d])
end

# the angle between the first and the last view
angular_span(scan::CircularScan) = abs(scan.angles[end] - scan.angles[1])
