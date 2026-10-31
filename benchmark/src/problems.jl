# Benchmark problems, shared by all workers: image geometry, projection geometry, a phantom and the
# rays on which the projection error against the analytic line integrals is measured. Generated
# deterministically, so every worker process sees the same problem.

using CTOperators, LinearAlgebra, StaticArrays
using ImagePhantoms: ellipsoid, phantom, radon
using ImageGeoms: axis

struct Problem
  name::String
  image::ImageGeom
  geometry::AbstractProjectionGeometry
end

# n³ volume of unit voxels, offsets so that no ray lies in a plane between voxels
volume(n) = ImageGeom(; dims=(n, n, n), deltas=(1.0, 1.0, 1.0), offsets=(0.37, 0.11, -0.23))

# Cone beam with magnification 2, a detector covering the volume, views over a full turn.
cone(n, nviews) = circular_cone_beam(; angles=range(0, 2pi, nviews + 1)[1:nviews], dso=2n, dsd=4n,
                                     nu=n, nv=n, du=2.0, dv=2.0)
# Helical cone beam: three turns, the object moves by half the volume height per turn.
helix(n, nviews) = helical_cone_beam(; angles=range(0, 6pi, nviews), pitch=n / 2, z0=-3n / 4, dso=2n,
                                     dsd=4n, nu=n, nv=n ÷ 2, du=2.0, dv=2.0)
# Parallel beam over half a turn, one detector cell per voxel.
parallel(n, nviews) = parallel_beam(; angles=range(0, pi, nviews + 1)[1:nviews], nu=n, nv=n, du=1.0)

const PROBLEMS = Dict(
  "cone128" => () -> Problem("cone128", volume(128), cone(128, 180)),
  "cone256" => () -> Problem("cone256", volume(256), cone(256, 360)),
  "helix256" => () -> Problem("helix256", volume(256), helix(256, 540)),
  "parallel128" => () -> Problem("parallel128", volume(128), parallel(128, 180)),
  "parallel256" => () -> Problem("parallel256", volume(256), parallel(256, 360)),
)

include("suites.jl")

# Three ellipsoids inside the volume (radius n/2): body, an inner structure, a small dense insert.
function phantom_objects(n)
  r = n / 2
  [ellipsoid((0.0, 0.0, 0.0), (0.8r, 0.65r, 0.85r), (0.0, 0.0, 0.0), 1.0),
   ellipsoid((0.2r, -0.1r, 0.1r), (0.3r, 0.2r, 0.35r), (0.4, 0.0, 0.0), 0.5),
   ellipsoid((-0.3r, 0.25r, -0.2r), (0.08r, 0.1r, 0.12r), (0.0, 0.0, 0.0), 2.0)]
end

# the phantom on the image grid (voxel averages of 2³ samples), Float32
function phantom_image(p::Problem)
  objects = phantom_objects(p.image.dims[1])
  Float32.(phantom(axis(p.image, 1), axis(p.image, 2), axis(p.image, 3), objects, 2))
end

# Coordinates (u, v, phi, theta) of ImagePhantoms' 3D `radon` for the line through a and b.
function line_coordinates(a, b)
  d = normalize(b - a)
  theta = asin(clamp(d[3], -1, 1))
  phi = atan(-d[1], d[2])
  sphi, cphi = sincos(phi)
  stheta, ctheta = sincos(theta)
  dot(a, SVector(cphi, sphi, 0)), dot(a, SVector(sphi * stheta, -cphi * stheta, ctheta)), phi, theta
end

# Rays on which the error is measured (evenly spaced linear indices into the projection data) and
# the exact line integrals of the phantom along them.
function error_rays(p::Problem; n=20_000)
  idx = unique(round.(Int, range(1, nrays(p.geometry), length=min(n, nrays(p.geometry)))))
  objects = phantom_objects(p.image.dims[1])
  exact = [radon(objects)(line_coordinates(ray(p.geometry, i)...)...) for i in idx]
  idx, exact
end
