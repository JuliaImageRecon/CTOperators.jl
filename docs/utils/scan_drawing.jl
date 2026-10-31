# Drawing helpers for scans (detector, cone, rays, volume, phantom), for any Makie backend: included
# by the figure files, which load the backend.

using LinearAlgebra: normalize

# centres of every `step`-th detector cell of view k, as coordinate matrices (for `surface!`)
function detector_grid(g::ConeBeam, k, step)
  c, u, v = g.centre[:, k], g.u[:, k], g.v[:, k]
  ius, ivs = 1:step:g.nu, 1:step:g.nv
  p(d, iu, iv) = c[d] + (iu - (g.nu + 1) / 2) * u[d] + (iv - (g.nv + 1) / 2) * v[d]
  ntuple(d -> Float32[p(d, iu, iv) for iu in ius, iv in ivs], 3)
end

# the four detector corners of view k
function detector_corners(g::ConeBeam, k)
  c, u, v = g.centre[:, k], g.u[:, k], g.v[:, k]
  [Point3f(c + su * g.nu / 2 * u + sv * g.nv / 2 * v) for (su, sv) in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
end

# the cone of view k: from the source to the detector corners, and the detector outline
function cone_segments(g::ConeBeam, k)
  corners = detector_corners(g, k)
  s = Point3f(g.source[:, k])
  vcat([s => c for c in corners], [corners[i] => corners[mod1(i + 1, 4)] for i in 1:4])
end

# a few rays of view k: from the source to an evenly spaced subset of detector cells
function ray_segments(g::ConeBeam, k; nu=7, nv=4)
  s = Point3f(g.source[:, k])
  c, u, v = g.centre[:, k], g.u[:, k], g.v[:, k]
  cells = [(iu, iv) for iu in round.(Int, range(1, g.nu, nu)), iv in round.(Int, range(1, g.nv, nv))]
  [s => Point3f(c + (iu - (g.nu + 1) / 2) * u + (iv - (g.nv + 1) / 2) * v) for (iu, iv) in vec(cells)]
end

# the volume as a box of n^3 unit voxels at the origin
function draw_box!(ax, n)
  lo, hi = -n / 2, n / 2
  corners = [Point3f(a, b, c) for a in (lo, hi), b in (lo, hi), c in (lo, hi)]
  edges = [corners[i] => corners[j] for i in eachindex(corners), j in eachindex(corners)
           if i < j && count(corners[i] .!= corners[j]) == 1]
  linesegments!(ax, edges; color=:gray40)
end

# The first ellipsoids of the Shepp-Logan phantom (head and inner structures) as transparent surfaces.
# Parameters as from ImagePhantoms.ellipsoid_parameters: (cx, cy, cz, rx, ry, rz, phi, theta, psi, value).
function draw_phantom!(ax, params; count=4)
  t, p = range(0, 2pi, 40), range(0, pi, 20)
  for (cx, cy, cz, rx, ry, rz, phi, _, _, _) in params[1:count]
    sp, cp = sincos(phi)
    x = [rx * cos(a) * sin(b) for a in t, b in p]
    y = [ry * sin(a) * sin(b) for a in t, b in p]
    z = [rz * cos(b) for a in t, b in p]
    surface!(ax, cx .+ cp .* x .- sp .* y, cy .+ sp .* x .+ cp .* y, cz .+ z;
             color=fill(0.0f0, size(x)), colormap=[:lightgray, :lightgray], transparency=true, alpha=0.25)
  end
end
