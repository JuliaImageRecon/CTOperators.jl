# Figures of the page "The Parameters of a Scan" (CairoMakie): a scan built up step by step, from one
# ray to a helical cone-beam scan. Drawn from geometries of the package, so they follow its
# conventions. Included invisibly by the page.

using CTOperators, CairoMakie
using LinearAlgebra: normalize, dot

const STEP_COLORS = (source=:darkorange, ray=(:darkorange, 0.6), cells=(:steelblue, :lightsteelblue),
                     image=(:gray60, 0.25), frame=:gray50)

# a dimension line from a to b, shifted by `shift`, with ticks at its ends and `label` in the middle
# (rotated along the line, never upside down), on the side of the shift
function dimension!(ax, a, b, shift, label; labelshift=0.2, fontsize=15)
  p, q = Point2f(a + shift), Point2f(b + shift)
  d = normalize(q - p)
  n = Point2f(-d[2], d[1])
  lines!(ax, [p, q]; color=:black, linewidth=1)
  for e in (p, q)
    lines!(ax, [e - 0.08n, e + 0.08n]; color=:black, linewidth=1)
  end
  angle = atan(d[2], d[1])
  angle = angle > pi / 2 ? angle - pi : angle <= -pi / 2 ? angle + pi : angle
  side = dot(n, shift) >= 0 ? 1 : -1
  text!(ax, ((p + q) / 2 + side * labelshift * n)...; text=label, align=(:center, :center), rotation=angle, fontsize)
end

# the cells of a detector row between the edge points `edges`, alternately colored
function cells!(ax, edges)
  for k in 1:length(edges)-1
    lines!(ax, [Point2f(edges[k]), Point2f(edges[k+1])]; color=STEP_COLORS.cells[isodd(k) ? 1 : 2], linewidth=9)
  end
end

function step_axis(fig, pos; title="", limits=(-3.4, 3.4, -3.2, 3.6))
  ax = Axis(fig[pos...]; aspect=DataAspect(), title, limits)
  hidedecorations!(ax); hidespines!(ax)
  ax
end

image_square!(ax; h=0.9) = poly!(ax, Rect2f(-h, -h, 2h, 2h); color=STEP_COLORS.image, strokecolor=STEP_COLORS.frame, strokewidth=1)

# view 1 of a fan of nu cells (the central detector row of a cone beam) from above: source, the edges
# of the cells and the center of the detector, in x, y
function topview_geometry(; dso=3.0, dsd=5.0, nu=6, du=0.55, beta=0.0)
  g = circular_cone_beam(; angles=[beta], dso, dsd, nu, nv=1, du)
  s, c, u = g.source[1:2, 1], g.center[1:2, 1], g.u[1:2, 1]
  s, c, u, [c + (k - nu / 2) * u for k in 0:nu]
end

# 1. one ray from the source to one detector cell through the image
function step_ray_figure()
  fig = Figure(size=(520, 430))
  ax = step_axis(fig, (1, 1); title="one ray")
  s, c, u, edges = topview_geometry(; nu=1, du=0.55)
  image_square!(ax)
  cells!(ax, edges)
  lines!(ax, [Point2f(s), Point2f(c)]; color=STEP_COLORS.ray, linewidth=2)
  scatter!(ax, [Point2f(s)]; color=STEP_COLORS.source, markersize=16)
  text!(ax, s...; text="source", align=(:left, :center), offset=(12, 0), fontsize=15)
  text!(ax, edges[end]...; text="detector cell", align=(:left, :center), offset=(10, 0), fontsize=15)
  text!(ax, 0, 0; text="image", align=(:right, :bottom), offset=(-6, 4), fontsize=14, color=:gray30)
  text!(ax, 0.12, 1.6; text="ray: the cell measures the\nline integral of the image\nalong it", align=(:left, :center), fontsize=13)
  fig
end

# 2. a detector row: nu cells of size du along u, the rays to their centers, the distance dsd
function step_row_figure(; nu=6, du=0.55)
  fig = Figure(size=(520, 430))
  ax = step_axis(fig, (1, 1); title="a detector row")
  s, c, u, edges = topview_geometry(; nu, du)
  image_square!(ax)
  for k in 1:nu
    lines!(ax, [Point2f(s), Point2f((edges[k] + edges[k+1]) / 2)]; color=STEP_COLORS.ray, linewidth=1)
  end
  cells!(ax, edges)
  scatter!(ax, [Point2f(s)]; color=STEP_COLORS.source, markersize=16)
  text!(ax, s...; text="source", align=(:left, :center), offset=(12, 0), fontsize=15)
  ray = normalize(c - s)
  dimension!(ax, edges[1], edges[2], -0.3 * ray, "du")
  arrows2d!(ax, [Point2f(edges[end] + 0.1 * normalize(u))], [Vec2f(0.7 * normalize(u))]; color=STEP_COLORS.cells[1])
  text!(ax, (edges[end] + 0.85 * normalize(u))...; text="u", align=(:left, :center), offset=(4, 0), fontsize=15)
  text!(ax, (c - 0.75 * ray)...; text="nu cells", align=(:center, :center), fontsize=15)
  dimension!(ax, s, c, [-2.3, 0.0], "dsd")
  fig
end

# 3. rotation: the source on a circle of radius dso about the rotation axis, at the angle β of the view
function step_rotation_figure(; dso=3.0, dsd=5.0, nu=6, du=0.55, beta=pi / 5)
  fig = Figure(size=(520, 470))
  ax = step_axis(fig, (1, 1); title="views: the scanner rotates", limits=(-3.6, 3.6, -3.4, 3.8))
  image_square!(ax)
  lines!(ax, [Point2f(dso * cos(t), dso * sin(t)) for t in range(0, 2pi, 200)]; color=(STEP_COLORS.source, 0.4), linestyle=:dash)
  # an earlier view, faint: the first one, at β = 0
  s0, _, _, edges0 = topview_geometry(; dso, dsd, nu, du, beta=0.0)
  lines!(ax, [Point2f(edges0[1]), Point2f(edges0[end])]; color=(STEP_COLORS.cells[2], 0.6), linewidth=6)
  scatter!(ax, [Point2f(s0)]; color=(STEP_COLORS.source, 0.4), markersize=12)
  text!(ax, s0...; text="first view", align=(:left, :bottom), offset=(8, 4), fontsize=12, color=:gray30)
  # the view at β
  s, c, u, edges = topview_geometry(; dso, dsd, nu, du, beta)
  for e in edges[[1, end]]
    lines!(ax, [Point2f(s), Point2f(e)]; color=STEP_COLORS.ray, linewidth=1)
  end
  cells!(ax, edges)
  scatter!(ax, [Point2f(s)]; color=STEP_COLORS.source, markersize=16)
  scatter!(ax, [Point2f(0, 0)]; color=:black, markersize=8)
  text!(ax, 0, 0; text="rotation axis", align=(:left, :top), offset=(6, -6), fontsize=13)
  dimension!(ax, s, [0.0, 0.0], -0.35 * normalize([s[2], -s[1]]), "dso")
  arc = [Point2f(-1.4 * sin(t), 1.4 * cos(t)) for t in range(0, beta, 30)]
  lines!(ax, [Point2f(0, 0), Point2f(0, 1.9)]; color=:gray40, linewidth=0.8, linestyle=:dot)
  lines!(ax, arc; color=:black)
  text!(ax, arc[end÷2]...; text="β", align=(:left, :bottom), offset=(-4, 6), fontsize=16)
  fig
end

# 4. cone beam in 3D, from the angle of the other 3D views (`scene_axis` of geometry_figures.jl): the
# detector has nu × nv cells of size du × dv, drawn as a checkerboard, along u and v
function step_cone_figure(; nu=8, nv=6, du=20.0, dv=20.0, dso=128.0, dsd=256.0, beta=0.0)
  g = circular_cone_beam(; angles=[beta], dso, dsd, nu, nv, du, dv)
  image = ImageGeom(; dims=(64, 64, 64), deltas=(1.0, 1.0, 1.0))
  s = Point3f(g.source[:, 1])
  c, u, v = g.center[:, 1], g.u[:, 1], g.v[:, 1]
  corner(a, b) = Point3f(c + (a - nu / 2) * u + (b - nv / 2) * v)   # a in 0:nu, b in 0:nv
  fig = Figure(size=(620, 480))
  ax = scene_axis(fig, (1, 1); title="cone beam: nu × nv detector cells")
  draw_volume!(ax, image)
  # the cells as a checkerboard
  for a in 0:nu-1, b in 0:nv-1
    quad = [corner(a, b), corner(a + 1, b), corner(a + 1, b + 1), corner(a, b + 1)]
    mesh!(ax, quad, [1 2 3; 1 3 4]; color=STEP_COLORS.cells[isodd(a + b) ? 1 : 2], shading=NoShading)
  end
  # the cone: the rays to the four corners of the detector, and a few rays to cells
  for (a, b) in ((0, 0), (nu, 0), (nu, nv), (0, nv))
    lines!(ax, [s, corner(a, b)]; color=:steelblue, linewidth=1)
  end
  for (a, b) in ((2, 2), (5, 3), (3, 4))
    lines!(ax, [s, corner(a + 0.5, b + 0.5)]; color=STEP_COLORS.ray, linewidth=1)
  end
  scatter!(ax, [s]; color=STEP_COLORS.source, markersize=14)
  text!(ax, s; text="source", align=(:right, :bottom), offset=(-6, 6))
  # u and v from the first cell, and the counts
  o = corner(0, 0)
  for (vec, name) in ((u, "u"), (v, "v"))
    tip = o + Point3f(60 * normalize(vec))
    lines!(ax, [o, tip]; color=:black, linewidth=2)
    text!(ax, tip; text=name, fontsize=16, align=(:left, :bottom))
  end
  text!(ax, corner(nu / 2, nv); text="nu cells", align=(:center, :bottom), offset=(0, 8), fontsize=14)
  text!(ax, corner(0, nv / 2); text="nv rows", align=(:right, :center), offset=(-8, 0), fontsize=14)
  # one cell outlined: its size du × dv
  cell = [corner(0, nv - 1), corner(1, nv - 1), corner(1, nv), corner(0, nv), corner(0, nv - 1)]
  lines!(ax, cell; color=STEP_COLORS.source, linewidth=2.5)
  text!(ax, corner(0, nv); text="du × dv", align=(:right, :bottom), offset=(-4, 4), fontsize=13)
  fig
end

# 5. helical scan: the height of the source over the angle, rising by pitch per turn from z0
function step_helix_figure(; pitch=24.0, z0=-32.0, turns=2.5)
  angles = range(0, 2pi * turns, 200)
  g = helical_cone_beam(; angles, pitch, z0, dso=128, dsd=256, nu=8, nv=4, du=1.0)
  fig = Figure(size=(560, 400))
  ax = Axis(fig[1, 1]; xlabel="angle β of the view (turns)", ylabel="height z of source and detector",
            title="helical scan")
  t = angles ./ 2pi
  lines!(ax, t, g.source[3, :]; color=STEP_COLORS.source, linewidth=2)
  hlines!(ax, [0]; color=:gray60, linestyle=:dot)
  scatter!(ax, [0], [z0]; color=STEP_COLORS.source, markersize=12)
  text!(ax, 0, z0; text="first view: z0", align=(:left, :top), offset=(8, -4), fontsize=14)
  # pitch: the rise in one turn
  lines!(ax, [1, 2], [z0 + pitch, z0 + pitch]; color=:gray40, linestyle=:dash)
  lines!(ax, [2, 2], [z0 + pitch, z0 + 2pitch]; color=:black)
  text!(ax, 2, z0 + 1.5pitch; text="pitch\n(one turn)", align=(:left, :center), offset=(8, 0), fontsize=14)
  ylims!(ax, z0 - 12, z0 + pitch * turns + 6)
  fig
end
