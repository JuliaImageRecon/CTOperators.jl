# Figures of the distance-driven explanation page (WGLMakie through Bonito for the interactive one,
# CairoMakie for the static ones). Included invisibly by the page. All geometry comes from the
# package: the rays through the cell edges from `cell_ray`, the weights from the back projection.

using CTOperators, WGLMakie, Bonito, CairoMakie
using CTOperators: cell_ray

include(joinpath(@__DIR__, "joseph_figures.jl"))   # coverage of Siddon and Joseph, sweep helpers

# The 2D fan beam of the figures: an n x n image of unit pixels, the source at distance dso from the
# center, a detector of nu cells of width du at distance dsd from the source, one view at angle theta.
fan(theta; nu=12, du=2.4, dso=10.0, dsd=20.0) =
  circular_cone_beam(; angles=[theta], dso, dsd, nu, nv=1, du)

# the in-plane point of a 3D point
p2(a) = Point2f(a[1], a[2])

# the axis along which the central ray of the view (through the detector center) advances most:
# the principal axis of the view (1 = x, 2 = y)
function view_axis(g)
  nu = detectorsize(g)[1]
  a, b = cell_ray(g, (nu + 1) / 2, 1.0, 1)
  abs(b[1] - a[1]) >= abs(b[2] - a[2]) ? 1 : 2
end

# where the ray a -> b crosses the line `axis = c`: the other coordinate
function crossing(a, b, axis, c)
  t = (c - a[axis]) / (b[axis] - a[axis])
  a[3-axis] + t * (b[3-axis] - a[3-axis])
end

# The footprint of cell iu on every layer of the n x n image: on the center line of the layer (the
# line `axis = c` through its pixel centers), the interval between the rays through the two cell
# edges, clipped to the image. Returns the segments as pairs of points.
function footprints(g, iu, n)
  axis = view_axis(g)
  (sl, el), (sr, er) = cell_ray(g, iu - 0.5, 1.0, 1), cell_ray(g, iu + 0.5, 1.0, 1)
  half = n / 2
  segments = Pair{Point2f,Point2f}[]
  for c in range(-(n - 1) / 2, (n - 1) / 2, n)
    lo, hi = minmax(crossing(sl, el, axis, c), crossing(sr, er, axis, c))
    lo, hi = max(lo, -half), min(hi, half)
    lo < hi || continue
    point(q) = axis == 1 ? Point2f(c, q) : Point2f(q, c)
    push!(segments, point(lo) => point(hi))
  end
  segments
end

# the weights of cell iu in the pixels: the back projection of the value 1 in that cell
function cell_weights(image, g, iu)
  op = DistanceDrivenProjector(image, g; eltype=Float64)
  y = zeros(datasize(op))
  y[iu, 1, 1] = 1
  reshape(adjoint(op) * y, image.dims)
end

# the detector cells as segments between their edges
detector_cells(g) = [p2(cell_ray(g, iu - 0.5, 1.0, 1)[2]) => p2(cell_ray(g, iu + 0.5, 1.0, 1)[2])
                     for iu in 1:detectorsize(g)[1]]

# Everything the fan figure draws for view angle theta and cell iu.
function fan_scene(theta, iu, n)
  g = fan(theta)
  image = ImageGeom(; dims=(n, n), deltas=(1.0, 1.0))
  s = p2(cell_ray(g, 1.0, 1.0, 1)[1])
  el, er = p2(cell_ray(g, iu - 0.5, 1.0, 1)[2]), p2(cell_ray(g, iu + 0.5, 1.0, 1)[2])
  axis = view_axis(g)
  centers = range(-(n - 1) / 2, (n - 1) / 2, n)
  layers = [axis == 1 ? (Point2f(c, -n / 2) => Point2f(c, n / 2)) : (Point2f(-n / 2, c) => Point2f(n / 2, c))
            for c in centers]
  (; weights=cell_weights(image, g, iu), source=s, beam=[s, el, er], cells=detector_cells(g),
   cell=[el => er], central=[s => p2(cell_ray(g, Float64(iu), 1.0, 1)[2])],
   footprints=footprints(g, iu, n), layers, axis=axis == 1 ? "x" : "y")
end

# Draws the fan scene (an Observable of `fan_scene`) into a new axis at `pos`.
function fan_axis!(pos, scene, n; lim=11.5)
  ax = Axis(pos; aspect=DataAspect(), limits=((-lim, lim), (-lim, lim)), xlabel="x", ylabel="y",
            title=map(s -> "principal axis of the view: $(s.axis)", scene))
  centers = range(-(n - 1) / 2, (n - 1) / 2, n)
  hm = heatmap!(ax, centers, centers, map(s -> s.weights, scene); colormap=:Blues,
                colorrange=(1e-6, 1.6), lowclip=:white)
  for w in -n/2:n/2
    lines!(ax, [w, w], [-n / 2, n / 2]; color=:gray70, linewidth=0.5)
    lines!(ax, [-n / 2, n / 2], [w, w]; color=:gray70, linewidth=0.5)
  end
  poly!(ax, map(s -> s.beam, scene); color=(:darkorange, 0.12), strokecolor=(:darkorange, 0.8), strokewidth=1,
        label="beam of the cell, between its edge rays")
  linesegments!(ax, map(s -> s.layers, scene); color=:gray45, linestyle=:dash, linewidth=0.8,
                label="center lines of the layers")
  linesegments!(ax, map(s -> s.central, scene); color=:black, linestyle=:dot, linewidth=1, label="ray of the cell")
  linesegments!(ax, map(s -> s.cells, scene); color=:steelblue, linewidth=4, label="detector cells")
  linesegments!(ax, map(s -> s.cell, scene); color=:darkorange, linewidth=6, label="the cell")
  linesegments!(ax, map(s -> s.footprints, scene); color=:sienna, linewidth=4, label="footprints of the cell")
  scatter!(ax, map(s -> s.source, scene); color=:black, markersize=10, label="source")
  ax, hm
end

const FAN_N = 8
const FAN_CELL = 7

# The interactive figure: the slider rotates the scan; one cell (orange) with its beam, the layers
# of the view (dashed), the cell's footprint on every layer (thick orange) and the weights.
function fan_app(; nsweep=48)
  App() do session
    k = Bonito.Slider(1:nsweep)
    scene = map(i -> fan_scene(pi * (i - 1) / nsweep, FAN_CELL, FAN_N), k.value)
    fig = Figure(size=(620, 640))
    ax, hm = fan_axis!(fig[1, 1], scene, FAN_N)
    Colorbar(fig[1, 2], hm; label="weight of the pixel")
    Legend(fig[2, 1:2], ax; orientation=:horizontal, nbanks=3, framevisible=false, labelsize=12)
    Bonito.record_states(session, DOM.div(DOM.div("view ", k), fig))
  end
end

# The fan scene as a static figure, for one angle.
function fan_figure(theta)
  fig = Figure(size=(620, 640))
  ax, hm = fan_axis!(fig[1, 1], Observable(fan_scene(theta, FAN_CELL, FAN_N)), FAN_N)
  Colorbar(fig[1, 2], hm; label="weight of the pixel")
  Legend(fig[2, 1:2], ax; orientation=:horizontal, nbanks=3, framevisible=false, labelsize=12)
  fig
end

# All cells of one view and their footprints on the layers (left, alternating colors), and one
# layer's center line drawn as an axis (right): the pixel edges above, the footprints of the cells
# below. The distances between these boundaries are the weights.
function tiling_figure(theta=0.35; n=FAN_N, layer=3)
  g = fan(theta)
  nu = detectorsize(g)[1]
  colors = [isodd(iu) ? :darkorange : :seagreen for iu in 1:nu]
  fig = Figure(size=(980, 480))
  ax = Axis(fig[1, 1]; aspect=DataAspect(), limits=((-11.5, 11.5), (-11.5, 11.5)), xlabel="x", ylabel="y",
            title="all cells of one view")
  for w in -n/2:n/2
    lines!(ax, [w, w], [-n / 2, n / 2]; color=:gray70, linewidth=0.5)
    lines!(ax, [-n / 2, n / 2], [w, w]; color=:gray70, linewidth=0.5)
  end
  s = p2(cell_ray(g, 1.0, 1.0, 1)[1])
  for iu in 1:nu
    el, er = p2(cell_ray(g, iu - 0.5, 1.0, 1)[2]), p2(cell_ray(g, iu + 0.5, 1.0, 1)[2])
    poly!(ax, [s, el, er]; color=(colors[iu], 0.10), strokecolor=(colors[iu], 0.5), strokewidth=0.5)
    linesegments!(ax, [el => er]; color=colors[iu], linewidth=5)
    linesegments!(ax, footprints(g, iu, n); color=colors[iu], linewidth=4)
  end
  scatter!(ax, [s]; color=:black, markersize=10)
  axis = view_axis(g)
  c = -(n - 1) / 2 + (layer - 1)
  lineseg = axis == 1 ? [Point2f(c, -n / 2) => Point2f(c, n / 2)] : [Point2f(-n / 2, c) => Point2f(n / 2, c)]
  linesegments!(ax, lineseg; color=:black, linewidth=1.5)
  # the center line of the chosen layer, as an axis
  other = axis == 1 ? "y" : "x"
  ax2 = Axis(fig[1, 2]; limits=((-n / 2 - 0.3, n / 2 + 0.3), (-1.6, 1.4)), xlabel="$other on the center line of the layer",
             title="one layer ($(axis == 1 ? "x" : "y") = $c, black line on the left)")
  hideydecorations!(ax2); hidespines!(ax2, :l, :r, :t)
  for j in 1:n
    lo = -n / 2 + (j - 1)
    poly!(ax2, Rect2f(lo, 0.25, 1, 0.8); color=(:steelblue, isodd(j) ? 0.15 : 0.3), strokecolor=:steelblue,
          strokewidth=1)
  end
  text!(ax2, 0, 1.15; text="pixels of the layer", align=(:center, :bottom), fontsize=13, color=:steelblue)
  for iu in 1:nu
    for seg in footprints(g, iu, n)
      p, q = axis == 1 ? (seg[1][2], seg[2][2]) : (seg[1][1], seg[2][1])
      lo, hi = minmax(p, q)
      abs(axis == 1 ? seg[1][1] - c : seg[1][2] - c) < 1e-6 || continue
      poly!(ax2, Rect2f(lo, -0.85, hi - lo, 0.8); color=(colors[iu], 0.5), strokecolor=colors[iu], strokewidth=1)
      text!(ax2, (lo + hi) / 2, -0.45; text="$iu", align=(:center, :center), fontsize=12)
    end
  end
  # the pixel edges, continued through the footprints: the overlaps are the weights
  for w in -n/2:n/2
    lines!(ax2, [w, w], [-0.9, 0.25]; color=:steelblue, linestyle=:dash, linewidth=0.8)
  end
  text!(ax2, 0, -1.0; text="footprints of the cells", align=(:center, :top), fontsize=13, color=:gray30)
  colsize!(fig.layout, 1, Relative(0.48))
  fig
end

# The cone beam of the 3D figure: an 8 x 8 x 6 image of unit voxels, one view, a flat detector.
cone() = circular_cone_beam(; angles=[0.35], dso=8.0, dsd=16.0, nu=8, nv=6, du=2.4)
const CONE_DIMS = (8, 8, 6)
const CONE_CELL = (5, 4)

# where the ray a -> b crosses the plane `axis = c`
function plane_point(a, b, axis, c)
  t = (c - a[axis]) / (b[axis] - a[axis])
  a + t * (b - a)
end

# The footprint of cell `CONE_CELL` in layer `layer` along the principal axis ka of the view: the
# crossings of the four edge rays with the center plane of the layer, and the rectangle they span
# (b: the transverse axis that u runs along, c = z).
function cone_footprint(g, layer)
  nx, ny, nz = CONE_DIMS
  iu, iv = CONE_CELL
  nu, nv = detectorsize(g)
  a0, b0 = cell_ray(g, (nu + 1) / 2, (nv + 1) / 2, 1)
  ka = abs(b0[1] - a0[1]) >= abs(b0[2] - a0[2]) ? 1 : 2
  kb = 3 - ka
  c = -(CONE_DIMS[ka] - 1) / 2 + (layer - 1)
  edges = [cell_ray(g, iu - 0.5, iv, 1), cell_ray(g, iu + 0.5, iv, 1), cell_ray(g, iu, iv - 0.5, 1),
           cell_ray(g, iu, iv + 0.5, 1)]
  points = [plane_point(a, b, ka, c) for (a, b) in edges]
  blo, bhi = minmax(points[1][kb], points[2][kb])
  clo, chi = minmax(points[3][3], points[4][3])
  (; ka, kb, c, edges, points, blo, bhi, clo, chi)
end

# Left: the source, the detector, one cell with the rays through the midpoints of its four edges,
# and their crossings with one layer's center plane, where they span the footprint rectangle.
# Right: that plane seen along the principal axis of the view, with the voxels of the layer colored by their weight
# for the cell (from the back projection), the rectangle and the four crossings.
function footprint_figure(; layer=5, azimuth=1.2pi, elevation=0.15pi)
  g = cone()
  image = ImageGeom(; dims=CONE_DIMS, deltas=(1.0, 1.0, 1.0))
  f = cone_footprint(g, layer)
  nx, ny, nz = CONE_DIMS
  fig = Figure(size=(1000, 520))
  ax = Axis3(fig[1, 1]; aspect=:data, azimuth, elevation, title="the four edge rays of one cell")
  hidedecorations!(ax); hidespines!(ax)
  # the outline of the volume
  box = [Point3f(x, y, z) for x in (-nx / 2, nx / 2), y in (-ny / 2, ny / 2), z in (-nz / 2, nz / 2)]
  for (i, j) in ((1, 2), (3, 4), (5, 6), (7, 8), (1, 3), (2, 4), (5, 7), (6, 8), (1, 5), (2, 6), (3, 7), (4, 8))
    lines!(ax, [box[i], box[j]]; color=:gray60, linewidth=0.8)
  end
  # the center plane of the layer
  plane(q, r) = f.ka == 1 ? Point3f(f.c, q, r) : Point3f(q, f.c, r)
  half = (CONE_DIMS[f.kb] / 2, nz / 2)
  poly!(ax, [plane(-half[1], -half[2]), plane(half[1], -half[2]), plane(half[1], half[2]), plane(-half[1], half[2])];
        color=(:steelblue, 0.15), strokecolor=:steelblue, strokewidth=1)
  # the detector and the cell
  nu, nv = detectorsize(g)
  corners = [Point3f(cell_ray(g, u, v, 1)[2]) for (u, v) in ((0.5, 0.5), (nu + 0.5, 0.5), (nu + 0.5, nv + 0.5), (0.5, nv + 0.5))]
  poly!(ax, corners; color=(:steelblue, 0.15), strokecolor=:steelblue, strokewidth=1)
  for u in 1.5:nu-0.5
    lines!(ax, [Point3f(cell_ray(g, u, 0.5, 1)[2]), Point3f(cell_ray(g, u, nv + 0.5, 1)[2])]; color=(:steelblue, 0.5), linewidth=0.6)
  end
  for v in 1.5:nv-0.5
    lines!(ax, [Point3f(cell_ray(g, 0.5, v, 1)[2]), Point3f(cell_ray(g, nu + 0.5, v, 1)[2])]; color=(:steelblue, 0.5), linewidth=0.6)
  end
  iu, iv = CONE_CELL
  cell = [Point3f(cell_ray(g, u, v, 1)[2]) for (u, v) in ((iu - 0.5, iv - 0.5), (iu + 0.5, iv - 0.5), (iu + 0.5, iv + 0.5), (iu - 0.5, iv + 0.5))]
  lines!(ax, [cell; cell[1:1]]; color=:darkorange, linewidth=3)
  colors = (:darkorange, :darkorange, :purple, :purple)
  for ((a, b), col) in zip(f.edges, colors)
    lines!(ax, [Point3f(a), Point3f(b)]; color=col, linewidth=1.2)
  end
  scatter!(ax, Point3f.(f.points); color=collect(colors), markersize=9)
  rect = [plane(f.blo, f.clo), plane(f.bhi, f.clo), plane(f.bhi, f.chi), plane(f.blo, f.chi), plane(f.blo, f.clo)]
  lines!(ax, rect; color=:black, linewidth=2)
  scatter!(ax, [Point3f(f.edges[1][1])]; color=:black, markersize=10)
  # the plane seen along the principal axis of the view
  w = cell_weights3(image, g)
  slice = f.ka == 1 ? w[layer, :, :] : w[:, layer, :]
  bname = f.kb == 1 ? "x" : "y"
  ax2 = Axis(fig[1, 2]; aspect=DataAspect(), xlabel=bname, ylabel="z",
             title="layer $(f.ka == 1 ? "x" : "y") = $(f.c): footprint and weights")
  bc = range(-(CONE_DIMS[f.kb] - 1) / 2, (CONE_DIMS[f.kb] - 1) / 2, CONE_DIMS[f.kb])
  zc = range(-(nz - 1) / 2, (nz - 1) / 2, nz)
  hm = heatmap!(ax2, bc, zc, slice; colormap=:Blues, colorrange=(1e-6, maximum(slice)), lowclip=:white)
  for q in -CONE_DIMS[f.kb]/2:CONE_DIMS[f.kb]/2
    lines!(ax2, [q, q], [-nz / 2, nz / 2]; color=:gray70, linewidth=0.5)
  end
  for q in -nz/2:nz/2
    lines!(ax2, [-CONE_DIMS[f.kb] / 2, CONE_DIMS[f.kb] / 2], [q, q]; color=:gray70, linewidth=0.5)
  end
  for (j, k) in Iterators.product(eachindex(bc), eachindex(zc))
    slice[j, k] > 0 && text!(ax2, bc[j], zc[k]; text=string(round(slice[j, k]; digits=2)), align=(:center, :center),
                             fontsize=11)
  end
  lines!(ax2, [Point2f(f.blo, f.clo), Point2f(f.bhi, f.clo), Point2f(f.bhi, f.chi), Point2f(f.blo, f.chi),
               Point2f(f.blo, f.clo)]; color=:black, linewidth=2)
  scatter!(ax2, [Point2f(p[f.kb], p[3]) for p in f.points]; color=collect(colors), markersize=9)
  Colorbar(fig[1, 3], hm; label="weight of the voxel")
  Legend(fig[2, 1:3], [LineElement(color=:darkorange), LineElement(color=:purple), LineElement(color=:black, linewidth=2)],
         ["rays through the two edges that bound the cell along u", "rays through the two edges along v",
          "footprint in the layer"]; orientation=:horizontal, framevisible=false)
  colsize!(fig.layout, 1, Relative(0.5))
  fig
end

function cell_weights3(image, g)
  op = DistanceDrivenProjector(image, g; eltype=Float64)
  y = zeros(datasize(op))
  y[CONE_CELL..., 1] = 1
  reshape(adjoint(op) * y, image.dims)
end

# Coverage of the pixels by the cells of one parallel-beam view, one cell per pixel: Siddon, Joseph
# and distance-driven (rows), at 45 degrees and at 0.3 rad (columns).
function coverage_three(n)
  image = ImageGeom(; dims=(n, n), deltas=(1.0, 1.0))
  fig = Figure(size=(640, 860))
  names = ("Siddon", "Joseph", "distance-driven")
  for (i, P) in enumerate((SiddonProjector, JosephProjector, DistanceDrivenProjector)), (j, angle) in enumerate((pi / 4, 0.3))
    g = parallel_beam(; angles=[angle], nu=2n, nv=1, du=1.0)
    c = colsums(P(image, g; eltype=Float64))
    ax = Axis(fig[i, j]; aspect=DataAspect(), title="$(names[i]), $(j == 1 ? "45°" : "17°")")
    hidedecorations!(ax)
    hm = heatmap!(ax, c; colormap=:RdBu, colorrange=(0.7, 1.3))
    i == 1 && j == 2 && Colorbar(fig[1:3, 3], hm; label="coverage")
  end
  fig
end
