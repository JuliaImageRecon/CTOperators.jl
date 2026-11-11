# Figures of the explanation page "System Matrix" (WGLMakie through Bonito). Included invisibly by
# the page.

using CTOperators, WGLMakie, Bonito, CairoMakie

# One ray at angle theta (to the x axis) through the n x n image of unit pixels, shifted by `shift`
# perpendicular to itself: its principal axis, the layers along that axis, and the pieces of the ray
# per layer (between the layer planes) and per pixel (between all pixel boundaries), with the parts
# of the pieces inside the image.
function layer_scene(theta, n; shift=0.3)
  d = (cos(theta), sin(theta))
  c = (-shift * d[2], shift * d[1])
  k = abs(d[1]) >= abs(d[2]) ? 1 : 2
  half = n / 2
  P(t) = Point2f(c[1] + t * d[1], c[2] + t * d[2])
  inside(p) = abs(p[1]) <= half + 1e-6 && abs(p[2]) <= half + 1e-6
  # the parameters where the ray crosses the planes w of axis `axis`
  crossings(axis) = abs(d[axis]) < 1e-12 ? Float64[] : [(w - c[axis]) / d[axis] for w in -half:half]
  # the part of the piece [t1, t2] inside the image (the image is convex)
  function clipped(t1, t2)
    ts = range(t1, t2, 201)
    ins = [t for t in ts if inside(P(t))]
    isempty(ins) ? nothing : (first(ins), last(ins))
  end
  # per layer: between consecutive layer planes
  tl = sort(crossings(k))
  layer_pieces = [P(tl[j]) => P(tl[j+1]) for j in 1:n]
  layer_inside = [P(r[1]) => P(r[2]) for r in (clipped(tl[j], tl[j+1]) for j in 1:n) if !isnothing(r)]
  # per pixel: between consecutive crossings of all pixel boundaries, inside the image
  ts = sort(vcat(crossings(1), crossings(2)))
  pixel_pieces = Pair{Point2f,Point2f}[]
  lengths = Float64[]
  for j in 1:length(ts)-1
    ts[j+1] - ts[j] > 1e-9 || continue
    inside(P((ts[j] + ts[j+1]) / 2)) || continue
    push!(pixel_pieces, P(ts[j]) => P(ts[j+1]))
    push!(lengths, ts[j+1] - ts[j])
  end
  axis = k == 1 ? "x" : "y"
  len = round(1 / abs(d[k]); digits=2)
  ray = "ray at $(round(Int, rad2deg(theta)))°"
  (; k, layer_pieces, layer_inside, pixel_pieces,
   layer_info="$ray: principal axis $axis, $(round(Int, rad2deg(acos(abs(d[k])))))° to it, $len per layer",
   pixel_info="$ray: from $(round(minimum(lengths); digits=2)) to $(round(maximum(lengths); digits=2)) per pixel")
end

# Draws one panel: the layers along x and along y (shaded alternately, one set visible), the pixel
# grid, the pieces in alternating colors (parts outside the image dashed) and a text line inside the
# axis. Only data and visibility change with the slider, not the layout, which keeps the recorded
# states of the static page light.
function layer_panel!(ax, scene, n, pieces, inside_pieces, info)
  half = n / 2
  colors = [isodd(j) ? (:steelblue, 0.10) : (:steelblue, 0.25) for j in 1:n]
  along_x = map(s -> s.k == 1, scene)
  poly!(ax, [Rect2f(-half + j - 1, -half, 1, n) for j in 1:n]; color=colors, strokecolor=:steelblue,
        strokewidth=1.5, visible=along_x)
  poly!(ax, [Rect2f(-half, -half + j - 1, n, 1) for j in 1:n]; color=colors, strokecolor=:steelblue,
        strokewidth=1.5, visible=map(!, along_x))
  grid = [w for v in -half:half for w in (Point2f(v, -half) => Point2f(v, half), Point2f(-half, v) => Point2f(half, v))]
  linesegments!(ax, grid; color=:gray70, linewidth=0.5)
  linesegments!(ax, map(pieces, scene); color=:gray40, linestyle=:dash, linewidth=1)
  linesegments!(ax, map(inside_pieces, scene); color=map(s -> [isodd(j) ? :darkorange : :black
                                                                for j in 1:length(inside_pieces(s))], scene), linewidth=3)
  text!(ax, -half, half + 0.15; text=map(info, scene), align=(:left, :bottom), fontsize=13)
end

# the pieces per layer, and with `pixels=true` next to them the pieces per pixel of the same ray
function layer_figure(scene, n; pixels=true)
  fig = Figure(size=pixels ? (1000, 540) : (540, 540))
  lim = ((-n / 2 - 0.7, n / 2 + 0.7), (-n / 2 - 0.7, n / 2 + 1.2))
  ax1 = Axis(fig[1, 1]; aspect=DataAspect(), limits=lim, xlabel="x", ylabel="y", title="pieces per layer")
  layer_panel!(ax1, scene, n, s -> s.layer_pieces, s -> s.layer_inside, s -> s.layer_info)
  if pixels
    ax2 = Axis(fig[1, 2]; aspect=DataAspect(), limits=lim, xlabel="x", ylabel="y", title="pieces per pixel")
    layer_panel!(ax2, scene, n, s -> s.layer_pieces, s -> s.pixel_pieces, s -> s.pixel_info)
  end
  fig
end

# The interactive figure: the slider turns the ray in steps of 9° from 0° to 171° (through 45° and
# 135°), the pieces per layer only (one panel keeps the recorded states of the page light).
function principal_axis_app(n=8; nsweep=20)
  App() do session
    k = Bonito.Slider(1:nsweep)
    scene = map(i -> layer_scene(pi * (i - 1) / nsweep, n), k.value)
    fig = layer_figure(scene, n; pixels=false)
    Bonito.record_states(session, DOM.div(DOM.div("ray ", k), fig))
  end
end

# a static figure for one angle: the pieces per layer next to the pieces per pixel of the same ray
principal_axis_figure(theta, n=8) = layer_figure(Observable(layer_scene(theta, n)), n)
