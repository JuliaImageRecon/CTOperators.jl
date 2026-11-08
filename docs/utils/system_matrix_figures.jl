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
  layers = [k == 1 ? Rect2f(-half + j - 1, -half, 1, n) : Rect2f(-half, -half + j - 1, n, 1) for j in 1:n]
  axis = k == 1 ? "x" : "y"
  len = round(1 / abs(d[k]); digits=2)
  (; layers, layer_pieces, layer_inside, pixel_pieces,
   layer_title="per layer: $len each (principal axis $axis, $(round(Int, rad2deg(acos(abs(d[k])))))° to it)",
   pixel_title="per pixel: from $(round(minimum(lengths); digits=2)) to $(round(maximum(lengths); digits=2))",
   ray_title="ray at $(round(Int, rad2deg(theta)))°")
end

# Draws one panel: the layers (shaded alternately), the pixel grid, and the pieces in alternating
# colors; parts of pieces outside the image dashed.
function layer_panel!(ax, scene, n, pieces, inside_pieces)
  colors = [isodd(j) ? (:steelblue, 0.10) : (:steelblue, 0.25) for j in 1:n]
  poly!(ax, map(s -> s.layers, scene); color=colors, strokecolor=:steelblue, strokewidth=1.5)
  for w in -n/2:n/2
    lines!(ax, [w, w], [-n / 2, n / 2]; color=:gray70, linewidth=0.5)
    lines!(ax, [-n / 2, n / 2], [w, w]; color=:gray70, linewidth=0.5)
  end
  linesegments!(ax, map(pieces, scene); color=:gray40, linestyle=:dash, linewidth=1)
  linesegments!(ax, map(inside_pieces, scene); color=map(s -> [isodd(j) ? :darkorange : :black
                                                                for j in 1:length(inside_pieces(s))], scene), linewidth=3)
end

# The interactive figure: the slider turns the ray in steps of 5° from 0° to 175°. Left the pieces
# per layer, right the pieces per pixel, of the same ray.
function principal_axis_app(n=8; nsweep=36)
  App() do session
    k = Bonito.Slider(1:nsweep)
    scene = map(i -> layer_scene(pi * (i - 1) / nsweep, n), k.value)
    fig = Figure(size=(1000, 560))
    lim = ((-n / 2 - 0.7, n / 2 + 0.7), (-n / 2 - 0.7, n / 2 + 0.7))
    ax1 = Axis(fig[1, 1]; aspect=DataAspect(), limits=lim, xlabel="x", ylabel="y",
               title=map(s -> s.layer_title, scene), titlesize=14)
    ax2 = Axis(fig[1, 2]; aspect=DataAspect(), limits=lim, xlabel="x", ylabel="y",
               title=map(s -> s.pixel_title, scene), titlesize=14)
    Label(fig[0, 1:2], map(s -> s.ray_title, scene); fontsize=16, font=:bold)
    layer_panel!(ax1, scene, n, s -> s.layer_pieces, s -> s.layer_inside)
    layer_panel!(ax2, scene, n, s -> s.layer_pieces, s -> s.pixel_pieces)
    Bonito.record_states(session, DOM.div(DOM.div("ray ", k), fig))
  end
end

# the same as a static figure, for one angle (to check the figure without building the docs)
function principal_axis_figure(theta, n=8)
  scene = Observable(layer_scene(theta, n))
  fig = Figure(size=(1000, 560))
  ax1 = Axis(fig[1, 1]; aspect=DataAspect(), title=scene[].layer_title, titlesize=14)
  ax2 = Axis(fig[1, 2]; aspect=DataAspect(), title=scene[].pixel_title, titlesize=14)
  Label(fig[0, 1:2], scene[].ray_title; fontsize=16, font=:bold)
  layer_panel!(ax1, scene, n, s -> s.layer_pieces, s -> s.layer_inside)
  layer_panel!(ax2, scene, n, s -> s.layer_pieces, s -> s.pixel_pieces)
  fig
end
