# Figures of the explanation page "System Matrix" (WGLMakie through Bonito). Included invisibly by
# the page.

using CTOperators, WGLMakie, Bonito, CairoMakie

# One ray through the center of the n x n image of unit pixels at angle theta (to the x axis): its
# principal axis, the layers along that axis, the pieces of the ray in the layers and their middles.
function principal_axis_scene(theta, n)
  d = (cos(theta), sin(theta))
  k = abs(d[1]) >= abs(d[2]) ? 1 : 2
  edges = -n/2:n/2
  # the points where the ray crosses the boundaries between the layers
  point(w) = k == 1 ? Point2f(w, w * d[2] / d[1]) : Point2f(w * d[1] / d[2], w)
  pieces = [point(edges[j]) => point(edges[j+1]) for j in 1:n]
  middles = [Point2f((p[1] + q[1]) / 2, (p[2] + q[2]) / 2) for (p, q) in pieces]
  layers = [k == 1 ? Rect2f(edges[j], -n / 2, 1, n) : Rect2f(-n / 2, edges[j], n, 1) for j in 1:n]
  to_axis = round(Int, rad2deg(acos(abs(d[k]))))
  title = "ray at $(round(Int, rad2deg(theta)))°: principal axis $(k == 1 ? "x" : "y"), " *
          "angle to it $(to_axis)°, length per layer $(round(1 / abs(d[k]); digits=2))"
  (; pieces, middles, layers, title)
end

# The slider turns one ray about the center, in steps of 5° from 0° to 175°.
function principal_axis_app(n=8; nsweep=36)
  App() do session
    k = Bonito.Slider(1:nsweep)
    scene = map(i -> principal_axis_scene(pi * (i - 1) / nsweep, n), k.value)
    fig = Figure(size=(620, 600))
    ax = Axis(fig[1, 1]; aspect=DataAspect(), limits=((-n / 2 - 0.5, n / 2 + 0.5), (-n / 2 - 0.5, n / 2 + 0.5)),
              xlabel="x", ylabel="y", title=map(s -> s.title, scene), titlesize=14)
    colors = [isodd(j) ? (:steelblue, 0.10) : (:steelblue, 0.25) for j in 1:n]
    poly!(ax, map(s -> s.layers, scene); color=colors, strokecolor=:steelblue, strokewidth=1.5)
    for w in -n/2:n/2
      lines!(ax, [w, w], [-n / 2, n / 2]; color=:gray70, linewidth=0.5)
      lines!(ax, [-n / 2, n / 2], [w, w]; color=:gray70, linewidth=0.5)
    end
    linesegments!(ax, map(s -> s.pieces, scene); color=[isodd(j) ? :darkorange : :black for j in 1:n], linewidth=3)
    scatter!(ax, map(s -> s.middles, scene); color=:orange, strokecolor=:black, strokewidth=1, markersize=10)
    Bonito.record_states(session, DOM.div(DOM.div("ray ", k), fig))
  end
end
