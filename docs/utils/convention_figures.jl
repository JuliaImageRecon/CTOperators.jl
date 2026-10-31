# Figure of the conventions page: the geometry of one cone-beam view seen from above (CairoMakie).
# Included invisibly by the page.

using CTOperators, CairoMakie

# One view of `circular_cone_beam` at angle `beta`, top view (x, y): image square, source circle,
# source, detector with its u direction, the outer rays and the angle beta from the y axis.
function topview_figure(; beta=pi / 6, dso=3.0, dsd=5.0, nu=8, du=0.5, n=2.0)
  g = circular_cone_beam(; angles=[beta], dso, dsd, nu, nv=1, du)
  s = g.source[1:2, 1]
  c = g.centre[1:2, 1]
  u = g.u[1:2, 1]
  first_cell = c - (nu - 1) / 2 * u
  last_cell = c + (nu - 1) / 2 * u

  fig = Figure(size=(560, 560))
  ax = Axis(fig[1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", limits=((-4, 4), (-4.2, 4)))
  # image (the volume) and the circle of the source
  poly!(ax, Rect2f(-n / 2, -n / 2, n, n); color=(:gray, 0.25), strokecolor=:gray40, strokewidth=1)
  text!(ax, -n / 2 + 0.1, n / 2 - 0.1; text="image", align=(:left, :top), fontsize=13, color=:gray30)
  t = range(0, 2pi, 200)
  lines!(ax, dso .* cos.(t), dso .* sin.(t); color=:gray70, linestyle=:dash)
  # the rays to the outermost cells and the central ray
  for p in (first_cell, last_cell)
    lines!(ax, [s[1], p[1]], [s[2], p[2]]; color=(:orange, 0.6))
  end
  lines!(ax, [s[1], c[1]], [s[2], c[2]]; color=:orange, linestyle=:dot)
  # detector cells and the u direction
  cells = [c + (i - (nu + 1) / 2) * u for i in 1:nu]
  lines!(ax, [first_cell[1] - u[1] / 2, last_cell[1] + u[1] / 2], [first_cell[2] - u[2] / 2, last_cell[2] + u[2] / 2];
         color=:steelblue, linewidth=4)
  scatter!(ax, first.(cells), last.(cells); color=:white, strokecolor=:steelblue, strokewidth=1, markersize=6)
  arrows2d!(ax, [last_cell[1] + 0.4 * u[1] / du], [last_cell[2] + 0.4 * u[2] / du], [0.8 * u[1] / du], [0.8 * u[2] / du];
            color=:steelblue)
  text!(ax, last_cell[1] + 1.3 * u[1] / du, last_cell[2] + 1.3 * u[2] / du; text="u", color=:steelblue,
        fontsize=16, align=(:center, :center))
  text!(ax, first_cell[1], first_cell[2]; text="cell 1", color=:steelblue, fontsize=12, align=(:left, :top),
        offset=(8, -6))
  text!(ax, c[1], c[2]; text="detector", color=:steelblue, fontsize=13, align=(:left, :top), offset=(8, -6))
  # source and the angle beta from the y axis
  scatter!(ax, [s[1]], [s[2]]; color=:orange, markersize=14)
  text!(ax, s[1], s[2]; text="source", align=(:right, :bottom), offset=(-6, 6), fontsize=13)
  lines!(ax, [0, 0], [0, 3.8]; color=:black, linewidth=0.8)
  arc = range(pi / 2, pi / 2 + beta, 30)
  lines!(ax, 1.2 .* cos.(arc), 1.2 .* sin.(arc); color=:black)
  text!(ax, 1.35 * cos(pi / 2 + beta / 2), 1.35 * sin(pi / 2 + beta / 2); text="β", fontsize=18,
        align=(:right, :bottom))
  scatter!(ax, [0], [0]; color=:black, markersize=6)
  text!(ax, 0, 0; text="z axis (rotation)", align=(:left, :top), offset=(6, -4), fontsize=12)
  fig
end
