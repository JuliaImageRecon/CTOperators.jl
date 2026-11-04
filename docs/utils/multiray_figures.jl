# Figures of the pages on several rays per detector cell (CairoMakie): rays of one parallel-beam
# view through a pixel grid, and the positions of the point strategies in one cell. Included
# invisibly by the pages.

using CairoMakie
using CTOperators: point_offset, subray_weight

# One parallel-beam view along y through an n × n pixel grid, with detector cells of width du
# (pixels) below it: the rays through the cell centers (left) and `k` regular rays per cell (right).
# Positions on the detector are in cell units; integers are the cell centers, half-integers the
# edges.
function rays_figure(; n=8, du=2.0, k=4, size=(820, 470))
  ncells = round(Int, n / du) + 2
  fig = Figure(; size)
  for (col, (title, offsets)) in enumerate((("one ray per cell, at its center", [0.0]),
                                            ("$k rays per cell", [(j + 0.5) / k - 0.5 for j in 0:k-1])))
    ax = Axis(fig[1, col]; aspect=DataAspect(), title, limits=(-0.5, n + 0.5, -2.6, n + 0.5))
    hidedecorations!(ax); hidespines!(ax)
    for w in 0:n
      lines!(ax, [w, w], [0, n]; color=:gray75, linewidth=0.8)
      lines!(ax, [0, n], [w, w]; color=:gray75, linewidth=0.8)
    end
    # the detector: cells c = 1, ..., ncells; the rays through their centers run in the middle of
    # pixels, so that they do not hide the lines of the grid
    x(c) = (c - (ncells + 1) / 2) * du + n / 2 + 0.5
    for c in 1:ncells
      poly!(ax, Rect2f(x(c - 0.5), -1.2, du, 0.5); color=isodd(c) ? (:steelblue, 0.25) : (:steelblue, 0.45),
            strokecolor=:steelblue, strokewidth=0.8)
      for o in offsets
        xr = x(c + o)
        0 <= xr <= n && lines!(ax, [xr, xr], [-0.7, n]; color=:darkorange, linewidth=1.5)
      end
    end
    if col == 1
      for c in 2:3
        text!(ax, x(c), -1.35; text="$c", align=(:center, :top), fontsize=13, color=:steelblue)
        text!(ax, x(c + 0.5), -1.35; text="$(c + 0.5)", align=(:center, :top), fontsize=11, color=:gray40)
      end
      text!(ax, n / 2, -2.0; text="positions on the detector, in cells", align=(:center, :top), fontsize=12, color=:gray30)
    end
  end
  fig
end

# The points of several strategies in one cell (-1/2 to 1/2 in u and v), with the subcells dotted
# and the marker area proportional to the weight. `strategies` holds (title, points, weights) tuples.
function points_figure(strategies; size=(220 * length(strategies), 270))
  fig = Figure(; size)
  for (col, (title, points, weights)) in enumerate(strategies)
    ax = Axis(fig[1, col]; aspect=DataAspect(), title, limits=(-0.9, 0.9, -0.9, 0.9))
    hidedecorations!(ax)
    poly!(ax, Rect2f(-0.5, -0.5, 1, 1); color=(:steelblue, 0.12), strokecolor=:steelblue, strokewidth=1.5)
    # the subcells of the strategies that have them
    if hasproperty(points, :su)
      for j in 1:points.su-1
        lines!(ax, fill(j / points.su - 0.5, 2), [-0.5, 0.5]; color=:steelblue, linestyle=:dot, linewidth=1)
      end
      for j in 1:points.sv-1
        lines!(ax, [-0.5, 0.5], fill(j / points.sv - 0.5, 2); color=:steelblue, linestyle=:dot, linewidth=1)
      end
    end
    K = typeof(points).parameters[1]
    offs = [point_offset(points, Float64, 3, 2, 1, k) for k in 1:K]
    w = [subray_weight(weights, o...) for o in offs]
    scatter!(ax, first.(offs), last.(offs); markersize=14 .* sqrt.(w ./ maximum(w)), color=:darkorange)
  end
  fig
end
