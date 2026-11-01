# Figure of the guide "Implement a CT operator" (CairoMakie): the sampling projector on a small 2D
# grid. Included invisibly by the page.

using CairoMakie
using LinearAlgebra: norm

# Samples of the ray from a to b (voxel coordinates, pixel (i, j) is the unit square centered at (i,
# j)) on a grid of nx × ny pixels, `samples` per pixel length: the sample points and the weight of
# every pixel (the summed step lengths of the samples nearest to it). The same rule as the kernel.
function sampling_weights(a, b, nx, ny, samples)
  d = b - a
  amin, amax = 0.0, 1.0
  for k in 1:2
    a1, a2 = (0.5 - a[k]) / d[k], ((k == 1 ? nx : ny) + 0.5 - a[k]) / d[k]
    amin, amax = max(amin, min(a1, a2)), min(amax, max(a1, a2))
  end
  n = ceil(Int, samples * norm(d) * (amax - amin))
  dalpha = (amax - amin) / n
  points = [a + (amin + (k - 0.5) * dalpha) * d for k in 1:n]
  weights = zeros(nx, ny)
  for p in points
    weights[clamp(round(Int, p[1]), 1, nx), clamp(round(Int, p[2]), 1, ny)] += norm(d) * dalpha
  end
  points, weights, a + amin * d, a + amax * d
end

function sampling_figure(; nx=8, ny=5, a=[-0.5, 1.0], b=[9.5, 5.2])
  fig = Figure(size=(900, 380))
  for (j, samples) in enumerate((1, 2))
    points, weights, entry, exit = sampling_weights(a, b, nx, ny, samples)
    ax = Axis(fig[1, j]; aspect=DataAspect(), title="$samples sample$(samples == 1 ? "" : "s") per pixel length",
              limits=(-0.7, nx + 1.7, 0, ny + 1))
    hidedecorations!(ax)
    hm = heatmap!(ax, 1:nx, 1:ny, replace(weights, 0.0 => NaN); colormap=:Blues, colorrange=(0, 1.6),
                  nan_color=:white)
    for x in 0.5:1:nx+0.5
      lines!(ax, [x, x], [0.5, ny + 0.5]; color=:gray60, linewidth=0.8)
    end
    for y in 0.5:1:ny+0.5
      lines!(ax, [0.5, nx + 0.5], [y, y]; color=:gray60, linewidth=0.8)
    end
    lines!(ax, [a[1], b[1]], [a[2], b[2]]; color=(:orange, 0.5), linewidth=1.5)
    lines!(ax, [entry[1], exit[1]], [entry[2], exit[2]]; color=:orange, linewidth=2.5)
    scatter!(ax, first.(points), last.(points); color=:black, markersize=8)
    j == 2 && Colorbar(fig[1, 3], hm; label="weight (length)")
  end
  fig
end
