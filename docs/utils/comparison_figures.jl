# Figures of the explanation page "Comparing Projectors" (CairoMakie): the same checks for several
# projectors and ratios of detector cell size to pixel size. Included invisibly by the page.

using CairoMakie

# a 2D image of unit pixels and a parallel-beam scan with cells of width du (in pixels) that covers it
comparison_image(n) = ImageGeom(; dims=(n, n), deltas=(1.0, 1.0))
comparison_scan(n, du, angles) = parallel_beam(; angles, nu=ceil(Int, 1.5n / du), nv=1, du)

# the variation of an image inside the circle of radius r around its center, in percent
function variation(img, r)
  n = size(img, 1)
  inside = [img[i, j] for i in 1:n, j in 1:n if hypot(i - (n + 1) / 2, j - (n + 1) / 2) <= r]
  m = sum(inside) / length(inside)
  100 * sqrt(sum((inside .- m) .^ 2) / length(inside)) / m
end

# `models` holds (name, projector, geometry -> geometry) tuples, `ratios` the cell sizes in pixels.
# One panel per model and ratio; `panel!(ax, op, du)` draws it.
function comparison_grid(panel!, models, ratios, n, angles, window; size=(250 * length(ratios) + 140, 250 * length(models)))
  fig = Figure(; size)
  Colorbar(fig[1:length(models), length(ratios)+1]; colormap=:RdBu, limits=window, label="coverage")
  for (i, (name, P, wrap)) in enumerate(models), (j, du) in enumerate(ratios)
    ax = Axis(fig[i, j]; title=i == 1 ? "cells of $du pixels" : "", ylabel=j == 1 ? name : "")
    op = P(comparison_image(n), wrap(comparison_scan(n, du, angles)); eltype=Float64)
    panel!(ax, op, du)
  end
  fig
end

function image_panel!(ax, img, window; crop=nothing)
  ax.aspect = DataAspect()
  ax.xticksvisible = ax.yticksvisible = false
  ax.xticklabelsvisible = ax.yticklabelsvisible = false
  shown = isnothing(crop) ? img : img[crop, crop]
  heatmap!(ax, shown; colormap=:RdBu, colorrange=window)
end

# the back projection of one view of ones, scaled so that even coverage is 1
function view_figure(models, ratios; n=64, angle=0.3, crop=25:40)
  window = (0, 2)
  comparison_grid(models, ratios, n, [angle], window) do ax, op, du
    image_panel!(ax, du .* colsums(op), window; crop)
  end
end

# the back projection of all views of ones (the column sums), scaled so that even coverage is 1,
# with its variation below the panel
function support_figure(models, ratios; n=64, angles=range(0, pi, 181)[1:180])
  window = (0.8, 1.2)
  comparison_grid(models, ratios, n, angles, window) do ax, op, du
    c = du .* colsums(op) ./ length(angles)
    image_panel!(ax, c, window)
    circle = 24 .* (cos.(range(0, 2pi, 100)), sin.(range(0, 2pi, 100)))
    lines!(ax, (n + 1) / 2 .+ circle[1], (n + 1) / 2 .+ circle[2]; color=:black, linewidth=0.7)
    v = variation(c, 24)
    ax.xlabel = v < 0.05 ? "variation < 0.05 %" : "variation $(round(v; digits=1)) %"
  end
end
