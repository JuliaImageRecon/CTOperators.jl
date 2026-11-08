# Figure of the explanation page "Masks" (CairoMakie): rays through a 2D image with a mask; each ray
# is traced only inside the bounding box of the image mask. Included invisibly by the page.

using CairoMakie

function mask_box_figure(; n=12)
  # an elliptical mask; its box is computed from it as in the projectors
  mask = [((i - 6.5) / 3.2)^2 + ((j - 8) / 2.6)^2 <= 1 for i in 1:n, j in 1:n]
  rows, cols = findall(vec(any(mask; dims=2))), findall(vec(any(mask; dims=1)))
  fig = Figure(size=(520, 460))
  ax = Axis(fig[1, 1]; aspect=DataAspect(), title="rays: solid where traced, dashed where not",
            limits=(-0.5, n + 1.5, -0.5, n + 1.5))
  hidedecorations!(ax); hidespines!(ax)
  heatmap!(ax, 1:n, 1:n, Float32.(mask); colormap=[:white, (:steelblue, 0.45)], colorrange=(0, 1))
  for w in 0.5:1:n+0.5
    lines!(ax, [w, w], [0.5, n + 0.5]; color=:gray80, linewidth=0.6)
    lines!(ax, [0.5, n + 0.5], [w, w]; color=:gray80, linewidth=0.6)
  end
  # the box around the mask
  lo, hi = (first(rows) - 0.5, first(cols) - 0.5), (last(rows) + 0.5, last(cols) + 0.5)
  poly!(ax, Rect2f(lo..., (hi .- lo)...); color=:transparent, strokecolor=:steelblue, strokewidth=2)
  text!(ax, hi[1], hi[2]; text="bounding box of the image mask", align=(:left, :bottom), offset=(4, 2), fontsize=13, color=:steelblue)
  # rays: dashed where they are not traced, solid inside the box
  for (a, b) in (((-0.5, 2.0), (13.5, 9.0)), ((-0.5, 7.5), (13.5, 11.0)), ((2.0, -0.5), (11.0, 13.5)), ((-0.5, 1.0), (13.5, 2.5)))
    lines!(ax, [Point2f(a), Point2f(b)]; color=(:darkorange, 0.5), linestyle=:dash)
    t0, t1 = clip_segment(a, b, lo, hi)
    if t0 < t1
      p0, p1 = a .+ t0 .* (b .- a), a .+ t1 .* (b .- a)
      lines!(ax, [Point2f(p0), Point2f(p1)]; color=:darkorange, linewidth=3)
    end
  end
  text!(ax, 0.5, 0.2; text="grid: image voxels; blue: image mask", align=(:left, :top), fontsize=12, color=:gray30)
  fig
end

# the parameters t in [0, 1] of the segment a -> b inside the box [lo, hi]
function clip_segment(a, b, lo, hi)
  t0, t1 = 0.0, 1.0
  for k in 1:2
    d = b[k] - a[k]
    if iszero(d)
      (lo[k] <= a[k] <= hi[k]) || return 1.0, 0.0
    else
      s0, s1 = (lo[k] - a[k]) / d, (hi[k] - a[k]) / d
      t0, t1 = max(t0, min(s0, s1)), min(t1, max(s0, s1))
    end
  end
  t0, t1
end
