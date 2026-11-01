# Figures of the tutorials (CairoMakie): scans seen from above, sinograms and image slices. Included
# invisibly by the tutorial pages, which load CairoMakie.

include(joinpath(@__DIR__, "geometry_figures.jl"))
using LinearAlgebra: norm, normalize

# the extent of the image in x and y (outer faces of the voxels)
function image_extent(image, d)
  lo, hi = extrema(axis(image, d))
  h = abs(image.deltas[d]) / 2
  lo - h, hi + h
end

# the rays of the central detector row of view k: (start, end) in x, y
function central_rays(g, k; count=9)
  nu, nv = detectorsize(g)
  iv = div(nv + 1, 2)
  cells = round.(Int, range(1, nu, count))
  index(iu) = LinearIndices((nu, nv, nviews(g)))[iu, iv, k]
  [ray(g, index(iu)) for iu in cells], [ray(g, index(iu)) for iu in 1:nu]
end

"""
    topview!(ax, g, image; view=1, count=9, reach=150)

A scan seen from above: the image (gray square), the rays of one view through the central detector
row (orange), the detector (blue) and, for cone beams, the source positions of all views (gray dots)
with the source of the view (orange dot). Parallel-beam rays are lines, drawn `reach` long on both
sides of the detector.
"""
function topview!(ax, g, image; view=1, count=9, reach=150)
  (x0, x1), (y0, y1) = image_extent(image, 1), image_extent(image, 2)
  poly!(ax, Rect2f(x0, y0, x1 - x0, y1 - y0); color=(:gray, 0.3), strokecolor=:gray40, strokewidth=1)
  shown, all = central_rays(g, view; count)
  xy(p) = Point2f(p[1], p[2])
  if raytype(g) == RayLine()
    segs = [xy(a - reach * normalize(b - a)) => xy(a + reach * normalize(b - a)) for (a, b) in shown]
    detector = [xy(a) for (a, _) in all]
  else
    scatter!(ax, g.source[1, :], g.source[2, :]; color=:gray60, markersize=4)
    segs = [xy(a) => xy(b) for (a, b) in shown]
    detector = [xy(b) for (_, b) in all]
    scatter!(ax, [xy(g.source[:, view])]; color=:orange, markersize=12)
  end
  linesegments!(ax, segs; color=(:orange, 0.6))
  lines!(ax, detector; color=:steelblue, linewidth=4)
  ax
end

topview_axis(pos; title="", limits=nothing) = Axis(pos; aspect=DataAspect(), xlabel="x", ylabel="y", title, limits)

# several scans from above, side by side; `scans` holds (title, geometry, view) tuples
function topview_figure(scans, image; size=(400 * length(scans), 420), limits=nothing, kwargs...)
  fig = Figure(; size)
  for (j, (title, g, view)) in enumerate(scans)
    topview!(topview_axis(fig[1, j]; title, limits), g, image; view, kwargs...)
  end
  fig
end

# the sinogram of 2D data (nu, 1, nviews): detector cell against view
function sinogram_figure(y; title="sinogram", size=(560, 380), angles=nothing)
  fig = Figure(; size)
  ys = angles === nothing ? (1:Base.size(y, 3)) : rad2deg.(angles)
  ax = Axis(fig[1, 1]; xlabel="detector cell", ylabel=angles === nothing ? "view" : "angle (degrees)", title)
  hm = heatmap!(ax, 1:Base.size(y, 1), ys, y[:, 1, :]; colormap=:grays)
  Colorbar(fig[1, 2], hm; label="line integral")
  fig
end

# 2D images side by side, with a common window; `images` holds (title, image) pairs, or
# (title, image, colorrange) for an image with its own window. With `truth`, the titles of the other
# images in the common window show their relative error.
function images_figure(images; colorrange=(0, 1), size=(300 * length(images), 330), truth=nothing)
  fig = Figure(; size)
  for (j, entry) in enumerate(images)
    title, img = entry[1], entry[2]
    own = length(entry) == 3
    t = truth === nothing || own || img === truth ? title : "$title\nerror $(relative_error(img, truth))"
    ax = Axis(fig[1, j]; aspect=DataAspect(), title=t)
    hidedecorations!(ax)
    heatmap!(ax, dropdims_2d(img); colormap=:grays, colorrange=own ? entry[3] : colorrange)
  end
  fig
end

relative_error(img, truth) = round(norm(img - truth) / norm(truth); digits=3)

dropdims_2d(img) = ndims(img) == 3 ? img[:, :, 1] : img

# 3D images: an axial slice (x, y at index z) and a sagittal slice (y, z at index x) per image
function slices_figure(images; z, x, colorrange=(0, 1), size=(280 * length(images), 560), truth=nothing)
  fig = Figure(; size)
  for (j, (title, img)) in enumerate(images)
    t = truth === nothing || img === truth ? title : "$title\nerror $(relative_error(img, truth))"
    ax1 = Axis(fig[1, j]; aspect=DataAspect(), title=t, ylabel=j == 1 ? "axial (x, y)" : "")
    ax2 = Axis(fig[2, j]; aspect=DataAspect(), ylabel=j == 1 ? "sagittal (y, z)" : "")
    heatmap!(ax1, img[:, :, z]; colormap=:grays, colorrange)
    heatmap!(ax2, img[x, :, :]; colormap=:grays, colorrange)
    hidexdecorations!(ax1); hidexdecorations!(ax2)
    hideydecorations!(ax1; label=false); hideydecorations!(ax2; label=false)
  end
  fig
end

# one view of a cone-beam scan in 3D with the path of the source; with `height`, next to the source
# height per view
function cone_scan_figure(g::ConeBeam, image; view=1, title="", height=true,
                          size=height ? (900, 400) : (560, 440))
  fig = Figure(; size)
  draw_cone_view!(scene_axis(fig, (1, 1); title), g, image; view)
  if height
    ax = Axis(fig[1, 2]; xlabel="view", ylabel="source z", title="height of the source")
    lines!(ax, g.source[3, :]; color=:orange)
    colsize!(fig.layout, 1, Relative(0.6))
  end
  fig
end

# the ray directions of all views of parallel-beam scans, as lines through the center of the image;
# `scans` holds (title, geometry) pairs
function directions_figure(scans, image; size=(330 * length(scans), 360), reach=100)
  fig = Figure(; size)
  for (j, (title, g)) in enumerate(scans)
    ax = Axis(fig[1, j]; aspect=DataAspect(), title, limits=(-reach, reach, -reach, reach))
    hidedecorations!(ax)
    (x0, x1), (y0, y1) = image_extent(image, 1), image_extent(image, 2)
    poly!(ax, Rect2f(x0, y0, x1 - x0, y1 - y0); color=(:gray, 0.3), strokecolor=:gray40, strokewidth=1)
    d = [Point2f(normalize(g.direction[1:2, k])) for k in 1:nviews(g)]
    linesegments!(ax, [-reach * p => reach * p for p in d]; color=(:orange, 0.7))
  end
  fig
end

# slices of 3D images at heights `zs` (world coordinates) and an x-z slice at the y index `iy`; one
# row per image, `images` holds (title, image) pairs
function depth_figure(images, image; zs, iy, colorrange=(0, 1), size=(250 * (length(zs) + 1), 270 * length(images)))
  fig = Figure(; size)
  zaxis = axis(image, 3)
  for (i, (title, img)) in enumerate(images), (j, z) in enumerate(zs)
    k = argmin(abs.(zaxis .- z))
    ax = Axis(fig[i, j]; aspect=DataAspect(), title=i == 1 ? "z = $z mm" : "", ylabel=j == 1 ? title : "")
    hidexdecorations!(ax); hideydecorations!(ax; label=false)
    heatmap!(ax, img[:, :, k]; colormap=:grays, colorrange)
  end
  for (i, (_, img)) in enumerate(images)
    ax = Axis(fig[i, length(zs) + 1]; aspect=DataAspect(), title=i == 1 ? "x-z slice" : "")
    hidedecorations!(ax)
    heatmap!(ax, img[:, iy, :]; colormap=:grays, colorrange)
  end
  fig
end
