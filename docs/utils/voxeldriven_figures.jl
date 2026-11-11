# Figures of the explanation page "Voxel-Driven Method" (WGLMakie through Bonito for the interactive
# one, CairoMakie for the static ones). Included invisibly by the page. The geometry comes from the package: the point where a pixel center projects from
# `project_point`, the weights from the forward projection of that pixel.

using CTOperators, WGLMakie, Bonito, CairoMakie, LinearAlgebra
using CTOperators: cell_ray, project_point
using ImagePhantoms
using ImageGeoms: axis

# The 2D fan beam of the figures (as on the distance-driven page): an n x n image of unit pixels,
# the source at distance dso from the center, a detector of nu cells of width du at distance dsd
# from the source, one view at angle theta.
vd_fan(theta; nu=12, du=2.4, dso=10.0, dsd=20.0) = circular_cone_beam(; angles=[theta], dso, dsd, nu, nv=1, du)

vd_p2(a) = Point2f(a[1], a[2])

# the center of pixel (i, j) of an n x n image of unit pixels centered at the origin
pixel_center(i, j, n) = (i - (n + 1) / 2, j - (n + 1) / 2)

# One view of a fan beam and one pixel: the ray from the source through the pixel center to the
# detector, the point where it meets the detector, and the cells colored by the weight of the pixel
# in them (the forward projection of the image that is 1 in this pixel).
function pixel_figure(theta=0.35; n=8, pixel=(6, 3))
  g = vd_fan(theta)
  image = ImageGeom(; dims=(n, n), deltas=(1.0, 1.0))
  x = zeros(n, n)
  x[pixel...] = 1
  w = vec(VoxelDrivenProjector(image, g; eltype=Float64) * x)
  nu = length(w)
  c = pixel_center(pixel..., n)
  s = vd_p2(cell_ray(g, 1.0, 1.0, 1)[1])
  hit = project_point(g, [c[1], c[2], 0.0], 1)
  b = vd_p2(cell_ray(g, hit.cu, 1.0, 1)[2])
  fig = Figure(size=(620, 600))
  ax = Axis(fig[1, 1]; aspect=DataAspect(), limits=((-11.5, 11.5), (-11.5, 11.5)), xlabel="x", ylabel="y")
  for v in -n/2:n/2
    lines!(ax, [v, v], [-n / 2, n / 2]; color=:gray70, linewidth=0.5)
    lines!(ax, [-n / 2, n / 2], [v, v]; color=:gray70, linewidth=0.5)
  end
  poly!(ax, Rect2f(c[1] - 0.5, c[2] - 0.5, 1, 1); color=(:darkorange, 0.5), strokecolor=:darkorange, strokewidth=1.5,
        label="the pixel")
  lines!(ax, [s, b]; color=:black, linestyle=:dot, linewidth=1.5, label="ray through the pixel center")
  cells = [vd_p2(cell_ray(g, iu - 0.5, 1.0, 1)[2]) => vd_p2(cell_ray(g, iu + 0.5, 1.0, 1)[2]) for iu in 1:nu]
  linesegments!(ax, cells; color=[isodd(iu) ? :steelblue : :lightsteelblue for iu in 1:nu], linewidth=4)
  hot = findall(>(0), w)
  linesegments!(ax, cells[hot]; color=w[hot], colormap=:Oranges, colorrange=(0, maximum(w)), linewidth=9)
  for iu in hot
    m = (cells[iu][1] + cells[iu][2]) / 2
    off = normalize(m - s) * 0.9
    text!(ax, m + off; text=string(round(w[iu]; digits=2)), align=(:center, :center), fontsize=13)
  end
  scatter!(ax, [s]; color=:black, markersize=10, label="source")
  scatter!(ax, [b]; color=:black, marker=:xcross, markersize=12, label="projected center")
  Legend(fig[2, 1], ax; orientation=:horizontal, nbanks=2, framevisible=false, labelsize=12)
  fig
end

# Everything the interactive figure draws for view angle theta: the source, the ray through the
# pixel center to the detector, the cells (in alternating colors, the two that get weights in
# orange shades) and the weights at these two cells.
function pixel_scene(theta; n=8, pixel=(6, 3))
  g = vd_fan(theta)
  image = ImageGeom(; dims=(n, n), deltas=(1.0, 1.0))
  x = zeros(n, n)
  x[pixel...] = 1
  w = vec(VoxelDrivenProjector(image, g; eltype=Float64) * x)
  nu = length(w)
  c = pixel_center(pixel..., n)
  s = vd_p2(cell_ray(g, 1.0, 1.0, 1)[1])
  hit = project_point(g, [c[1], c[2], 0.0], 1)
  b = vd_p2(cell_ray(g, hit.cu, 1.0, 1)[2])
  cells = [vd_p2(cell_ray(g, iu - 0.5, 1.0, 1)[2]) => vd_p2(cell_ray(g, iu + 0.5, 1.0, 1)[2]) for iu in 1:nu]
  colors = [w[iu] > 0 ? RGBf(1.0, 1 - 0.85 * w[iu], 1 - w[iu]) : (isodd(iu) ? RGBf(0.27, 0.51, 0.71) : RGBf(0.69, 0.77, 0.87))
            for iu in 1:nu]
  hot = sort(findall(>(0), w); by=iu -> -w[iu])
  hot = length(hot) >= 2 ? hot[1:2] : vcat(hot, hot, [1, 1])[1:2]   # two labels in every state
  labels = [w[iu] > 0 ? string(round(w[iu]; digits=2)) : "" for iu in hot]
  spots = [let m = (cells[iu][1] + cells[iu][2]) / 2; m + normalize(m - s) * 0.9 end for iu in hot]
  (; source=[s], ray=[s => b], hit=[b], cells, colors, labels, spots)
end

# The interactive figure: the slider rotates the scan around the image in steps of 15°, starting at
# theta0 (the view of `weights_figure`); the pixel stays, its projected center moves along the
# detector, and the weights follow.
function pixel_app(; theta0=0.35, nsweep=24, n=8, pixel=(6, 3))
  App() do session
    k = Bonito.Slider(1:nsweep)
    scene = map(i -> pixel_scene(theta0 + 2pi * (i - 1) / nsweep; n, pixel), k.value)
    fig = Figure(size=(560, 560))
    ax = Axis(fig[1, 1]; aspect=DataAspect(), limits=((-11.5, 11.5), (-11.5, 11.5)), xlabel="x", ylabel="y")
    grid = [w for v in -n/2:n/2 for w in (Point2f(v, -n / 2) => Point2f(v, n / 2), Point2f(-n / 2, v) => Point2f(n / 2, v))]
    linesegments!(ax, grid; color=:gray70, linewidth=0.5)
    c = pixel_center(pixel..., n)
    poly!(ax, Rect2f(c[1] - 0.5, c[2] - 0.5, 1, 1); color=(:darkorange, 0.5), strokecolor=:darkorange, strokewidth=1.5)
    linesegments!(ax, map(s -> s.ray, scene); color=:black, linestyle=:dot, linewidth=1.5)
    linesegments!(ax, map(s -> s.cells, scene); color=map(s -> s.colors, scene), linewidth=8)
    text!(ax, map(s -> s.spots, scene); text=map(s -> s.labels, scene), align=(:center, :center), fontsize=13)
    scatter!(ax, map(s -> s.source, scene); color=:black, markersize=10)
    scatter!(ax, map(s -> s.hit, scene); color=:black, marker=:xcross, markersize=12)
    Bonito.record_states(session, DOM.div(DOM.div("view ", k), fig))
  end
end

# The two factors of the weights of the pixel of `pixel_figure` (the first view of `pixel_app`), along the detector: left the
# interpolation (the tent function of each cell, 1 at its center and 0 at the neighboring centers,
# read off at the projected center), right the weights, the interpolation times the scale.
function weights_figure(theta=0.35; n=8, pixel=(6, 3), cells=4:10)
  g = vd_fan(theta)
  image = ImageGeom(; dims=(n, n), deltas=(1.0, 1.0))
  x = zeros(n, n)
  x[pixel...] = 1
  w = vec(VoxelDrivenProjector(image, g; eltype=Float64) * x)
  c = pixel_center(pixel..., n)
  cu = project_point(g, [c[1], c[2], 0.0], 1).cu
  i0 = floor(Int, cu)
  f = (1 - (cu - i0), cu - i0)             # the interpolation factors of cells i0 and i0 + 1
  scale = w[i0] / f[1]
  fig = Figure(size=(1000, 360))
  lim = (first(cells) - 0.5, last(cells) + 0.5)
  ax1 = Axis(fig[1, 1]; title="interpolation", xlabel="position on the detector (cells)", xticks=cells,
             limits=(lim, (-0.25, 1.25)))
  two(v) = string(round(v; digits=2))
  ax2 = Axis(fig[1, 2]; title="weights: interpolation (light) times the scale $(rpad(two(scale), 4, '0')) (dark)",
             xlabel="cell", xticks=cells, limits=(lim, (-0.25, 1.25)))
  for ax in (ax1, ax2)
    hidespines!(ax, :t, :r)
    for iu in cells
      poly!(ax, Rect2f(iu - 0.5, -0.2, 1, 0.1); color=isodd(iu) ? :steelblue : :lightsteelblue)
    end
  end
  for iu in cells
    hot = iu in (i0, i0 + 1)
    lines!(ax1, [iu - 1, iu, iu + 1], [0, 1, 0]; color=hot ? :darkorange : :gray75, linewidth=hot ? 2 : 1)
  end
  vlines!(ax1, [cu]; color=:black, linestyle=:dot)
  scatter!(ax1, [cu], [-0.15]; color=:black, marker=:xcross, markersize=12)
  scatter!(ax1, [cu, cu], collect(f); color=:darkorange, markersize=10)
  for (k, iu) in enumerate((i0, i0 + 1))
    text!(ax1, cu + 0.08, f[k]; text=two(f[k]), align=(:left, :center), fontsize=13)
    barplot!(ax2, [iu], [f[k]]; color=(:darkorange, 0.25), strokecolor=:darkorange, strokewidth=1, width=0.6)
    barplot!(ax2, [iu], [w[iu]]; color=:darkorange, width=0.6)
    text!(ax2, iu + 0.35, w[iu]; text=two(w[iu]), align=(:left, :center), fontsize=13)
  end
  fig
end

# Miao's check: the forward projection of a uniform disk (radius r pixels) in one parallel-beam
# view at angle theta, for cells of du pixels: voxel-driven and distance-driven against the chord
# lengths of the disk at the cell centers.
function disk_projections(du; n=64, r=20.0, theta=0.3)
  image = ImageGeom(; dims=(n, n), deltas=(1.0, 1.0))
  x = Float64.(phantom(axis(image, 1), axis(image, 2), [circle((0.0, 0.0), r, 1.0)]))
  nu = ceil(Int, 1.5n / du)
  g = parallel_beam(; angles=[theta], nu, nv=1, du)
  t = ((1:nu) .- (nu + 1) / 2) .* du
  chord = [abs(s) < r ? 2sqrt(r^2 - s^2) : 0.0 for s in t]
  yv = vec(VoxelDrivenProjector(image, g; eltype=Float64) * x)
  yd = vec(DistanceDrivenProjector(image, g; eltype=Float64) * x)
  (; t, chord, yv, yd)
end

function disk_figure(ratios=(0.25, 0.5, 1.0, 2.0); kwargs...)
  fig = Figure(size=(1000, 560))
  for (j, du) in enumerate(ratios)
    d = disk_projections(du; kwargs...)
    ax = Axis(fig[(j - 1) ÷ 2 + 1, (j - 1) % 2 + 1]; title="cells of $du pixels", xlabel="position on the detector (pixels)",
              ylabel=isodd(j) ? "projection" : "", limits=((-26, 26), (0, 48)))
    lines!(ax, d.t, d.chord; color=:gray50, linewidth=1, label="chord length of the disk")
    scatterlines!(ax, d.t, d.yd; color=:seagreen, markersize=4, linewidth=1, label="distance-driven")
    scatterlines!(ax, d.t, d.yv; color=:darkorange, markersize=4, linewidth=1, label="voxel-driven")
    j == 1 && axislegend(ax; position=:cb, framevisible=false, labelsize=11)
  end
  fig
end

# the relative difference of the voxel-driven and the distance-driven projection of the disk, for the
# text of the page
disk_difference(du; kwargs...) = (d = disk_projections(du; kwargs...); norm(d.yv - d.yd) / norm(d.yd))
