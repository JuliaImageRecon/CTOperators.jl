# Static figures of the geometry how-to (CairoMakie). Included invisibly by the page, which loads
# CairoMakie.

include(joinpath(@__DIR__, "scan_drawing.jl"))
using ImageGeoms: axis

# the edges of the box covered by the voxels of an ImageGeom (2D: thickness 1 at z = 0)
function draw_volume!(ax, image; color=:gray40)
  ext = [extrema(axis(image, d)) .+ (-abs(image.deltas[d]) / 2, abs(image.deltas[d]) / 2)
         for d in 1:length(image.dims)]
  length(ext) == 2 && push!(ext, (-0.5, 0.5))
  corners = [Point3f(a, b, c) for a in ext[1], b in ext[2], c in ext[3]]
  edges = [corners[i] => corners[j] for i in eachindex(corners), j in eachindex(corners)
           if i < j && count(corners[i] .!= corners[j]) == 1]
  linesegments!(ax, edges; color)
end

scene_axis(fig, pos; title="", azimuth=1.2pi, elevation=0.12pi) =
  Axis3(fig[pos...]; title, aspect=:data, azimuth, elevation, xlabel="x", ylabel="y", zlabel="z",
        protrusions=(30, 10, 10, 20))

# one view of a cone-beam geometry: source, cone, some rays, the detector with u and v, the volume,
# and the source positions of all views
function draw_cone_view!(ax, g::ConeBeam, image; view=1, path=true)
  draw_volume!(ax, image)
  path && lines!(ax, Point3f.(eachcol(g.source)); color=(:orange, 0.4))
  linesegments!(ax, ray_segments(g, view; nu=5, nv=3); color=(:orange, 0.5))
  corners = detector_corners(g, view)
  mesh!(ax, corners, [1 2 3; 1 3 4]; color=(:steelblue, 0.35), transparency=true)
  linesegments!(ax, cone_segments(g, view); color=:steelblue)
  scatter!(ax, [Point3f(g.source[:, view])]; color=:orange, markersize=12)
  # u and v from the first cell (corner)
  first_cell = Point3f(corners[1])
  for (vec, name) in ((g.u[:, view], "u"), (g.v[:, view], "v"))
    tip = first_cell + Point3f(60 * normalize(vec))
    lines!(ax, [first_cell, tip]; color=:black, linewidth=2)
    text!(ax, tip; text=name, fontsize=16, align=(:left, :bottom))
  end
  text!(ax, Point3f(g.source[:, view]); text="source", align=(:right, :bottom), offset=(-6, 6))
end

function cone_beam_figure(circular, helix, image)
  fig = Figure(size=(900, 380))
  ax1 = scene_axis(fig, (1, 1); title="circular cone beam, view 1")
  draw_cone_view!(ax1, circular, image)
  ax2 = scene_axis(fig, (1, 2); title="helical cone beam, view 1")
  draw_cone_view!(ax2, helix, image)
  fig
end

# one view of a parallel-beam geometry: rays as long lines through the volume, the detector plane
function parallel_beam_figure(g::ParallelBeam, image; view=1, length=120)
  fig = Figure(size=(600, 460))
  ax = scene_axis(fig, (1, 1); title="parallel beam, view 1")
  draw_volume!(ax, image)
  c, u, v, d = (g.center[:, view], g.u[:, view], g.v[:, view], g.direction[:, view])
  cell(iu, iv) = c + (iu - (g.nu + 1) / 2) * u + (iv - (g.nv + 1) / 2) * v
  rays = [Point3f(cell(iu, iv) - length * d) => Point3f(cell(iu, iv) + length * d)
          for iu in round.(Int, range(1, g.nu, 6)), iv in round.(Int, range(1, g.nv, 3))]
  linesegments!(ax, vec(rays); color=(:orange, 0.6))
  corners = [Point3f(c + su * g.nu / 2 * u + sv * g.nv / 2 * v) for (su, sv) in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
  mesh!(ax, corners, [1 2 3; 1 3 4]; color=(:steelblue, 0.35), transparency=true)
  tip = Point3f(c + length / 2 * d)
  lines!(ax, [Point3f(c), tip]; color=:black, linewidth=2)
  text!(ax, tip; text="direction", align=(:left, :bottom))
  fig
end

# lines of response of a ray list in the plane z = 0, with the image
function raylist_figure(rl::RayList, image; every=7)
  fig = Figure(size=(460, 460))
  ax = Axis(fig[1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="ray list: every $(every)th line of response")
  idx = 1:every:size(rl.starts, 2)
  linesegments!(ax, [Point2f(rl.starts[1:2, i]) => Point2f(rl.ends[1:2, i]) for i in idx];
                color=(:orange, 0.25))
  ends = unique([Point2f(rl.starts[1:2, i]) for i in axes(rl.starts, 2)])
  scatter!(ax, ends; color=:steelblue, markersize=6)
  ext = [extrema(axis(image, d)) .+ (-abs(image.deltas[d]) / 2, abs(image.deltas[d]) / 2) for d in 1:2]
  poly!(ax, Rect2f(ext[1][1], ext[2][1], ext[1][2] - ext[1][1], ext[2][2] - ext[2][1]);
        color=(:gray, 0.3), strokecolor=:gray40, strokewidth=1)
  fig
end

# source positions of all views seen from above, the views of a subset highlighted
function subset_figure(g::ConeBeam, views)
  fig = Figure(size=(460, 460))
  ax = Axis(fig[1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="source positions, subset highlighted")
  scatter!(ax, g.source[1, :], g.source[2, :]; color=:gray70, markersize=4, label="all views")
  scatter!(ax, g.source[1, views], g.source[2, views]; color=:orange, markersize=9, label="subset")
  axislegend(ax; position=:cc, framevisible=false)
  fig
end
