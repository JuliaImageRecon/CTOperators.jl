# Figure of the explanation page "Workgroups and Ray Order on GPUs" (CairoMakie). Included
# invisibly by the page.

using CTOperators, CairoMakie
using CTOperators: TileOrder, DataOrder, ray_index, nthreads

# the workgroup of every detector cell of view 1 when workgroups of `wg` threads take the rays in
# `order`
function workgroup_map(g, order, wg)
  nu, nv = detectorsize(g)
  groups = zeros(Int, nu, nv)
  for j in 1:nthreads(order, g)
    i = ray_index(order, j)
    i == 0 && continue
    c = CartesianIndices(datasize(g))[i]
    c[3] == 1 && (groups[c[1], c[2]] = cld(j, wg))
  end
  groups
end

# The points where the rays of workgroup w cross the plane through the center of the image
# perpendicular to the axis the rays run most along, and the voxels Joseph's method reads there
# (the four neighbors of every point). Unit voxels, the image centered at the origin.
function workgroup_samples(g, order, wg, w)
  points = Point2f[]
  voxels = Set{Tuple{Int,Int}}()
  for j in (w-1)*wg+1:w*wg
    i = ray_index(order, j)
    i == 0 && continue
    a, b = ray(g, i)
    d = b - a
    k = argmax(abs.(d))
    q = a + (-a[k] / d[k]) * d
    o = [m for m in 1:3 if m != k]
    p = (q[o[1]], q[o[2]])
    push!(points, Point2f(p...))
    lb, lc = floor(Int, p[1] - 0.5), floor(Int, p[2] - 0.5)   # lower neighbor: center lb + 1/2
    for db in 0:1, dc in 0:1
      push!(voxels, (lb + db, lc + dc))
    end
  end
  points, voxels
end

function launch_figure(; n=64, nu=64, nv=32)
  g = circular_cone_beam(; angles=[0.0], dso=2n, dsd=4n, nu, nv, du=2.0, dv=2.0)
  orders = (("data order: 64 consecutive cells", DataOrder()), ("tile order: 8 × 8 cells", TileOrder(nu, nv, 8, 8)))
  fig = Figure(size=(1000, 560))
  for (col, (title, order)) in enumerate(orders)
    groups = workgroup_map(g, order, 64)
    w = groups[nu÷2+4, nv÷2+4]          # a workgroup near the center of the detector
    ax = Axis(fig[1, col]; aspect=DataAspect(), title, xlabel="detector cell u", ylabel="detector cell v")
    heatmap!(ax, 1:nu, 1:nv, mod.(groups, 6); colormap=Makie.Categorical(:Pastel1_6))
    heatmap!(ax, 1:nu, 1:nv, replace(Float64.(groups .== w), 0.0 => NaN); colormap=[:black], alpha=0.8)
    points, voxels = workgroup_samples(g, order, 64, w)
    ax2 = Axis(fig[2, col]; aspect=DataAspect(), title="black workgroup: voxels read in the central layer",
               xlabel="x (voxels)", ylabel="z (voxels)", limits=(-36, 36, -3, 11))
    for (vb, vc) in voxels
      poly!(ax2, Rect2f(vb, vc, 1, 1); color=(:steelblue, 0.4), strokecolor=:steelblue, strokewidth=0.5)
    end
    scatter!(ax2, points; color=:black, markersize=4)
  end
  fig
end
