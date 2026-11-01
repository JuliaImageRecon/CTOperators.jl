# Figures of the Siddon explanation page (WGLMakie through Bonito for the interactive one,
# CairoMakie for the static one). Included invisibly by the page.

using CTOperators, WGLMakie, Bonito, CairoMakie

# A ray of the sweep shown with the slider: angle theta, offset from the center perpendicular to it;
# start and end point far outside the n x n image of unit pixels.
function sweep_ray(k, nsweep, n)
  theta = 0.15 + (pi - 0.3) * (k - 1) / (nsweep - 1)
  offset = 0.3 * n * sin(2 * theta + 0.4)
  d = (cos(theta), sin(theta))
  p = (-offset * d[2], offset * d[1])
  (p[1] - n * d[1], p[2] - n * d[2]), (p[1] + n * d[1], p[2] + n * d[2])
end

# the parameters t in [0, 1] at which the segment a -> b crosses the planes x = w (axis 1) or y = w
# (axis 2) between the pixels, inside the image
function plane_crossings(a, b, n, axis)
  lo, hi = -n / 2, n / 2
  ts = Float64[]
  for w in lo:hi
    da = b[axis] - a[axis]
    iszero(da) && continue
    t = (w - a[axis]) / da
    other = a[3-axis] + t * (b[3-axis] - a[3-axis])
    0 <= t <= 1 && lo <= other <= hi && push!(ts, t)
  end
  [Point2f(a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])) for t in ts]
end

# The weights of one ray (Siddon's by default): the back projection of the value 1 along it
function ray_weights(image, a, b; projector=SiddonProjector)
  rays = RayList([a[1]; a[2]; 0.0;;], [b[1]; b[2]; 0.0;;])
  reshape(adjoint(projector(image, rays; eltype=Float64)) * [1.0], image.dims)
end

function siddon_app(n; nsweep=60)
  image = ImageGeom(; dims=(n, n), deltas=(1.0, 1.0))
  App() do session
    k = Bonito.Slider(1:nsweep)
    ray = map(i -> sweep_ray(i, nsweep, n), k.value)
    fig = Figure(size=(620, 560))
    ax = Axis(fig[1, 1]; aspect=DataAspect(), limits=((-n / 2 - 1, n / 2 + 1), (-n / 2 - 1, n / 2 + 1)),
              xlabel="x", ylabel="y")
    centers = range(-(n - 1) / 2, (n - 1) / 2, n)
    hm = heatmap!(ax, centers, centers, map(r -> ray_weights(image, r...), ray); colormap=:Blues,
                  colorrange=(1e-6, sqrt(2)), lowclip=:white)
    for w in -n/2:n/2
      lines!(ax, [w, w], [-n / 2, n / 2]; color=:gray70, linewidth=0.5)
      lines!(ax, [-n / 2, n / 2], [w, w]; color=:gray70, linewidth=0.5)
    end
    lines!(ax, map(r -> [Point2f(r[1]), Point2f(r[2])], ray); color=:black, linewidth=1.5)
    scatter!(ax, map(r -> plane_crossings(r..., n, 1), ray); color=:orange, markersize=9,
             label="crossings of x planes")
    scatter!(ax, map(r -> plane_crossings(r..., n, 2), ray); color=:crimson, marker=:diamond,
             markersize=9, label="crossings of y planes")
    Colorbar(fig[1, 2], hm; label="length of the ray in the pixel")
    axislegend(ax; position=:lb, framevisible=true)
    Bonito.record_states(session, DOM.div(DOM.div("ray ", k), fig))
  end
end

# How evenly the rays of one parallel-beam view at 45 degrees cover the pixels: Siddon's column sums
# (summed lengths of all rays per pixel, times the cell size; ideally 1 everywhere), for one, two
# and four rays per pixel.
function coverage_figure(n)
  image = ImageGeom(; dims=(n, n), deltas=(1.0, 1.0))
  fig = Figure(size=(900, 330))
  for (j, du) in enumerate((1.0, 0.5, 0.25))
    g = parallel_beam(; angles=[pi / 4], nu=round(Int, 2n / du), nv=1, du)
    c = du .* colsums(SiddonProjector(image, g; eltype=Float64))
    r = round(Int, 1 / du)
    ax = Axis(fig[1, j]; aspect=DataAspect(), title=r == 1 ? "1 ray per pixel" : "$r rays per pixel")
    hidedecorations!(ax)
    hm = heatmap!(ax, c; colormap=:RdBu, colorrange=(0.6, 1.4))
    j == 3 && Colorbar(fig[1, 4], hm; label="coverage")
  end
  fig
end
