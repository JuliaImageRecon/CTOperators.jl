# Figures of the Joseph explanation page. Included invisibly by the page; uses the ray sweep and the
# weights of the Siddon page's figures.

include(joinpath(@__DIR__, "siddon_figures.jl"))

# Joseph's samples of the ray a -> b in the n x n image of unit pixels: its crossings with the
# center lines of the columns (or rows, if it runs more along y), where the image can be nonzero
function joseph_samples(a, b, n)
  d = (b[1] - a[1], b[2] - a[2])
  k = abs(d[1]) >= abs(d[2]) ? 1 : 2
  points = Point2f[]
  for c in range(-(n - 1) / 2, (n - 1) / 2, n)
    t = (c - a[k]) / d[k]
    other = a[3-k] + t * d[3-k]
    abs(other) < (n + 1) / 2 || continue
    push!(points, k == 1 ? Point2f(c, other) : Point2f(other, c))
  end
  points
end

function joseph_app(n; nsweep=60)
  image = ImageGeom(; dims=(n, n), deltas=(1.0, 1.0))
  App() do session
    k = Bonito.Slider(1:nsweep)
    ray = map(i -> sweep_ray(i, nsweep, n), k.value)
    fig = Figure(size=(620, 560))
    ax = Axis(fig[1, 1]; aspect=DataAspect(), limits=((-n / 2 - 1, n / 2 + 1), (-n / 2 - 1, n / 2 + 1)),
              xlabel="x", ylabel="y")
    centers = range(-(n - 1) / 2, (n - 1) / 2, n)
    hm = heatmap!(ax, centers, centers, map(r -> ray_weights(image, r...; projector=JosephProjector), ray);
                  colormap=:Blues, colorrange=(1e-6, sqrt(2)), lowclip=:white)
    for w in -n/2:n/2
      lines!(ax, [w, w], [-n / 2, n / 2]; color=:gray70, linewidth=0.5)
      lines!(ax, [-n / 2, n / 2], [w, w]; color=:gray70, linewidth=0.5)
    end
    scatter!(ax, vec([Point2f(a, b) for a in centers, b in centers]); color=:gray50, markersize=3)
    lines!(ax, map(r -> [Point2f(r[1]), Point2f(r[2])], ray); color=:black, linewidth=1.5)
    scatter!(ax, map(r -> joseph_samples(r..., n), ray); color=:orange, markersize=9,
             label="samples (interpolated)")
    Colorbar(fig[1, 2], hm; label="weight of the pixel")
    axislegend(ax; position=:lb, framevisible=true)
    Bonito.record_states(session, DOM.div(DOM.div("ray ", k), fig))
  end
end

# Coverage of the pixels by the rays of one parallel-beam view, one ray per pixel: Siddon (top) and
# Joseph (bottom), at 45 degrees and at 0.3 rad.
function coverage_comparison(n)
  image = ImageGeom(; dims=(n, n), deltas=(1.0, 1.0))
  fig = Figure(size=(640, 600))
  for (i, P) in enumerate((SiddonProjector, JosephProjector)), (j, angle) in enumerate((pi / 4, 0.3))
    g = parallel_beam(; angles=[angle], nu=2n, nv=1, du=1.0)
    c = colsums(P(image, g; eltype=Float64))
    title = "$(P === SiddonProjector ? "Siddon" : "Joseph"), $(j == 1 ? "45°" : "17°")"
    ax = Axis(fig[i, j]; aspect=DataAspect(), title)
    hidedecorations!(ax)
    hm = heatmap!(ax, c; colormap=:RdBu, colorrange=(0.7, 1.3))
    i == 1 && j == 2 && Colorbar(fig[1:2, 3], hm; label="coverage")
  end
  fig
end
