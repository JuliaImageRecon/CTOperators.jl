# Interactive 3D figures of a cone-beam scan for the documentation (WGLMakie through Bonito,
# exported as static pages; see docs/make.jl). Included invisibly by the examples, so that the pages
# show the construction of the scan, not the plotting code.

using CTOperators, WGLMakie, Bonito
include(joinpath(@__DIR__, "scan_drawing.jl"))
using LinearAlgebra: normalize

# One view: phantom, a fan of rays, the detector colored by the projection of that view.
function single_view_app(g::ConeBeam, params, y, n; view=1, step=2)
  App() do session
    fig = Figure(size=(700, 600))
    ax = LScene(fig[1, 1]; show_axis=false)
    draw_box!(ax, n)
    draw_phantom!(ax, params)
    # faint and thin, so that the projection on the detector behind them stays visible
    linesegments!(ax, ray_segments(g, view); color=(:orange, 0.2), linewidth=1, transparency=true)
    linesegments!(ax, cone_segments(g, view)[5:8]; color=:steelblue, linewidth=2)
    surface!(ax, detector_grid(g, view, step)...; color=y[1:step:end, 1:step:end, view], colormap=:grays)
    scatter!(ax, [Point3f(g.source[:, view])]; color=:orange, markersize=12)
    DOM.div(fig)
  end
end

# The detector of a ConeBeam view as a rigid transformation of a fixed surface: the surface lies in
# the local frame (cells along x and z, normal along y); per view it is rotated into the frame
# (u, v x u, v) and moved to the detector center. Only this transformation and the image change
# from view to view, which keeps the recorded slider states small.
function detector_frame(g::ConeBeam, k)
  u, v = normalize(g.u[:, k]), normalize(g.v[:, k])
  w = cross3(v, u)
  rotation_quaternion(hcat(u, w, v)), Vec3f(g.center[:, k])
end
cross3(a, b) = [a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1]]

# unit quaternion of the rotation matrix R (Shepperd's method)
function rotation_quaternion(R)
  tr = R[1, 1] + R[2, 2] + R[3, 3]
  if tr > 0
    s = 2sqrt(tr + 1)
    return Makie.Quaternionf((R[3, 2] - R[2, 3]) / s, (R[1, 3] - R[3, 1]) / s, (R[2, 1] - R[1, 2]) / s, s / 4)
  end
  i = argmax((R[1, 1], R[2, 2], R[3, 3]))
  j, k = mod1(i + 1, 3), mod1(i + 2, 3)
  s = 2sqrt(1 + R[i, i] - R[j, j] - R[k, k])
  q = zeros(4)
  q[i] = s / 4
  q[j] = (R[j, i] + R[i, j]) / s
  q[k] = (R[k, i] + R[i, k]) / s
  q[4] = (R[k, j] - R[j, k]) / s
  Makie.Quaternionf(q...)
end

# the detector surface in its local frame, every `step`-th cell
function local_detector(g::ConeBeam, step)
  du, dv = sqrt(sum(abs2, g.u[:, 1])), sqrt(sum(abs2, g.v[:, 1]))
  us = [(iu - (g.nu + 1) / 2) * du for iu in 1:step:g.nu]
  vs = [(iv - (g.nv + 1) / 2) * dv for iv in 1:step:g.nv]
  Float32[u for u in us, v in vs], zeros(Float32, length(us), length(vs)), Float32[v for u in us, v in vs]
end

# The motion: the source on its helix; the slider moves source, cone and detector with its
# projection.
function motion_app(g::ConeBeam, params, y, n; every=6, step=3)
  App() do session
    view = Bonito.Slider(1:every:size(g.source, 2))
    fig = Figure(size=(700, 600))
    ax = LScene(fig[1, 1]; show_axis=false)
    draw_box!(ax, n)
    draw_phantom!(ax, params)
    lines!(ax, Point3f.(eachcol(g.source)); color=(:orange, 0.2), linewidth=2, transparency=true)
    linesegments!(ax, map(k -> cone_segments(g, k), view.value); color=:steelblue, linewidth=2)
    scatter!(ax, map(k -> [Point3f(g.source[:, k])], view.value); color=:orange, markersize=12)
    detector = surface!(ax, local_detector(g, step)...; color=map(k -> y[1:step:end, 1:step:end, k], view.value),
                        colormap=:grays, colorrange=(0, maximum(y)))
    on(view.value; update=true) do k
      q, center = detector_frame(g, k)
      Makie.rotate!(detector, q)
      Makie.translate!(detector, center)
    end
    Bonito.record_states(session, DOM.div(DOM.div("view ", view), fig))
  end
end
