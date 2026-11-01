# # Helical cone-beam CT
#
# [Circular cone-beam CT](@ref) showed that a circular orbit cannot measure planes parallel to it away
# from the orbit. A helical scan moves the source along the rotation axis while it turns, so that it
# passes every height of the object. This tutorial reconstructs the same stack of discs from helical
# scans and looks at the pitch, the advance per turn.

using CTOperators, ImagePhantoms, RegularizedLeastSquares, CairoMakie
using ImageGeoms: axis
include(joinpath(pkgdir(CTOperators), "docs", "utils", "normal_operator.jl")) #hide
include(joinpath(pkgdir(CTOperators), "docs", "utils", "tutorial_figures.jl")) #hide

# The object: nine discs, 3 mm thick and 6 mm apart, in a volume of 64³ voxels of 1 mm.

n = 64
image = ImageGeom(; dims=(n, n, n), deltas=(1.0, 1.0, 1.0))
discs = [ellipsoid((0, 0, z), (24, 24, 1.5), (0, 0, 0), 1.0) for z in -24:6:24]
x_true = Float32.(phantom(axis(image, 1), axis(image, 2), axis(image, 3), discs, 2));

# ## The scan
#
# Source and detector as in the circular tutorial (source 128 mm from the axis, flat detector of
# 144 × 96 cells of 2 mm at 256 mm), with 180 views per turn. At the axis the detector covers
# `96 · 2 mm / 2 = 96` mm along `z`. The helix rises by `pitch = 48` mm per turn, half of that, over
# four turns from `z0 = -96` to `z = 96`: the source starts below the volume and ends above it.

function helix(pitch, turns)
  nv = round(Int, 180 * turns)
  helical_cone_beam(; angles=range(0, 2pi * turns, nv + 1)[1:nv], pitch, z0=-pitch * turns / 2,
                    dso=128, dsd=256, nu=144, nv=96, du=2.0)
end
geometry = helix(48, 4);

# The first view, the path of the source (orange helix) and its height over the views:

fig = cone_scan_figure(geometry, image; title="helical cone beam, view 1") #hide
save("helical_scan.png", fig) #hide
nothing #hide
# ![One view of the helical scan](helical_scan.png)

# ## Reconstruction
#
# 20 CGNR iterations, for the helix and, for comparison, for the circular scan:

function reconstruct(geometry)
  op = JosephProjector(image, geometry)
  reshape(solve!(createLinearSolver(CGNR, op; iterations=20), vec(op * x_true)), n, n, n)
end
circle = circular_cone_beam(; angles=range(0, 2pi, 181)[1:180], dso=128, dsd=256, nu=144, nv=96, du=2.0)
x_circle, x_helix = reconstruct(circle), reconstruct(geometry);

#-

fig = slices_figure([("phantom", x_true), ("circular scan", x_circle), ("helical scan", x_helix)]; #hide
                    z=33, x=33, colorrange=(0, 1.2), truth=x_true, size=(840, 560)) #hide
save("helical_reconstruction.png", fig) #hide
nothing #hide
# ![Circular and helical scan](helical_reconstruction.png)

# Every disc is reconstructed: each plane through the object now meets the path of the source.
#
# ## The pitch
#
# A larger pitch scans faster and with less dose per length. With the same views per turn, three
# pitches: half the height the detector covers at the axis (as above), all of it, and twice of it.

pitches = (48, 96, 192)
x_pitch = [x_helix, (reconstruct(helix(p, 192 / p + 1)) for p in pitches[2:end])...];

#-

fig = slices_figure([("pitch $p mm", x) for (p, x) in zip(pitches, x_pitch)]; z=33, x=33, #hide
                    colorrange=(0, 1.2), truth=x_true, size=(840, 560)) #hide
save("helical_pitch.png", fig) #hide
nothing #hide
# ![Three pitches](helical_pitch.png)

# A voxel near the axis stays in the cone while the source rises by the detector height: for one
# turn at a pitch of 96 mm, for half a turn at 192 mm. Half a turn is less than the half turn plus
# fan angle that [Fan beam with a curved detector](@ref) found necessary, and the reconstruction
# shows shading and streaks. Clinical scanners use pitches between about half and one and a half
# times the detector height.
#
# An object moving along the rotation axis during a circular scan, on a table or a conveyor belt,
# sees the same helix: only the relative motion counts. [Getting Started](@ref) reconstructs such a
# scan, and [Set up a geometry](@ref) shows how to describe any other motion.
