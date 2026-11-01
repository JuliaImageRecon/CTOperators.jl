# # Fan beam with a curved detector
#
# A clinical CT scanner measures a slice with a fan of rays from a point source to a detector row
# curved on an arc about the source. This tutorial sets up such a scan and asks how far the scanner
# has to turn: a full turn, a short scan of half a turn plus the fan angle, or only half a turn.

using CTOperators, ImagePhantoms, RegularizedLeastSquares, CairoMakie
using ImageGeoms: axis
include(joinpath(pkgdir(CTOperators), "docs", "utils", "normal_operator.jl")) #hide
include(joinpath(pkgdir(CTOperators), "docs", "utils", "tutorial_figures.jl")) #hide

# The slice: the Shepp-Logan phantom on 128 × 128 pixels of 1 mm.

n = 128
image = ImageGeom(; dims=(n, n), deltas=(1.0, 1.0))
x_true = Float32.(phantom(axis(image, 1), axis(image, 2), shepp_logan(SheppLoganToft(); fovs=(n, n)), 2));

# ## The scan
#
# The source runs on a circle of radius `dso = 300` mm; the detector row is an arc of radius
# `dsd = 600` mm about the source. Its cells are 1 mm wide along the arc, an angle of `1/600` each.
# The fan has to cover the slice, whose corners are `64√2 ≈ 90.5` mm from the centre: a half fan
# angle of `asin(90.5 / 300)`, 368 cells. A fan beam is a cone beam with a single detector row,
# `nv = 1`; `detector=:arc` makes the row curved.

dso, dsd = 300.0, 600.0
gamma = asin(hypot(n / 2, n / 2) / dso)   # half fan angle
nu = 2 * ceil(Int, gamma * dsd)
fan_beam(angles) = circular_cone_beam(; angles, dso, dsd, nu, nv=1, du=1.0, detector=:arc);

# Three scans with the same angular step, about one degree:
#
# * a **full scan**, one turn;
# * a **short scan** of half a turn plus the fan angle, `π + 2γ`;
# * **half a turn**, `π`.

step = 2pi / 360
full = fan_beam(range(0, 2pi, 361)[1:360])
short = fan_beam(range(0, pi + 2gamma, round(Int, (pi + 2gamma) / step) + 1))
half = fan_beam(range(0, pi, 181))
nviews(full), nviews(short), nviews(half)

# From above: the source positions of all views (grey dots), and for the first view the source
# (orange), some of its rays and the curved detector (blue).

fig = topview_figure([("full scan", full, 1), ("short scan", short, 1), ("half a turn", half, 1)], image; #hide
                     limits=(-320, 320, -330, 320), size=(1050, 400)) #hide
save("fan_scans.png", fig) #hide
nothing #hide
# ![The three scans from above](fan_scans.png)

# ## The data
#
# The sinogram of the full scan. Rays through the same line are measured twice per turn, once from
# each side: the lower and the upper half of the sinogram hold the same line integrals, rearranged.

op_full = JosephProjector(image, full)
y_full = op_full * x_true
fig = sinogram_figure(y_full; angles=range(0, 2pi, 361)[1:360], title="sinogram of the full scan") #hide
save("fan_sinogram.png", fig) #hide
nothing #hide
# ![Sinogram of the full scan](fan_sinogram.png)

# ## How far to turn
#
# A line through the slice is measured when the source passes either of its two ends. In parallel
# beam half a turn sees every line; in a fan beam the rays of one view spread over the fan angle, and
# only half a turn plus the fan angle sees every line at least once. Reconstructions of the three
# scans with 30 CGNR iterations:

function reconstruct(geometry)
  op = JosephProjector(image, geometry)
  y = op * x_true
  reshape(solve!(createLinearSolver(CGNR, op; iterations=30), vec(y)), n, n)
end
x_full, x_short, x_half = reconstruct(full), reconstruct(short), reconstruct(half);

#-

fig = images_figure([("phantom", x_true), ("full scan", x_full), ("short scan", x_short), #hide
                    ("half a turn", x_half)]; #hide
                    colorrange=(0, 0.4), truth=x_true, size=(1150, 330)) #hide
save("fan_reconstructions.png", fig) #hide
nothing #hide
# ![Reconstructions of the three scans](fan_reconstructions.png)

# The short scan is as good as the full one: it measured every line, some twice. Half a turn misses
# lines near the edges of the fan, and the reconstruction shows streaks and wrong values on the side
# that was seen least.
#
# Filtered back projection must weight the doubly measured lines of a short scan by hand (Parker
# weights); an iterative method gets this right by itself, since it only asks for an image that
# matches all measured line integrals.
