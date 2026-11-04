# # Use Several Rays per Cell
#
# A detector cell measures the radiation that falls onto its whole area. The projectors trace one
# ray per cell, through its center. When the cells are large compared to the voxels, the back
# projection of such rays has gaps (Miao 2013, p. 29). This guide shows the effect on a small example
# and how `MultiRay` traces several rays per cell instead. [Several Rays per Cell](@ref) explains how the rays are placed and weighted,
# and [Comparing Projectors](@ref) when it matters.

using CTOperators, ImagePhantoms, RegularizedLeastSquares, LinearOperatorCollection, CairoMakie
using ImageGeoms: axis
include(joinpath(pkgdir(CTOperators), "docs", "utils", "tutorial_figures.jl")) #hide
include(joinpath(pkgdir(CTOperators), "docs", "utils", "multiray_figures.jl")) #hide
include(joinpath(pkgdir(CTOperators), "docs", "utils", "comparison_figures.jl")) #hide

# ## The example
#
# A 2D Shepp-Logan phantom of 128 × 128 pixels and a parallel-beam scan with 180 views (projection
# directions). The detector cells are 2 pixels wide; `u` runs along a detector row, `v` across the
# rows. In a cone beam, the size to compare with the voxels is the cell size at the object, the cell
# size divided by the magnification `dsd / dso`.

n = 128
image = ImageGeom(; dims=(n, n), deltas=(1.0, 1.0))
x_true = Float32.(phantom(axis(image, 1), axis(image, 2), shepp_logan(SheppLoganToft(); fovs=(n, n)), 2))
geometry = parallel_beam(; angles=range(0, pi, 181)[1:180], nu=96, nv=1, du=2.0);

# The data average the line integrals over the width of each cell. They are simulated on a 4 times
# finer image with 16 rays per cell, so that they do not come from any of the models compared below:

fine = ImageGeom(; dims=(4n, 4n), deltas=(0.25, 0.25))
x_fine = Float32.(phantom(axis(fine, 1), axis(fine, 2), shepp_logan(SheppLoganToft(); fovs=(n, n)), 2))
y = JosephProjector(fine, MultiRay(geometry, RegularPoints(16, 1))) * x_fine;

# The reconstructions use 40 iterations of CGNR (conjugate gradients on the normal equations):

function reconstruct(op)
  solver = createLinearSolver(CGNR, op; AHA=normalOperator(op), iterations=40)
  reshape(solve!(solver, vec(y)), n, n)
end;

# ## One ray per cell
#
# Without `MultiRay`, each data element (one number of the projection data: one cell in one view) is
# one ray, through the center of its cell. Positions on the detector are counted in cells: cell 2 has
# its center at position 2 and its edges at 1.5 and 2.5. With cells twice as wide as the pixels, the
# rays of a view miss every other line of pixels (left; gray: pixels, orange: rays, blue: cells):

fig = rays_figure() #hide
save("multiray_rays.png", fig) #hide
nothing #hide
# ![Rays through the cell centers and four rays per cell](multiray_rays.png)

# From view to view the rays shift and miss other pixels, but the misses do not cancel. The column
# sums of the system matrix show it: `colsums` is the back projection of data that are all ones, in
# every pixel the summed weights of all rays, and with even coverage it would be flat.

op = SiddonProjector(image, geometry)
x_center = reconstruct(op);

# ## Several rays per cell
#
# `MultiRay` wraps the geometry and traces several rays per cell, here at the centers of 4 equal
# parts of the cell along `u` (right in the figure above). The value of a cell is the mean of the
# line integrals of its rays; the data keep their size.

op_multi = SiddonProjector(image, MultiRay(geometry, RegularPoints(4, 1)))
x_multi = reconstruct(op_multi);

#-

c, c_multi = colsums(op), colsums(op_multi) #hide
m = sum(c[49:80, 49:80]) / 32^2 #hide
label(name, img) = "$name\nvariation $(round(variation(img, 60); digits=2)) %" #hide
fig = images_figure([(label("1 ray per cell", c), c), (label("4 rays per cell", c_multi), c_multi)]; #hide
                    colorrange=(0.8m, 1.2m)) #hide
save("multiray_colsums.png", fig) #hide
nothing #hide
# ![Column sums with one and with four rays per cell](multiray_colsums.png)

# The titles give the variation, the standard deviation relative to the mean inside a circle of radius
# 60 pixels; the window is ±20 % around the mean. With one ray per cell the column sums show rings and
# a cross around the center, with peaks of about ±30 %. With 4 rays per cell the variation drops by a
# factor of about 30, and the peaks to about ±1 %.
#
# The reconstructions, in the color window 0 to 0.4, which shows the soft tissue and clips the skull,
# with the relative error ``\|x - x_\text{true}\| / \|x_\text{true}\|``:

fig = images_figure([("phantom", x_true), ("1 ray per cell", x_center), ("4 rays per cell", x_multi)]; #hide
                    colorrange=(0, 0.4), truth=x_true) #hide
save("multiray_reconstructions.png", fig) #hide
nothing #hide
# ![Reconstructions with one and with four rays per cell](multiray_reconstructions.png)

# With 4 rays per cell the grainy pattern inside the head and between the ellipses goes away, and the
# error drops from 0.33 to 0.27. The streaks outside the head remain. On this example, 16 rays give
# 0.27 as well, and the gap depends on when the iterations stop: with 10 iterations the errors are
# 0.35 and 0.28, with 200 they are 0.33 and 0.32.
#
# `JosephProjector` takes a `MultiRay` the same way. The rays of a 2D image stay in the image plane,
# so the points are spread along `u` only (`RegularPoints(4, 1)`; `RegularPoints(4)` would be an
# error). For a 3D image, `RegularPoints(4)` places 4 × 4 = 16 rays per cell, and the time grows
# accordingly.
#
# ## Other point strategies
#
# Besides `RegularPoints`, the rays of a cell can end at random offsets, fixed by a seed (default 0):
#
# * `JitteredPoints(4, 1; seed=1)`: each regular point moved to a random position within its part of
#   the cell;
# * `RandomPoints(4; seed=1)`: 4 random points anywhere in the cell;
# * `GaussianPoints(4, 0.3; seed=1)`: 4 random points, normally distributed around the cell center
#   with a standard deviation of 0.3 cells, which can also land in neighboring cells (blur).
#
# `Points(offsets)` takes given offsets, a `2 × K` matrix in cell units, the same in every cell. How
# the strategies compare on the data above:

op_jittered = SiddonProjector(image, MultiRay(geometry, JitteredPoints(4, 1; seed=1)))
op_random = SiddonProjector(image, MultiRay(geometry, RandomPoints(4; seed=1)))
x_jittered, x_random = reconstruct(op_jittered), reconstruct(op_random);

#-

fig = images_figure([("4 regular rays", x_multi), ("4 jittered rays", x_jittered), ("4 random rays", x_random)]; #hide
                    colorrange=(0, 0.4), truth=x_true) #hide
save("multiray_jittered.png", fig) #hide
nothing #hide
# ![Regular, jittered and random rays](multiray_jittered.png)

# On these data, whose cells average evenly over their width, regular points give the smallest error
# of the three, 4 jittered rays lie close to one ray per cell, and 4 random rays above it. This is one
# example; other detectors and data can behave differently.
#
# ## Weights
#
# The rays of a cell can also count differently, e.g. for a detector response that falls off towards
# the cell edges. The weights are normalized per cell:

op_weighted = SiddonProjector(image, MultiRay(geometry, RegularPoints(4, 1); weights=GaussianWeights(0.4)));

# On the data above the error is 0.27, as with uniform weights.
#
# ## Cost and devices
#
# * The time grows with the number of rays per cell: 4 rays take about 4 times as long. Jittered
#   points and Gaussian weights added nothing measurable on a GPU (see the
#   [benchmarks](https://github.com/JuliaImageRecon/CTOperators.jl/blob/main/benchmark/README.md#Several-rays-per-cell-(GPU))).
#   No extra memory is needed.
# * `MultiRay` works on every device (`storage=CuArray` etc.), and the random positions are the same
#   on all of them, in the forward and the back projection, and for `subset(op, views)`.
# * Masks work as on plain geometries: a masked data element skips all its rays.
# * Only detector geometries have cells: a `RayList` cannot be wrapped.
#
# ## References
#
# - C. Miao, "Comparative studies of different system models for iterative CT image
#   reconstruction", master's thesis, Wake Forest University, 2013.
