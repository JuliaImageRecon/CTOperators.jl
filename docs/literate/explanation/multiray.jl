# # Several Rays per Cell
#
# This page explains what a projector computes on a `MultiRay` geometry, how the rays are placed and
# weighted, where the random positions come from, and how it is implemented.
# [Use Several Rays per Cell](@ref) shows how to use it.

using CTOperators
include(joinpath(pkgdir(CTOperators), "docs", "utils", "multiray_figures.jl")) #hide

# ## What is computed
#
# A ray-driven projector such as [Siddon's Method](@ref) or [Joseph's Method](@ref) treats a
# detector cell as a point: one line from the source to the center of the cell. When the cells are
# large compared to the voxels, the lines miss voxels between them, and the back projection covers
# the voxels unevenly (De Man & Basu 2004). A real cell integrates over its area. `MultiRay(geometry, points; weights)` replaces the one line by `K` rays
# per cell, ending at the positions given by `points`, and averages their line integrals:
#
# ```math
# y_i = \sum_{k=1}^K \frac{w_{ik}}{\sum_l w_{il}} \int_{\text{ray } (i, k)} x
# ```
#
# The back projection is the exact adjoint: every ray adds its share ``w_{ik} / \sum_l w_{il}`` of
# ``y_i``. The rays of a cell approximate the integral over the cell, in the classification of
# system models by Miao (2013) the "area integral" model, by sampling.
#
# The weights average **line integrals**, which keeps the operator linear. A detector averages
# intensities, ``\exp(-\int x)``, which for a cell that sees an edge of the object is not the same
# (the partial-volume effect); modeling that would make the operator nonlinear.
#
# ## Positions on the detector
#
# Positions are counted in cells: cell `(iu, iv)` has its center at `(iu, iv)` and its edges at
# half-integers. Every detector geometry knows the ray that ends at any position
# (`CTOperators.cell_ray`): on a flat detector a point on the plane, on an arc detector a point on
# the cylinder around the source at the angle that belongs to the position, on a parallel-beam
# detector a line through that point. So the rays of a cell sample its actual shape, also on curved
# detectors.
#
# The point strategies give the offsets of the rays from the cell center, between -1/2 and 1/2;
# the weighting strategies give a weight for every offset. Marker areas show the weights:

fig = points_figure([("RegularPoints(3, 3)", RegularPoints(3, 3), UniformWeights()), #hide
                     ("JitteredPoints(3, 3)", JitteredPoints(3, 3; seed=1), UniformWeights()), #hide
                     ("RandomPoints(9)", RandomPoints(9; seed=1), UniformWeights()), #hide
                     ("RegularPoints(3, 3),\nGaussianWeights(0.3)", RegularPoints(3, 3), GaussianWeights(0.3))]) #hide
save("multiray_points.png", fig) #hide
nothing #hide
# ![Point strategies in one cell](multiray_points.png)

# * **Regular points** sit at the centers of `su × sv` subcells: the midpoint rule for the integral
#   over the cell. For data that are cell averages they are the most accurate with few points.
# * **Jittered points** move each regular point to a random position within its subcell ("jittered
#   sampling", Cook 1986, also called stratified sampling). They keep the even spread, but neighboring
#   cells no longer share the same pattern, so regular structures of the rays do not line up with
#   the voxel grid.
# * **Random points** are spread anywhere in the cell. With few points their mean can differ
#   noticeably from the cell average.
# * **Given points** (`Points`) are the same in every cell, also outside it.
# * **Gaussian weights** model a detector response that falls off towards the edges of a cell.
#
# For 2D images, which have one layer of voxels along `z`, the projectors keep all rays in the image
# plane: the offsets along `v` are set to 0, and strategies with several subcells along `v` are
# rejected.
#
# ## Random positions without a random number generator
#
# The random positions must be the same in the forward and in the back projection, in every call
# and on every device: otherwise the back projection would not be the adjoint of the forward
# projection, and an iterative solver would see a different matrix in every iteration. A random
# number generator with state cannot give that in parallel kernels. Instead the positions are
# computed from the indices with a hash function, as GPU renderers do: `pcg4d` of Jarzynski and
# Olano (2020) maps four numbers, here the detector cell, the view, the number of the ray in the cell
# and the seed, to four numbers that look independent even for neighboring inputs; two of them
# give the offsets.
#
# The view is numbered as in the full scan: `subset(op, views)` keeps these numbers, so the rays of
# a subset are those of the same views of the full operator, as ordered-subset methods require. A
# different seed gives a different, again fixed, set of positions: `MultiRay(geometry,
# JitteredPoints(4; seed=2))`.
#
# ## Implementation
#
# * The number of rays per cell `K` is a type parameter of the point strategy. The loop over the
#   rays of a cell has a length known to the compiler, and with `K = 1` the projector is as fast as
#   on the plain geometry.
# * The kernels are those of the plain projectors: each thread handles one data element. For a
#   `MultiRay` it traces the `K` rays one after the other and adds them up (forward), or first sums
#   the weights and then lets every ray add its share (back). No memory per ray is needed.
# * The time is about `K` times that of one ray; computing the positions and weights is negligible
#   (see the [benchmarks](https://github.com/JuliaImageRecon/CTOperators.jl/blob/main/benchmark/README.md)).
# * The ray mask applies per data element (a masked element skips all its rays), the clipping to the
#   bounding box of the image mask per ray.
# * A `MultiRay` has no single ray per data element. A projector that traces only one ray per cell
#   does not accept it, instead of silently ignoring the extra rays.
#
# ## References
#
# - R. L. Cook, "Stochastic sampling in computer graphics", ACM Transactions on Graphics 5(1),
#   51–72, 1986.
# - M. Jarzynski, M. Olano, "Hash functions for GPU rendering", Journal of Computer Graphics
#   Techniques 9(3), 21–38, 2020.
# - C. Miao, "Comparative studies of different system models for iterative CT image
#   reconstruction", master's thesis, Wake Forest University, 2013.
# - B. De Man, S. Basu, "Distance-driven projection and backprojection in three dimensions", Phys.
#   Med. Biol. 49(11), 2463–2475, 2004.
