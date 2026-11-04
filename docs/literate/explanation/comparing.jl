# # Comparing Projectors
#
# How evenly a projector covers the image depends mainly on the ratio of detector cell size to voxel
# size and on the number of rays per cell, less on the projector. This page compares
# [Siddon's Method](@ref) and [Joseph's Method](@ref) with one and with several rays per cell (see
# [Use Several Rays per Cell](@ref)) for cells smaller than, equal to and larger than the pixels of a
# 2D image, with two checks from Miao (2013): the back projection of one view and of all views of
# ones. It compares coverage, not accuracy on real data or speed.

using CTOperators
include(joinpath(pkgdir(CTOperators), "docs", "utils", "comparison_figures.jl")) #hide

models = [("Siddon", SiddonProjector, identity), ("Joseph", JosephProjector, identity),
          ("Siddon, 4 rays", SiddonProjector, g -> MultiRay(g, RegularPoints(4, 1))),
          ("Joseph, 4 rays", JosephProjector, g -> MultiRay(g, RegularPoints(4, 1)))]
ratios = (0.5, 1.0, 2.0);

# ## One view
#
# The back projection of a single parallel-beam view of ones, at an angle of 0.3 radians, shows how
# the rays of a view cover the pixels: every pixel gets the summed weights of the rays through it.
#
# This coverage should be the same for every pixel. If rays covered the detector continuously, then
# adding up the lengths of all rays inside a pixel would slice the pixel into thin parallel strips
# and sum them: the result is the area of the pixel, at any angle. One ray per cell samples this
# integral with the spacing `du` of the cells, so the sum times `du` should be 1, and any other value
# is an error of the sampling. The figure shows this scaled coverage for the 16 × 16 central pixels,
# from 0 (red) over even coverage 1 (white) to 2 (blue); rows without a number of rays have one ray
# per cell. The angle 0.3 avoids rays parallel to the pixel grid, which hide the effect for cells up
# to the size of a pixel:

fig = view_figure(models, ratios) #hide
save("comparison_view.png", fig) #hide
nothing #hide
# ![Back projection of one view](comparison_view.png)

# With cells as large as the pixels or smaller, every method covers the pixels nearly evenly. With
# cells of 2 pixels, one ray per cell misses pixels: Siddon's method leaves many pixels with little or
# no weight, and Joseph's interpolation softens the pattern without removing it. Four rays per cell
# fill the gaps: their spacing is then that of cells of 0.5 pixels.
#
# ## All views
#
# The back projection of all 180 views of ones (the "pixel support", `colsums`), scaled in the same
# way and divided by the number of views, shows what remains after the views are added up. Below each
# panel is the variation, the standard deviation relative to the mean, inside the circle of radius 24
# pixels that is drawn; the color window is ±20 %:

fig = support_figure(models, ratios) #hide
save("comparison_support.png", fig) #hide
nothing #hide
# ![Back projection of all views](comparison_support.png)

# The pattern of the single views does not average out. With cells of 2 pixels, the coverage varies
# by about 10 %, in a cross and a checker pattern around the center; four rays per cell reduce that
# to a fraction of a percent. An iterative reconstruction sees this pattern as part of the model, and
# it shows up as artifacts in the images, see [Use Several Rays per Cell](@ref).
#
# ## How many rays
#
# The variation for cells of 2 pixels and regular points, for several numbers of rays per cell:

coverage(P, k) = 2 .* colsums(P(comparison_image(64), MultiRay(comparison_scan(64, 2.0, range(0, pi, 181)[1:180]), RegularPoints(k, 1)); eltype=Float64)) ./ 180
[(rays=k, Siddon=round(variation(coverage(SiddonProjector, k), 24); digits=2),
  Joseph=round(variation(coverage(JosephProjector, k), 24); digits=2)) for k in (1, 2, 3, 4, 8)]

# More rays help, but not steadily: the number matters relative to the pixels, and an odd number of
# rays can be worse than a smaller even one. A spacing of the rays at the object of at most half a
# voxel covers the image well.
#
# ## What to take from it
#
# * **The ratio matters most.** With cells no larger than the voxels, both methods cover the image
#   well with one ray per cell, and Joseph's method is the more even of the two.
# * **For cells larger than the voxels,** one ray per cell is not enough for either method; several
#   rays per cell are, at the cost of about that many times the computing time (see the
#   [benchmarks](https://github.com/JuliaImageRecon/CTOperators.jl/blob/main/benchmark/README.md)).
#   Voxels no smaller than the cells avoid the problem as well, at the price of a coarser image.
# * **In a cone beam,** the size to compare with the voxels is the cell size at the object: the cell
#   size divided by the magnification.
#
# ## References
#
# - C. Miao, "Comparative studies of different system models for iterative CT image
#   reconstruction", master's thesis, Wake Forest University, 2013.
