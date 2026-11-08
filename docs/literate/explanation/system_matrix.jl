# # System Matrix
#
# Every projector of this package is a linear map from an image to projection data: a matrix
# ``A`` with
#
# ```math
# y = A x, \qquad y_i = \sum_j a_{ij} x_j.
# ```
#
# This page describes what the rows, columns and entries of this matrix are, what all projectors
# have in common, and the two choices in which they differ. The pages on the methods describe how
# each of them computes the entries.

using CTOperators
include(joinpath(pkgdir(CTOperators), "docs", "utils", "system_matrix_figures.jl")) #hide
WGLMakie.activate!(px_per_unit=2, scalefactor=1) #hide
CairoMakie.activate!() #hide
include(joinpath(pkgdir(CTOperators), "docs", "utils", "bonito_page.jl")) #hide
bonito_page("system_matrix") #hide

# ## Rows, columns and weights
#
# * A **column** belongs to a voxel of the image; ``x_j`` is its value.
# * A **row** belongs to a **data element**, one number of the projection data: one detector cell
#   in one view, or one ray of a [`RayList`](@ref); ``y_i`` is its value.
# * The entry ``a_{ij}``, the **weight** of voxel ``j`` in data element ``i``, says how much the
#   voxel contributes to the data element.
#
# The matrix is not stored. With a 256³ volume and 10⁷ data elements, each touching a few hundred
# voxels, it has billions of nonzero entries, tens of gigabytes. The projectors compute the weights
# of a data element whenever they need them: one thread per data element computes its row, uses it
# and discards it, in every application of the operator.
#
# ## Back projection and adjoint
#
# The **back projection** applies the transpose, ``x = A^T y``: every data element adds its value,
# times its weights, to the voxels of its row. The projectors compute the same rows for it as for the
# forward projection, so the back projection is the **exact adjoint** of the forward projection: for
# all images ``x`` and data ``y``, ``\langle A x, y \rangle = \langle x, A^T y \rangle``, up to
# rounding. Threads of different data elements add into the same voxels, so the additions are
# atomic; on a GPU they run in no fixed order, and the last digits of a back projection can differ
# from call to call.
#
# Two sums of the matrix appear in reconstruction methods:
#
# * the **row sums** ``A \mathbf{1}`` ([`rowsums`](@ref)), per data element the sum of its weights;
#   for Siddon's method the length of the ray inside the image;
# * the **column sums** ``A^T \mathbf{1}`` ([`colsums`](@ref)), per voxel the summed weights of all
#   data elements (the sensitivity image of OS-EM). Scaled so that an even distribution is 1, they
#   are called **coverage** on these pages.
#
# ## Image model and detector model
#
# What a weight means follows from two choices: what the voxel values describe (the image model) and
# what a data element measures (the detector model).
#
# | method | image model | detector model |
# |:-------|:------------|:---------------|
# | [Siddon's Method](@ref) | constant within each voxel | the line integral along one ray, through the center of the cell |
# | [Joseph's Method](@ref) | interpolated linearly between the voxel centers, within each layer (below) | the line integral along one ray, through the center of the cell |
# | [Several Rays per Cell](@ref) | that of the method used | the mean of the line integrals along several rays of the cell |
# | [Distance-Driven Method](@ref) | constant within each voxel, on the center plane of each layer | the mean over the area the cell sees, per layer |
#
# The first two are *ray-driven*: a data element is one ray, a line without width.
#
# ## Layers and the principal axis
#
# Joseph's method and the distance-driven method cut the image into layers:
#
# * The **principal axis** of a ray is the axis along which it advances most, counted in voxels.
# * A **layer** is the set of voxels with the same index along the principal axis: a row or column
#   of pixels in 2D, a plane of voxels in 3D.
# * The **center plane** of a layer runs through the centers of its voxels; in 2D it is a line, the
#   center line.
# * A ray that crosses a layer completely has the same length in every layer, the **length per
#   layer**
#
# ```math
# \frac{\Delta_k}{|\cos \theta_k|},
# ```
#
# where ``\Delta_k`` is the voxel size along the principal axis ``k`` and ``\theta_k`` the angle
# between the ray and that axis. For cubic voxels this angle is at most 45° in 2D and about 54.7° in
# 3D, since the ray advances most along ``k``, so the length per layer is at most ``\sqrt{2}`` (2D)
# or ``\sqrt{3}`` (3D) times the voxel size.
#
# The figure shows one ray through the center of an image of 8 × 8 pixels: the layers along its
# principal axis (shaded alternately), the piece of the ray in each layer (alternating colors) and
# the middle of each piece. The title gives the direction of the ray, its principal axis, the angle
# to it and the length per layer. The slider turns the ray: at 45° and at 135° the principal axis
# changes, and there the length per layer is largest, ``\sqrt{2}``.

principal_axis_app() #hide

# Joseph's method chooses the principal axis per ray, the distance-driven method once per view, for
# all cells of the view. Siddon's method does not use layers: it follows the crossings of the ray
# with the boundaries between all voxels.
