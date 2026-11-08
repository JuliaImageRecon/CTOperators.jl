# # Siddon's Method
#
# A projector applies the [System Matrix](@ref) ``A``, computing the weights ``a_{ij}`` whenever it
# needs them. Siddon's method takes the image as voxels, each with a constant value, and a data
# element as the line integral along one ray. The line integral is then a weighted sum: every voxel
# the ray passes through contributes its value times the length of the ray inside it,
#
# ```math
# y_i = \sum_j a_{ij} x_j, \qquad a_{ij} = \text{length of ray } i \text{ in voxel } j.
# ```
#
# Siddon's method computes these lengths exactly.

using CTOperators
include(joinpath(pkgdir(CTOperators), "docs", "utils", "siddon_figures.jl")) #hide
WGLMakie.activate!(px_per_unit=2, scalefactor=1) #hide
CairoMakie.activate!() #hide
include(joinpath(pkgdir(CTOperators), "docs", "utils", "bonito_page.jl")) #hide
bonito_page("siddon") #hide

# ## The idea
#
# Siddon (1985) does not look at the voxels but at the planes between them: three sets of equally
# spaced parallel planes, one per axis. Where a ray crosses these planes is a simple computation per
# axis, and between two consecutive crossings the ray lies inside a single voxel. The intersection
# lengths are the distances between consecutive crossings, the voxels follow from the order of the
# crossings.
#
# Siddon computed all crossings, sorted them and then found the voxel of every piece of the ray between two crossings. Jacobs et
# al. (1998) avoid both: starting from the voxel where the ray enters, they take the nearest next
# crossing of the three axes, which says which axis the ray crosses next, and step that axis's voxel
# index by one. Every step costs a few additions and comparisons, and a ray through a volume of n³
# voxels takes at most about 3n steps.
#
# In the figure a ray crosses a 2D image. Orange circles mark where the ray crosses the lines `x =
# const` between pixels, red diamonds the lines `y = const`; the pixels are colored by the length of
# the ray inside them. The slider rotates and shifts the ray.

siddon_app(8) #hide

# The weights in the figure come from the package itself: the back projection of the value 1 along a
# single ray puts its intersection lengths into the pixels. Rounded to two digits, first index x
# (rows of the printout), second index y:

image = ImageGeom(; dims=(8, 8), deltas=(1.0, 1.0))
ray = RayList([-6.0; -2.0; 0.0;;], [6.0; 3.0; 0.0;;])
weights = reshape(adjoint(SiddonProjector(image, ray; eltype=Float64)) * [1.0], 8, 8)
round.(weights; digits=2)

# ## Properties
#
# - **Exact for voxel images.** The weights are the exact lengths, so the projection of an object
#   made of whole voxels equals its analytic line integrals (the test suite checks this against
#   [ImagePhantoms.jl](https://github.com/JuliaImageRecon/ImagePhantoms.jl)). For smooth objects the
#   error is that of representing the object by voxels.
# - **Exact adjoint.** The back projection uses the same weights, so it is exactly the transpose of
#   the forward projection, as iterative solvers require.
# - **Any geometry.** The method only needs the start and end point of each ray: ray lists, cone and
#   parallel beam and any trajectory work the same way.
#
# ## Coverage of the voxels
#
# Siddon's method is *ray-driven*: it follows individual rays, and the detector cells are points.
# Whether a voxel is hit by many or by few rays then depends on how the rays fall onto the voxel
# grid. The figure shows, for one parallel-beam view at 45°, the summed lengths of all rays in every
# pixel, the coverage (see [System Matrix](@ref)). With cells of 1 pixel the coverage varies between
# about 0.83 and 1.38, in a regular pattern; smaller cells, and so more rays per pixel width, even it
# out here.

fig = coverage_figure(16) #hide
save("coverage.png", fig) #hide
nothing #hide
# ![Coverage of the pixels by the rays of one view](coverage.png)

# Ray-driven back projection tends to Moiré patterns, and an iterative reconstruction can show such
# artifacts (De Man & Basu 2004). More rays than voxels reduce them (finer detector sampling, or several
# rays per cell, see [Use Several Rays per Cell](@ref)), as do projectors that integrate over the
# detector cells, such as distance-driven ones (De Man & Basu 2004; Miao 2013).
#
# ## Implementation
#
# - The ray is traced in voxel coordinates (see [Coordinates and Conventions](@ref)), so voxels
#   of any size, also anisotropic or with flipped axes, need no special cases.
# - The forward projection runs one thread per data element and sums the weighted voxel values. The back
#   projection follows the same rays and adds the weighted value of the ray to every voxel it visits
#   with an atomic addition, because rays of different threads meet in the same voxels.
# - A ray that lies exactly in a plane between two voxels is ambiguous: its line integral could be
#   taken from either side. Siddon's method takes the side that the rounding gives, which may differ
#   between Float32 and Float64 or between devices.
# - How the rays are distributed over the threads of a GPU is not part of the method; see
#   [Workgroups and Ray Order on GPUs](@ref).
#
# ## References
#
# - R. L. Siddon, "Fast calculation of the exact radiological path for a three-dimensional CT
#   array", Med. Phys. 12(2), 252–255, 1985.
# - F. Jacobs, E. Sundermann, B. De Sutter, M. Christiaens, I. Lemahieu, "A fast algorithm to
#   calculate the exact radiological path through a pixel or voxel space", J. Comput. Inf. Technol.
#   6(1), 89–94, 1998.
# - B. De Man, S. Basu, "Distance-driven projection and backprojection in three dimensions", Phys.
#   Med. Biol. 49(11), 2463–2475, 2004.
# - C. Miao, "Comparative studies of different system models for iterative CT image
#   reconstruction", master's thesis, Wake Forest University, 2013.
