# # Coordinates and Conventions
#
# This page collects the conventions of CTOperators.jl: in which coordinates image and geometry are
# given, how the angles are measured and how the projection data is laid out. They follow those of
# Jeff Fessler's packages ([Sinograms.jl](https://github.com/JuliaImageRecon/Sinograms.jl),
# [ImagePhantoms.jl](https://github.com/JuliaImageRecon/ImagePhantoms.jl)), so that phantoms and
# their analytic projections fit directly, and geometries of Sinograms.jl convert with
# `projection_geometry` (see [Use Geometries of Sinograms.jl](@ref)).

using CTOperators
include(joinpath(pkgdir(CTOperators), "docs", "utils", "convention_figures.jl")) #hide

# ## World coordinates
#
# Everything lives in one right-handed coordinate system `(x, y, z)`. Lengths have no fixed unit:
# image and geometry only have to use the same one (mm in the examples). For rotating scanners the
# rotation axis is the `z` axis. Angles are in radians.
#
# ## The image
#
# The image is described by an `ImageGeom` of ImageGeoms.jl: `dims` voxels of size `deltas`,
# centered at the origin unless shifted by `offsets` (in voxels). Voxel `i` (counted from 1) has its
# center at
#
# ```math
# x_i = \left((i - 1) - \frac{n - 1}{2} - \text{offset}\right) \Delta
# ```
#
# per dimension; the voxel extends half a voxel to each side. Image arrays are ordered `(x, y, z)`,
# `x` fastest. A 2D `ImageGeom` is one slice at `z = 0` of thickness 1 (length unit); its rays must
# lie in that plane, for example those of a geometry with one detector row (`nv = 1`).
#
# ## Cone beam and its trajectories
#
# A view is given by the position of the source, the center of the flat detector and two vectors
# `u` and `v` from one detector cell to the next, along the rows and the columns of the detector
# ([`ConeBeam`](@ref)). The ray of cell `(iu, iv)` runs from the source to the center of the cell.
#
# [`circular_cone_beam`](@ref) and [`helical_cone_beam`](@ref) place them as in the figure: the
# source at angle ``β`` counterclockwise from the `y` axis, at distance `dso` from the rotation
# axis,
#
# ```math
# s(β) = (-d_\text{so} \sin β,\; d_\text{so} \cos β,\; z),
# ```
#
# with `z = 0` for a circle and `z = z0 + pitch (β − β₁) / 2π` for a helix (`β₁` the first angle;
# `pitch` is a length per turn), and the detector opposite, at distance `dsd` from the source. The
# detector rows (`u`) run along `+x` at ``β = 0`` and rotate with the source; the columns (`v`) run
# along `+z`, out of the page in the figure. Seen from above, for ``β = 30°``:

fig = topview_figure() #hide
save("topview.png", fig) #hide
nothing #hide
# ![One cone-beam view from above](topview.png)

# A curved detector ([`ArcConeBeam`](@ref), `detector=:arc`) lies on a cylinder of radius `dsd`
# about the source, with its axis along `z`; its cells are equally spaced in the fan angle,
# ``γ_i = (i - (n_u + 1)/2 - \text{offset}_u)\, d_u / d_\text{sd}``, and ``γ`` counts in the
# direction of `u`.
#
# Detector offsets `offset_u`, `offset_v` are given in cells. As the offsets of ImageGeoms, they are
# subtracted: the cell positions along `u` are `((0:nu-1) .- (nu - 1)/2 .- offset_u) .* du`, so a
# positive `offset_u` moves the detector by `offset_u` cells towards `−u`.
#
# ## Parallel beam
#
# At angle ``ϕ`` the rays of a [`ParallelBeam`](@ref) run along ``(-\sin ϕ, \cos ϕ, 0)`` and the
# detector rows along ``(\cos ϕ, \sin ϕ, 0)``. (The rays point the opposite way of the central ray
# of a cone-beam view at ``β = ϕ``; for line integrals the direction does not matter.) The detector
# is centered on the rotation axis, its columns along `+z`. These are the coordinates ``(u, v, ϕ)``
# of the analytic projections `radon` of ImagePhantoms.jl.
#
# ## Projection data
#
# Detector geometries produce arrays of size `(nu, nv, nviews)`: `u` fastest, then `v`, then the
# views, one detector image per view. Ray lists produce one value per ray, in the order of the rays
# (or an array of the shape given by `datasize`). The value of a ray is the line integral of the
# image along it, in the length unit of the geometry: image value times length.
#
# ## Segments and lines
#
# Cone-beam rays and the rays of a [`RayList`](@ref) are segments from a start to an end point
# ([`RaySegment`](@ref)): only the part between the two points counts, which matters when a source
# or detector lies inside the volume. Parallel-beam rays are infinite lines ([`RayLine`](@ref)).
#
# ## Voxel coordinates inside the projectors
#
# Internally, the projectors trace rays in continuous voxel coordinates, in which the center of
# voxel `(i, j, k)` has the coordinates `(i, j, k)` and every voxel is a unit cube. Voxel sizes,
# offsets and flipped axes (negative `deltas`) then need no special treatment in the kernels.
