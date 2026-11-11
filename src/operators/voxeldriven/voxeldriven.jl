"""
    AbstractVoxelScale

How a [`VoxelDrivenProjector`](@ref) scales the bilinear weights of a voxel in a view: the weight
of voxel `j` in cell `i` is `voxel_scale(scale, geometry, k, grid, hit)` times the bilinear weight
of cell `i` at the point where voxel `j` projects (`hit`, from
[`project_point`](@ref CTOperators.project_point)). Subtypes: [`LineIntegralScale`](@ref) (the
default) and [`UnitScale`](@ref). A new scale is a subtype with a method of
[`voxel_scale`](@ref CTOperators.voxel_scale); it must be usable inside kernels (an `isbits`
struct).
"""
abstract type AbstractVoxelScale end

"""
    LineIntegralScale()

The default scale of a [`VoxelDrivenProjector`](@ref): the forward projection approximates the
mean of the line integrals over each detector cell, the detector model of the distance-driven
method. A voxel of volume `V` projecting with density `ρ` (see
[`project_point`](@ref CTOperators.project_point)) gets the scale `V ρ`: its forward projection,
integrated over the detector area, is `V (R / r)² / cos φ`, what the line integrals through a point
mass `V` give (when all its cells lie on the detector). See [Voxel-Driven Method](@ref) for the
derivation.

For a 2D image (one voxel thick in `z`) the line integrals run within the image plane: the scale is
`A (R / r) / (Δu cos φ)` with `A` the pixel area and `Δu` the cell width, and every detector row
whose ray passes within the thickness of the image at the position of the pixel gets the full
weight (no averaging along `v`).
"""
struct LineIntegralScale <: AbstractVoxelScale end

"""
    UnitScale()

A scale of 1 for a [`VoxelDrivenProjector`](@ref): the weights are the bilinear interpolation
weights alone, without the voxel volume, the magnification or the obliquity of the ray. The
forward projection then no longer approximates line integrals; the back projection is plain
interpolation of the data, for methods that apply their own weights.
"""
struct UnitScale <: AbstractVoxelScale end

"""
    voxel_scale(scale::AbstractVoxelScale, geometry, k, grid::VoxelGrid, hit) -> T

The scale of the weights of a voxel in view `k`, where `hit` is the result of
[`project_point`](@ref CTOperators.project_point) for the voxel center. Called in kernels; returns
a number of the element type of `grid`.
"""
function voxel_scale end

@inline voxel_scale(::UnitScale, geometry, k, grid::VoxelGrid{T}, hit) where {T} = one(T)

@inline function voxel_scale(::LineIntegralScale, geometry, k, grid::VoxelGrid{T}, hit) where {T}
  dx, dy, dz = abs.(grid.deltas)
  if grid.dims[3] == 1
    # a slab: (R/r)^2 / (a cos phi) times the cell size along v over the height of the slab's shadow
    dx * dy * hit.density * hit.rowsize / hit.magnification
  else
    dx * dy * dz * hit.density
  end
end

"""
    VoxelDrivenProjector(image::ImageGeom, geometry; scale=LineIntegralScale(), eltype=Float32, storage=Array, launch=nothing, raymask=nothing)

Projector with the voxel-driven method (the pixel-driven method of De Man & Basu 2004): in every
view, the center of each voxel is projected from the source (along the direction for parallel
beams) onto the detector, and the voxel contributes to the 2 × 2 cells around that point with
bilinear weights, times a scale (keyword `scale`, an [`AbstractVoxelScale`](@ref)). The back
projection is the exact adjoint, the way implementations of filtered back projection usually back
project, with the weights of the scale. See [Voxel-Driven Method](@ref) and [System Matrix](@ref).
Compared with the ray-driven projectors:

* the back projection runs one thread per voxel and writes each voxel once, without atomic
  additions, so repeated back projections are equal; every voxel is covered, whatever the cell
  size;
* the forward projection adds into the data with atomic additions, and with cells (at the object)
  smaller than the voxels it shows a pattern at the scale of the cells, the high-frequency
  artifacts of De Man & Basu (2004).

For a 2D image the cells are interpolated along `u` only: every detector row whose ray passes
within the thickness of the image at the position of a pixel gets two cells with the full weight.

Works with detector geometries ([`AbstractDetectorGeometry`](@ref): [`ConeBeam`](@ref),
[`ArcConeBeam`](@ref), [`ParallelBeam`](@ref)), not with ray lists, which have no detector to
project onto, nor with a [`MultiRay`](@ref).

Keywords:

* `scale`: [`LineIntegralScale`](@ref) (the default: the forward projection approximates line
  integrals, like the other projectors), [`UnitScale`](@ref), or a scale of your own.
* `eltype` is the element type of the computation and of the stored geometry.
* `storage` is the array type that selects the device (`Array`, `CuArray`, `ROCArray`, …). Image
  and projection data passed to the operator must live on that device.
* `launch` is a [`LaunchConfig`](@ref) to tune the kernel launches for a device, `nothing` for the
  defaults of the device; only its workgroup size applies (the threads take the voxels in order).
* `raymask` is `nothing` or an array with one entry per data element (detector cell and view), of
  the data size (`raymask[u, v, view]`) or a vector, on any device: the cells that are used
  (`true`). It is stored as `Bool` on the device of the operator. Masked data elements give 0 in
  the forward projection and contribute nothing to the back projection.

The mask of `image` (e.g. `ImageGeom(MaskCircle(); dims, deltas)`, the circular field of view)
selects the voxels: voxels outside count as 0 in the forward projection and get 0 in the back
projection; the threads cover the bounding box of the mask. See [Use Masks](@ref).

# Example

```julia
image = ImageGeom(; dims=(128, 128, 64), deltas=(1, 1, 1))
geometry = circular_cone_beam(; angles=range(0, 2pi, 361)[1:360], dso=500, dsd=1000,
                              nu=256, nv=128, du=1, dv=1)
op = VoxelDrivenProjector(image, geometry)
x = rand(Float32, 128, 128, 64)
y = op * x                 # forward projection, size (256, 128, 360)
z = adjoint(op) * y        # back projection
```

References:
* B. De Man, S. Basu, "Distance-driven projection and backprojection in three dimensions", Phys.
  Med. Biol. 49(11), 2463–2475, 2004 (the pixel-driven method, its weight factors and artifacts).
* W. Zhuang, S. S. Gopal, T. J. Hebert, "Numerical evaluation of methods for computing tomographic
  projections", IEEE Trans. Nucl. Sci. 41(4), 1660–1665, 1994.
"""
struct VoxelDrivenProjector{T,G<:AbstractDetectorGeometry{T},B<:KA.Backend,I<:ImageGeom,L<:Union{Nothing,LaunchConfig},M<:ProjectorMasks,V<:VoxelGrid{T},S<:AbstractVoxelScale} <: AbstractProjector{T,G}
  image::I
  grid::V
  geometry::G
  backend::B
  launch::L
  masks::M
  scale::S
end

function VoxelDrivenProjector(image::ImageGeom, geometry::AbstractDetectorGeometry; scale::AbstractVoxelScale=LineIntegralScale(),
                              eltype::Type{T}=Float32, storage=Array, launch::Union{Nothing,LaunchConfig}=nothing,
                              raymask=nothing) where {T<:AbstractFloat}
  voxel_driven_projector(image, device_geometry(geometry, T, storage), launch, raymask, scale)
end

VoxelDrivenProjector(image::ImageGeom, geometry::MultiRay; kwargs...) =
  throw(ArgumentError("VoxelDrivenProjector projects voxels onto the detector; it takes the geometry of the " *
                      "MultiRay, not the MultiRay"))

VoxelDrivenProjector(image::ImageGeom, geometry::AbstractProjectionGeometry; kwargs...) =
  throw(ArgumentError("VoxelDrivenProjector needs a detector: a ConeBeam, ArcConeBeam or ParallelBeam, " *
                      "not a $(nameof(typeof(geometry)))"))

VoxelDrivenProjector(image::ImageGeom, geometry; kwargs...) =
  VoxelDrivenProjector(image, projection_geometry(geometry); kwargs...)

# from a geometry already in the operator's element type and on its device
function voxel_driven_projector(image::ImageGeom, geometry::AbstractDetectorGeometry{T}, launch, raymask, scale) where {T}
  backend = KA.get_backend(storage_array(geometry))
  VoxelDrivenProjector(image, VoxelGrid{T}(image), geometry, backend, launch, ProjectorMasks(image, geometry, raymask), scale)
end

geometry(op::VoxelDrivenProjector) = op.geometry
image_geometry(op::VoxelDrivenProjector) = op.image
similar_projector(op::VoxelDrivenProjector, geometry::AbstractDetectorGeometry; raymask=ray_mask(op)) =
  voxel_driven_projector(op.image, geometry, op.launch, raymask, op.scale)
projector_masks(op::VoxelDrivenProjector) = op.masks
handles_raymask(::VoxelDrivenProjector) = true
KA.get_backend(op::VoxelDrivenProjector) = op.backend

function forward!(y, op::VoxelDrivenProjector, x)
  fill!(y, zero(eltype(y)))
  launch_voxels!(voxel_driven_forward_kernel!, op, vec(y), vec(x))
  y
end

function backward!(x, op::VoxelDrivenProjector, y)
  fill!(x, zero(eltype(x)))
  launch_voxels!(voxel_driven_backward_kernel!, op, vec(x), vec(y))
  x
end

include("kernels.jl")
