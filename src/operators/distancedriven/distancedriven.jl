"""
    DistanceDrivenProjector(image::ImageGeom, geometry; eltype=Float32, storage=Array, launch=nothing, raymask=nothing)

Projector with the distance-driven method (De Man & Basu 2004), which integrates over the detector
cells instead of following single rays. Per view the image is cut into layers perpendicular to the
principal axis of the view, the axis along which the view's central ray (from the source through the
detector center) advances most. In every layer, the edge rays through the midpoints of the four
edges of a detector cell span a rectangle, the cell's footprint; the cell measures the mean of the
image over the footprint times the length per layer of its ray, summed over the layers. The
footprints of neighboring cells share their edges and cover each layer without gaps. The back
projection is the exact adjoint. See [Distance-Driven Method](@ref) and [System Matrix](@ref).

Works with detector geometries ([`AbstractDetectorGeometry`](@ref): [`ConeBeam`](@ref),
[`ArcConeBeam`](@ref), [`ParallelBeam`](@ref)), not with ray lists, which have no detector cells,
nor with a [`MultiRay`](@ref): the footprint already covers the whole cell. For a 2D image the
footprint extends only within the image plane.

Keywords:

* `eltype` is the element type of the computation and of the stored geometry.
* `storage` is the array type that selects the device (`Array`, `CuArray`, `ROCArray`, …). Image
  and projection data passed to the operator must live on that device.
* `launch` is a [`LaunchConfig`](@ref) to tune the kernel launches for a device, `nothing` for the
  defaults of the device.
* `raymask` is `nothing` or an array with one entry per data element (detector cell and view), of the data size
  (`raymask[u, v, view]`) or a vector, on any device: the cells that are used (`true`). It is stored
  as `Bool` on the device of the operator. Masked data elements give 0 in the forward projection and
  contribute nothing to the back projection.

The mask of `image` (e.g. `ImageGeom(MaskCircle(); dims, deltas)`, the circular field of view)
selects the voxels: voxels outside count as 0 in the forward projection and get 0 in the back
projection, and the layers are limited to the bounding box of the image mask. See [Use Masks](@ref).

# Example

```julia
image = ImageGeom(; dims=(128, 128, 64), deltas=(1, 1, 1))
geometry = circular_cone_beam(; angles=range(0, 2pi, 361)[1:360], dso=500, dsd=1000,
                              nu=256, nv=128, du=1, dv=1)
op = DistanceDrivenProjector(image, geometry)
x = rand(Float32, 128, 128, 64)
y = op * x                 # forward projection, size (256, 128, 360)
z = adjoint(op) * y        # back projection
```

References:
* B. De Man, S. Basu, "Distance-driven projection and backprojection in three dimensions", Phys.
  Med. Biol. 49(11), 2463–2475, 2004.
* R. Liu, L. Fu, B. De Man, H. Yu, "GPU-based branchless distance-driven projection and
  backprojection", IEEE Trans. Comput. Imaging 3(4), 617–632, 2017.
"""
struct DistanceDrivenProjector{T,G<:AbstractDetectorGeometry{T},B<:KA.Backend,I<:ImageGeom,L<:Union{Nothing,LaunchConfig},M<:ProjectorMasks,V<:VoxelGrid{T}} <: AbstractProjector{T,G}
  image::I
  grid::V
  geometry::G
  backend::B
  launch::L
  masks::M
end

function DistanceDrivenProjector(image::ImageGeom, geometry::AbstractDetectorGeometry; eltype::Type{T}=Float32,
                                 storage=Array, launch::Union{Nothing,LaunchConfig}=nothing, raymask=nothing) where {T<:AbstractFloat}
  distance_driven_projector(image, device_geometry(geometry, T, storage), launch, raymask)
end

DistanceDrivenProjector(image::ImageGeom, geometry::MultiRay; kwargs...) =
  throw(ArgumentError("DistanceDrivenProjector integrates over the whole detector cell; it takes the " *
                      "geometry of the MultiRay, not the MultiRay"))

DistanceDrivenProjector(image::ImageGeom, geometry::AbstractProjectionGeometry; kwargs...) =
  throw(ArgumentError("DistanceDrivenProjector needs detector cells: a ConeBeam, ArcConeBeam or ParallelBeam, " *
                      "not a $(nameof(typeof(geometry)))"))

DistanceDrivenProjector(image::ImageGeom, geometry; kwargs...) =
  DistanceDrivenProjector(image, projection_geometry(geometry); kwargs...)

# from a geometry already in the operator's element type and on its device
function distance_driven_projector(image::ImageGeom, geometry::AbstractDetectorGeometry{T}, launch, raymask) where {T}
  backend = KA.get_backend(storage_array(geometry))
  DistanceDrivenProjector(image, VoxelGrid{T}(image), geometry, backend, launch, ProjectorMasks(image, geometry, raymask))
end

geometry(op::DistanceDrivenProjector) = op.geometry
image_geometry(op::DistanceDrivenProjector) = op.image
similar_projector(op::DistanceDrivenProjector, geometry::AbstractDetectorGeometry; raymask=ray_mask(op)) =
  distance_driven_projector(op.image, geometry, op.launch, raymask)
projector_masks(op::DistanceDrivenProjector) = op.masks
handles_raymask(::DistanceDrivenProjector) = true
KA.get_backend(op::DistanceDrivenProjector) = op.backend

forward!(y, op::DistanceDrivenProjector, x) = launch_rays!(distance_driven_forward_kernel!, op, y, vec(x))

function backward!(x, op::DistanceDrivenProjector, y)
  fill!(x, zero(eltype(x)))
  launch_rays!(distance_driven_backward_kernel!, op, vec(x), y)
  x
end

include("kernels.jl")
