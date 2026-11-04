"""
    JosephProjector(image::ImageGeom, geometry; eltype=Float32, storage=Array, launch=nothing, raymask=nothing)

Projector with Joseph's method (Joseph 1982): the image is interpolated bilinearly within each
layer of voxels perpendicular to the axis the ray runs most along (its principal axis), and the
line integral is the sum over the layers of the interpolated value times the length of the ray in
the layer. Compared with [`SiddonProjector`](@ref):

* the image model is smooth, so the projection changes continuously when a ray moves;
* for one parallel view with one ray per voxel, the rays cover the voxels more evenly, except near
  45°, where the two methods give similar weights;
* the projection is exact for images that are linear functions of the position rather than for
  voxel images.

Works with every [`AbstractProjectionGeometry`](@ref). The back projection is the exact adjoint.

Outside the volume the image is zero; across the ray it fades to zero up to half a voxel beyond
the faces of the volume. If a segment starts or ends inside the volume, only the part of the end
layer it actually crosses is counted.

Keywords:

* `eltype` is the element type of the computation and of the stored geometry.
* `storage` is the array type that selects the device (`Array`, `CuArray`, `ROCArray`, …). Image
  and projection data passed to the operator must live on that device.
* `launch` is a [`LaunchConfig`](@ref) to tune the kernel launches for a device, `nothing` for the
  defaults of the device.
* `raymask` is `nothing` or an array with one entry per ray, of the data size
  (`raymask[u, v, view]`) or a vector, on any device: the rays that are used (`true`). It is stored
  as `Bool` on the device of the operator. Masked rays give 0 in the forward projection and
  contribute nothing to the back projection.

The mask of `image` (e.g. `ImageGeom(MaskCircle(); dims, deltas)`, the circular field of view)
selects the voxels: voxels outside count as 0 in the forward projection and get 0 in the back
projection, and the rays only traverse the bounding box of the mask. See [Use Masks](@ref).

# Example

```julia
image = ImageGeom(; dims=(128, 128, 64), deltas=(1, 1, 1))
geometry = circular_cone_beam(; angles=range(0, 2pi, 361)[1:360], dso=500, dsd=1000,
                              nu=256, nv=128, du=1, dv=1)
op = JosephProjector(image, geometry)
x = rand(Float32, 128, 128, 64)
y = op * x                 # forward projection, size (256, 128, 360)
z = adjoint(op) * y        # back projection
```

Reference:
* P. M. Joseph, "An improved algorithm for reprojecting rays through pixel images", IEEE Trans.
  Med. Imaging 1(3), 192–196, 1982.
"""
struct JosephProjector{T,G<:AbstractProjectionGeometry{T},B<:KA.Backend,I<:ImageGeom,L<:Union{Nothing,LaunchConfig},M<:ProjectorMasks} <: AbstractProjector{T,G}
  image::I
  grid::VoxelGrid{T}
  geometry::G
  backend::B
  launch::L
  masks::M
end

function JosephProjector(image::ImageGeom, geometry::AbstractProjectionGeometry; eltype::Type{T}=Float32,
                         storage=Array, launch::Union{Nothing,LaunchConfig}=nothing, raymask=nothing) where {T<:AbstractFloat}
  joseph_projector(image, device_geometry(geometry, T, storage), launch, raymask)
end

JosephProjector(image::ImageGeom, geometry; kwargs...) =
  JosephProjector(image, projection_geometry(geometry); kwargs...)

# from a geometry already in the operator's element type and on its device; for images of one layer
# the rays of a MultiRay stay in the image plane
function joseph_projector(image::ImageGeom, geometry::AbstractProjectionGeometry{T}, launch, raymask) where {T}
  g = one_layer(image) ? in_plane(geometry) : geometry
  backend = KA.get_backend(storage_array(g))
  JosephProjector(image, VoxelGrid{T}(image), g, backend, launch, ProjectorMasks(image, g, raymask))
end

geometry(op::JosephProjector) = op.geometry
image_geometry(op::JosephProjector) = op.image
similar_projector(op::JosephProjector, geometry::AbstractProjectionGeometry; raymask=ray_mask(op)) =
  joseph_projector(op.image, geometry, op.launch, raymask)
projector_masks(op::JosephProjector) = op.masks
handles_raymask(::JosephProjector) = true
KA.get_backend(op::JosephProjector) = op.backend

forward!(y, op::JosephProjector, x) = launch_rays!(joseph_forward_kernel!, op, y, vec(x))

function backward!(x, op::JosephProjector, y)
  fill!(x, zero(eltype(x)))
  launch_rays!(joseph_backward_kernel!, op, vec(x), y)
  x
end

include("kernels.jl")
