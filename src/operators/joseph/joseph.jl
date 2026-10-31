"""
    JosephProjector(image::ImageGeom, geometry; eltype=Float32, storage=Array)

Projector with Joseph's method: the image is interpolated linearly between the voxel centres in
every layer of voxels perpendicular to the axis the ray runs most along, and the line integral is
the sum over these layers of the interpolated value times the length of the ray in the layer
(Joseph 1982). Compared with [`SiddonProjector`](@ref) the image model is smooth: the projection
changes continuously with the rays, the rays of a view cover the voxels more evenly (except near
45°, where both methods coincide), and the projection is exact for images that are linear functions
of the position rather than for voxel images. Works with every
[`AbstractProjectionGeometry`](@ref). The back projection is the exact adjoint.

Outside the volume the image is zero; across the ray it fades to zero within one voxel outside the
outermost voxel centres. Segments that start or end inside the volume take the part of the first
and last layer they cover.

`eltype` is the element type of the computation and of the stored geometry; `storage` the array
type that selects the device (`Array`, `CuArray`, `ROCArray`, …). Image and projection data passed
to the operator must live on that device.

# Example

```julia
image = ImageGeom(; dims=(128, 128, 64), deltas=(1, 1, 1))
geometry = circular_cone_beam(; angles=range(0, 2pi, 361)[1:360], dso=500, dsd=1000,
                              nu=256, nv=128, du=1, dv=1)
op = JosephProjector(image, geometry)
y = op * x                 # forward projection, size (256, 128, 360)
z = adjoint(op) * y        # back projection
```

Reference:
* P. M. Joseph, "An improved algorithm for reprojecting rays through pixel images", IEEE Trans.
  Med. Imaging 1(3), 192–196, 1982.
"""
struct JosephProjector{T,G<:AbstractProjectionGeometry{T},B<:KA.Backend,I<:ImageGeom} <: AbstractProjector{T,G}
  image::I
  grid::VoxelGrid{T}
  geometry::G
  backend::B
end

function JosephProjector(image::ImageGeom, geometry::AbstractProjectionGeometry; eltype::Type{T}=Float32,
                         storage=Array) where {T<:AbstractFloat}
  joseph_projector(image, device_geometry(geometry, T, storage))
end

JosephProjector(image::ImageGeom, geometry; kwargs...) =
  JosephProjector(image, projection_geometry(geometry); kwargs...)

# from a geometry already in the operator's element type and on its device
function joseph_projector(image::ImageGeom, geometry::AbstractProjectionGeometry{T}) where {T}
  backend = KA.get_backend(storage_array(geometry))
  JosephProjector{T,typeof(geometry),typeof(backend),typeof(image)}(image, VoxelGrid{T}(image), geometry, backend)
end

geometry(op::JosephProjector) = op.geometry
image_geometry(op::JosephProjector) = op.image
similar_projector(op::JosephProjector, geometry::AbstractProjectionGeometry) = joseph_projector(op.image, geometry)
KA.get_backend(op::JosephProjector) = op.backend

function forward!(y, op::JosephProjector, x)
  n = nrays(op.geometry)
  n == 0 && return y
  joseph_forward_kernel!(op.backend)(y, vec(x), op.grid, op.geometry; ndrange=n)
  y
end

function backward!(x, op::JosephProjector, y)
  fill!(x, zero(eltype(x)))
  n = nrays(op.geometry)
  n == 0 && return x
  joseph_backward_kernel!(op.backend)(vec(x), y, op.grid, op.geometry; ndrange=n)
  x
end

include("kernels.jl")
