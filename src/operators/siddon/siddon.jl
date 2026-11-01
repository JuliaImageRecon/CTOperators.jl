"""
    SiddonProjector(image::ImageGeom, geometry; eltype=Float32, storage=Array, launch=nothing)

Projector with Siddon's method: the weight of a voxel for a ray is the exact length of the ray
inside the voxel (Siddon 1985), computed incrementally as in Jacobs et al. 1998. Works with every
[`AbstractProjectionGeometry`](@ref). The back projection is the exact adjoint.

Keywords:

* `eltype` is the element type of the computation and of the stored geometry.
* `storage` is the array type that selects the device (`Array`, `CuArray`, `ROCArray`, …). Image
  and projection data passed to the operator must live on that device.
* `launch` is a [`LaunchConfig`](@ref) to tune the kernel launches for a device, `nothing` for the
  defaults of the device.

# Example

```julia
image = ImageGeom(; dims=(128, 128, 64), deltas=(1, 1, 1))
geometry = circular_cone_beam(; angles=range(0, 2pi, 361)[1:360], dso=500, dsd=1000,
                              nu=256, nv=128, du=1, dv=1)
op = SiddonProjector(image, geometry)
x = rand(Float32, 128, 128, 64)
y = op * x                 # forward projection, size (256, 128, 360)
z = adjoint(op) * y        # back projection
```

References:
* R. L. Siddon, "Fast calculation of the exact radiological path for a three-dimensional CT
  array", Med. Phys. 12(2), 252–255, 1985.
* F. Jacobs, E. Sundermann, B. De Sutter, M. Christiaens, I. Lemahieu, "A fast algorithm to
  calculate the exact radiological path through a pixel or voxel space", J. Comput. Inf.
  Technol. 6(1), 89–94, 1998.
"""
struct SiddonProjector{T,G<:AbstractProjectionGeometry{T},B<:KA.Backend,I<:ImageGeom,L<:Union{Nothing,LaunchConfig}} <: AbstractProjector{T,G}
  image::I
  grid::VoxelGrid{T}
  geometry::G
  backend::B
  launch::L
end

function SiddonProjector(image::ImageGeom, geometry::AbstractProjectionGeometry; eltype::Type{T}=Float32,
                         storage=Array, launch::Union{Nothing,LaunchConfig}=nothing) where {T<:AbstractFloat}
  siddon_projector(image, device_geometry(geometry, T, storage), launch)
end

SiddonProjector(image::ImageGeom, geometry; kwargs...) =
  SiddonProjector(image, projection_geometry(geometry); kwargs...)

# from a geometry already in the operator's element type and on its device
function siddon_projector(image::ImageGeom, geometry::AbstractProjectionGeometry{T}, launch) where {T}
  backend = KA.get_backend(storage_array(geometry))
  SiddonProjector(image, VoxelGrid{T}(image), geometry, backend, launch)
end

geometry(op::SiddonProjector) = op.geometry
image_geometry(op::SiddonProjector) = op.image
similar_projector(op::SiddonProjector, geometry::AbstractProjectionGeometry) = siddon_projector(op.image, geometry, op.launch)
KA.get_backend(op::SiddonProjector) = op.backend

forward!(y, op::SiddonProjector, x) = launch_rays!(siddon_forward_kernel!, op, y, vec(x))

function backward!(x, op::SiddonProjector, y)
  fill!(x, zero(eltype(x)))
  launch_rays!(siddon_backward_kernel!, op, vec(x), y)
  x
end

include("kernels.jl")
