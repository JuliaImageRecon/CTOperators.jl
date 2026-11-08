# Masks: an image mask (the voxels that are reconstructed, from the mask of the `ImageGeom`) and a
# ray mask (the rays that are used, the projector keyword `raymask`); `true` means used, `false`
# masked. A projector with masks applies
#
#   forward:  y = M_rays A M_image x        back:  x = M_image Aᵀ M_rays y
#
# (the exact adjoint), with M the diagonal 0/1 matrices of the masks. The generic layer (`mul!`)
# applies the masks around `forward!`/`backward!`, so every operator gets these semantics; operators
# that apply a mask themselves say so with a trait (`handles_raymask`), and the generic layer leaves
# that mask to them.

"""
    ProjectorMasks(image::ImageGeom, geometry, raymask)

The masks of a projector on the device of `geometry`: the image mask of `image` (`nothing` if all
voxels are used) and the ray mask `raymask` (`nothing`, or a `Bool` array of the data size, `true`
for the rays that are used). For implementations of projectors.
"""
struct ProjectorMasks{I,R}
  image::I
  rays::R
end

const NO_MASKS = ProjectorMasks(nothing, nothing)

function ProjectorMasks(image::ImageGeom, geometry::AbstractProjectionGeometry, raymask)
  backend = KA.get_backend(storage_array(geometry))
  imagemask = all(image.mask) ? nothing : device_mask(backend, image.mask)
  if !isnothing(raymask)
    length(raymask) == nrays(geometry) ||
      throw(DimensionMismatch("raymask has $(length(raymask)) entries for $(nrays(geometry)) rays $(datasize(geometry))"))
    raymask = reshape(device_mask(backend, raymask), datasize(geometry))
  end
  ProjectorMasks(imagemask, raymask)
end

# a Bool array on the device of `backend` with the values of `mask` (from any device)
device_mask(backend, mask) = copyto!(KA.allocate(backend, Bool, size(mask)), Array{Bool}(Array(mask)))

"""
    projector_masks(op::AbstractCTOperator) -> ProjectorMasks

The masks of `op`. Projectors of this package store them. Default for an `AbstractProjector`: the
mask of its `ImageGeom`, copied to the device at every call, and no ray mask. Default for any other
`AbstractCTOperator`: no masks.
"""
projector_masks(op::AbstractCTOperator) = NO_MASKS
projector_masks(op::AbstractProjector) = ProjectorMasks(image_geometry(op), geometry(op), nothing)

"""
    image_mask(op) -> Union{Nothing,AbstractArray{Bool}}
    ray_mask(op) -> Union{Nothing,AbstractArray{Bool}}

The image mask (the voxels that are reconstructed, of the image size) and the ray mask (the rays
that are used, of the data size) of `op` on its device, `nothing` without a mask.
"""
image_mask(op::AbstractCTOperator) = projector_masks(op).image
ray_mask(op::AbstractCTOperator) = projector_masks(op).rays
@doc (@doc image_mask) ray_mask

"""
    handles_raymask(op) -> Bool

Whether `op` applies its ray mask itself (masked rays give 0 in the forward projection and add
nothing in the back projection); otherwise the generic layer multiplies by the mask. Default
`false`; the projectors of this package define it as `true` and skip masked rays in their kernels.
"""
handles_raymask(op::AbstractCTOperator) = false

# the operator with its masks, as `mul!` applies it
apply!(y, op::AbstractCTOperator, x) = masked_forward!(y, op, x)
function apply!(x, a::AdjointCTOperator, y)
  check_accumulation(x)
  masked_backward!(x, parent(a), y)
end

# The back projection adds many contributions into every voxel; in Float16 small ones would be lost.
# Data and the image of the forward projection may be Float16: each of their values is one rounding.
check_accumulation(x) = check_accumulation(eltype(x))
check_accumulation(::Type) = nothing
check_accumulation(::Type{T}) where {T<:AbstractFloat} =
  precision(T) >= precision(Float32) ||
    throw(ArgumentError("the back projection accumulates into the image: use at least Float32, got $T"))

function masked_forward!(y, op, x)
  masks = projector_masks(op)
  forward!(y, op, isnothing(masks.image) ? x : x .* masks.image)
  isnothing(masks.rays) || handles_raymask(op) || (y .*= masks.rays)
  y
end

function masked_backward!(x, op, y)
  masks = projector_masks(op)
  backward!(x, op, isnothing(masks.rays) || handles_raymask(op) ? y : y .* masks.rays)
  isnothing(masks.image) || (x .*= masks.image)
  x
end

"""
    is_masked(raymask, i)

Whether ray `i` is masked, with `raymask` from [`CTOperators.ray_mask`](@ref). Called in kernels;
without a mask (`nothing`) the check compiles away.
"""
@inline is_masked(::Nothing, i) = false
@inline is_masked(mask, i) = @inbounds !mask[i]
