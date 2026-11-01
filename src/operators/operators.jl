"""
    AbstractProjector{T,G<:AbstractProjectionGeometry} <: AbstractCTOperator{T}

Supertype of the projectors of this package: operators between an image on an `ImageGeom` and
projection data of a geometry of type `G` (stored with its arrays on the device).

Interface:

* `geometry(op)::G`, `image_geometry(op)::ImageGeom`
* `similar_projector(op, geometry)`: the same kind of projector, with the same image geometry and
  options, for another geometry (used by `subset` and `adapt`)
* the methods of [`AbstractCTOperator`](@ref)
"""
abstract type AbstractProjector{T,G<:AbstractProjectionGeometry} <: AbstractCTOperator{T} end

"""
    geometry(op::AbstractProjector)

The projection geometry of `op`, with its arrays on the device of the operator.
"""
function geometry end

"""
    image_geometry(op::AbstractProjector)

The `ImageGeom` the operator was built for.
"""
function image_geometry end

"""
    similar_projector(op::AbstractProjector, geometry)

A projector of the same kind and with the same image geometry and options as `op`, for `geometry`.
"""
function similar_projector end

"""
    forward!(y, op::AbstractCTOperator, x)

Forward projection of the image `x` (of size `imagesize(op)`) into `y` (of size `datasize(op)`),
overwriting `y`. Implemented by every operator; users call `mul!(y, op, x)` or `op * x`.
"""
function forward! end

"""
    backward!(x, op::AbstractCTOperator, y)

Back projection (the adjoint) of the projection data `y` into the image `x`, overwriting `x`.
Implemented by every operator; users call `mul!(x, adjoint(op), y)` or `adjoint(op) * y`.
"""
function backward! end

"""
    imagesize(op)

Size of the operator's input: the image for a projector (2D or 3D, as its `ImageGeom`), the
projection data for its adjoint.
"""
function imagesize end

## AbstractCTOperator: interface through forward!, backward!, imagesize, datasize

Base.eltype(::AbstractCTOperator{T}) where {T} = T
Base.size(op::AbstractCTOperator) = (prod(datasize(op)), prod(imagesize(op)))
Base.size(op::AbstractCTOperator, d::Integer) = d <= 2 ? size(op)[d] : 1

"""
    AdjointCTOperator(op)

The adjoint (back projection) of `op`, created by `adjoint(op)`; applying it calls
`backward!(x, op, y)`.
"""
struct AdjointCTOperator{T,O<:AbstractCTOperator{T}} <: AbstractCTOperator{T}
  parent::O
end

LinearAlgebra.adjoint(op::AbstractCTOperator) = AdjointCTOperator(op)
LinearAlgebra.adjoint(a::AdjointCTOperator) = a.parent
LinearAlgebra.transpose(op::AbstractCTOperator) = adjoint(op)   # real operators
Base.parent(a::AdjointCTOperator) = a.parent

imagesize(a::AdjointCTOperator) = datasize(a.parent)
datasize(a::AdjointCTOperator) = imagesize(a.parent)
KA.get_backend(a::AdjointCTOperator) = KA.get_backend(a.parent)
forward!(x, a::AdjointCTOperator, y) = backward!(x, a.parent, y)
backward!(y, a::AdjointCTOperator, x) = forward!(y, a.parent, x)

# Shaped arrays and flat vectors (RegularizedLeastSquares works on vectors) both work.
function LinearAlgebra.mul!(y::AbstractArray, op::AbstractCTOperator, x::AbstractArray)
  check_length(x, imagesize(op), "input")
  check_length(y, datasize(op), "output")
  forward!(reshape(y, datasize(op)), op, reshape(x, imagesize(op)))
  y
end

# 5-argument mul!, y = alpha * op * x + beta * y, through a temporary projection (iterative solvers
# use it only for start values)
function LinearAlgebra.mul!(y::AbstractArray, op::AbstractCTOperator, x::AbstractArray, alpha::Number, beta::Number)
  tmp = op * x
  y .= alpha .* reshape(tmp, size(y)) .+ beta .* y
end

function check_length(a, dims, what)
  length(a) == prod(dims) ||
    throw(DimensionMismatch("$what has $(length(a)) elements, the operator expects $(prod(dims)) $dims"))
end

# op * x: the output on the operator's device; a vector for a vector input, else shaped
function Base.:*(op::AbstractCTOperator, x::AbstractArray)
  y = KA.allocate(KA.get_backend(op), eltype(op), datasize(op))
  mul!(y, op, x)
  x isa AbstractVector ? vec(y) : y
end

"""
    rowsums(op::AbstractCTOperator)

`op * ones`: for projectors the length of each ray inside the image (exactly for Siddon's method,
approximately for Joseph's). A normalization of SART/SIRT-type methods.
"""
function rowsums(op::AbstractCTOperator)
  x = KA.ones(KA.get_backend(op), eltype(op), imagesize(op)...)
  op * x
end

"""
    colsums(op::AbstractCTOperator)

`adjoint(op) * ones`: for projectors the sensitivity image, the summed weights of all rays per
voxel. A normalization of SART/SIRT/OS-EM-type methods.
"""
colsums(op::AbstractCTOperator) = rowsums(adjoint(op))

## AbstractProjector: through the geometry

datasize(op::AbstractProjector) = datasize(geometry(op))
imagesize(op::AbstractProjector) = image_geometry(op).dims
KA.get_backend(op::AbstractProjector) = KA.get_backend(storage_array(geometry(op)))

"""
    subset(op::AbstractProjector, views)

The projector of the views `views` of `op` (detector geometries; a vector or range of view
indices), for ordered-subset methods. It shares the image geometry and options with `op`.
"""
subset(op::AbstractProjector{T,<:AbstractDetectorGeometry}, views) where {T} =
  similar_projector(op, subset(geometry(op), views))

"""
    subset_blocks(op::AbstractProjector, nsubsets; mode=:interleaved) -> (ops, rows)

The projectors of `nsubsets` subsets of the views of `op` (detector geometries) and, for each, the
indices of its data in `vec(op * x)`: the blocks of rows of ordered-subset methods. With
LinearOperatorCollection.jl, `RowBlockOp(subset_blocks(op, 10)...)` stacks them into one operator,
e.g. for the keyword `blocks` of the OS-SART solver of RegularizedLeastSquares.jl:
`createLinearSolver(OSSART, op; blocks=RowBlockOp(subset_blocks(op, 10)...))`.

`mode=:interleaved` takes every `nsubsets`-th view, so that consecutive subsets see the object from
different directions; `:contiguous` takes consecutive views, whose data indices are ranges (no
copies of the data, cheapest on GPUs). Index vectors live on the device of the projector.
"""
function subset_blocks(op::AbstractProjector{T,<:AbstractDetectorGeometry}, nsubsets::Integer;
                       mode::Symbol=:interleaved) where {T}
  nv = nviews(geometry(op))
  n = clamp(nsubsets, 1, nv)
  views = if mode == :interleaved
    [k:n:nv for k in 1:n]
  elseif mode == :contiguous
    [(div((k - 1) * nv, n) + 1):div(k * nv, n) for k in 1:n]
  else
    throw(ArgumentError("mode must be :interleaved or :contiguous, got :$mode"))
  end
  cells = prod(detectorsize(geometry(op)))
  [subset(op, v) for v in views], [data_rows(op, v, cells) for v in views]
end

# the indices of the data of the views v in the flat data of op: a range for consecutive views,
# else an index vector on the device of op
function data_rows(op, v, cells)
  if step(v) == 1
    return ((first(v) - 1) * cells + 1):(last(v) * cells)
  end
  idx = [(k - 1) * cells + j for k in v for j in 1:cells]
  copyto!(KA.allocate(KA.get_backend(op), Int, length(idx)), idx)
end

Adapt.adapt_structure(to, op::AbstractProjector) = similar_projector(op, adapt(to, geometry(op)))

"""
    device_geometry(geometry, T, storage)

`geometry` with its arrays in element type `T` on the device of `storage`, an array type such as
`Array` or `CuArray` (or a concrete one such as `CuArray{Float32}`). Projector constructors call it
with their keywords `eltype` and `storage`.
"""
device_geometry(geometry, ::Type{T}, storage) where {T} =
  adapt(storage isa UnionAll ? storage{T} : storage, geometry)

include("launch.jl")
include("siddon/siddon.jl")
include("joseph/joseph.jl")
