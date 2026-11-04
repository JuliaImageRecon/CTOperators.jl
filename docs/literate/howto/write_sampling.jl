# # Write a Sampling Strategy
#
# [`MultiRay`](@ref) places several rays in a detector cell according to a point strategy and weights
# them with a weighting strategy. Both are ordinary types, and new ones do not have to be part of
# CTOperators.jl: a script or a package of your own can define them, and they fit in through multiple
# dispatch. This guide writes one of each, and then lets the sampling projector of
# [Write a Projector](@ref) use them.

using CTOperators, KernelAbstractions, LinearAlgebra
using CTOperators: Gather, Scatter, VoxelGrid, clip, to_index, trace_element, visit
include(joinpath(pkgdir(CTOperators), "docs", "literate", "howto", "write_projector.jl")) #hide
include(joinpath(pkgdir(CTOperators), "docs", "utils", "multiray_figures.jl")) #hide

# ## A point strategy
#
# A point strategy is a subtype of [`AbstractPoints`](@ref)`{K}`, with `K` the number of rays per
# cell. It implements [`CTOperators.point_offset`](@ref), which returns the offset of ray `k` from
# the cell center in cell units, where ±1/2 are the cell edges. As an example, `K` points on a circle
# around the cell center, rotated by a random angle in every cell:

struct RingPoints{K,T} <: AbstractPoints{K}
  radius::T
  seed::UInt32
end

RingPoints(K, radius; seed=0) = RingPoints{K,float(typeof(radius))}(radius, UInt32(seed))

@inline function CTOperators.point_offset(p::RingPoints{K}, ::Type{T}, iu, iv, view, k) where {K,T}
  turn, _ = CTOperators.random_pair(T, p.seed, iu, iv, view, 1)
  s, c = sincos(2 * T(pi) * (turn + T(k) / K))
  T(p.radius) * c, T(p.radius) * s
end

# [`CTOperators.random_pair`](@ref) gives random numbers that are fixed by the seed, the cell, the
# view and a number of your choice, here always 1, so that all rays of a cell share the same rotation.
# They are the same in the forward and the back projection and on every device; [Several Rays per
# Cell](@ref) explains why a generator with state does not work here.
#
# `point_offset` runs inside the kernels, so it must not allocate. Since `RingPoints` stores a
# floating-point number, it also says how to convert it, so that it follows the element type of the
# projector (some GPUs have no Float64):

CTOperators.points_eltype(p::RingPoints{K}, ::Type{T}) where {K,T} = RingPoints{K,T}(p.radius, p.seed)

# ## A weighting strategy
#
# A weighting strategy is a subtype of [`AbstractWeights`](@ref) and implements
# [`CTOperators.subray_weight`](@ref), the positive weight of a ray at an offset. The weights of a
# cell are normalized in the kernel. A response that falls off linearly from the cell center:

struct LinearWeights <: AbstractWeights end

CTOperators.subray_weight(::LinearWeights, du, dv) = (1 - abs(du)) * (1 - abs(dv))

# ## Using them
#
# The new types work like those of the package, with every projector that supports `MultiRay`. The
# points in one cell, with the marker area showing the weights:

fig = points_figure([("RingPoints(8, 0.3)", RingPoints(8, 0.3; seed=1), UniformWeights()), #hide
                     ("RingPoints(8, 0.3; seed=2),\nLinearWeights()", RingPoints(8, 0.3; seed=2), LinearWeights())]) #hide
save("ring_points.png", fig) #hide
nothing #hide
# ![Ring points in one cell](ring_points.png)

g = MultiRay(geometry, RingPoints(8, 0.3; seed=1); weights=LinearWeights())
op = JosephProjector(image, g)
x, y = rand(Float32, imagesize(op)), rand(Float32, datasize(op))
dot(op * x, y) / dot(x, adjoint(op) * y)

# ## A projector with several rays per cell
#
# The sampling projector of [Write a Projector](@ref) asks every ray for its end points with `ray`, so
# it has one ray per data element. A `MultiRay` has no such ray. To work on every geometry, a projector
# moves the code for one ray into a function and traces through
# [`CTOperators.trace_element`](@ref), which calls it once for a plain geometry and once per ray of
# the cell for a `MultiRay`.
#
# The function takes the two end points. [`CTOperators.visit`](@ref) does what the two kernels did
# so far: with a [`CTOperators.Gather`](@ref) it sums the voxels, with a
# [`CTOperators.Scatter`](@ref) it adds to them. The option `samples` travels in a small struct,
# since a closure in a kernel could not be compiled for GPUs:

struct SamplingSegment{T}
  samples::T
end

@inline function (s::SamplingSegment)(action, grid::VoxelGrid{T}, a, b, rt) where {T}
  p = to_index(grid, a)
  d = (b - a) ./ grid.deltas
  amin, amax = clip(p, d, grid.dims, rt)
  n = amin < amax ? ceil(Int, s.samples * norm(d) * (amax - amin)) : 0
  dalpha = n > 0 ? (amax - amin) / n : zero(T)
  w = norm(b - a) * dalpha
  acc = zero(T)
  for j in 1:n
    acc = visit(action, acc, nearest_voxel(grid, p + (amin + (j - T(0.5)) * dalpha) * d), w)
  end
  acc
end

# The kernels shrink to one call each:

@kernel function traced_forward!(y, @Const(x), grid, geometry, samples)
  i = @index(Global, Linear)
  y[i] = trace_element(SamplingSegment(samples), Gather(x), grid, geometry, i)
end

@kernel function traced_backward!(x, @Const(y), grid, geometry, samples)
  i = @index(Global, Linear)
  trace_element(SamplingSegment(samples), Scatter(x, y[i]), grid, geometry, i)
end

function CTOperators.forward!(y, op::SamplingProjector, x)
  traced_forward!(op.backend)(y, vec(x), op.grid, op.geometry, op.samples; ndrange=nrays(op.geometry))
  y
end

function CTOperators.backward!(x, op::SamplingProjector, y)
  fill!(x, zero(eltype(x)))
  traced_backward!(op.backend)(vec(x), y, op.grid, op.geometry, op.samples; ndrange=nrays(op.geometry))
  x
end

# The sampling projector now takes every strategy, also the ones above, and its adjoint stays exact:

op = SamplingProjector(image, MultiRay(geometry, RingPoints(8, 0.3; seed=1); weights=LinearWeights()))
dot(op * x, y) / dot(x, adjoint(op) * y)

# On plain geometries it computes what it did before. For 2D images, which have one layer of voxels
# along `z`, a projector's constructor should pass the geometry through
# [`CTOperators.in_plane`](@ref), as the projectors of the package do; it keeps the rays in the image
# plane.
#
# ## What fits where
#
# | to change | subtype | implement |
# |:----------|:--------|:----------|
# | where the rays of a cell end | [`AbstractPoints`](@ref) | [`CTOperators.point_offset`](@ref), [`CTOperators.points_eltype`](@ref) if it stores floats |
# | how they are weighted | [`AbstractWeights`](@ref) | [`CTOperators.subray_weight`](@ref), [`CTOperators.weights_eltype`](@ref) if it stores floats |
# | how a ray meets the voxels | [`AbstractProjector`](@ref) | see [Write a Projector](@ref), tracing through [`CTOperators.trace_element`](@ref) |
#
# These functions are declared `public`: they are not exported, but they are part of the interface.
