"""
    trace_element(segment, action, grid, geometry, i)

Traces the rays of data element `i` for a ray-driven method: one ray for plain geometries, the rays
of a cell with normalized ray weights for a [`MultiRay`](@ref). `segment(action, grid, a, b,
raytype)`, the function that traces one ray, traces the ray from `a` to `b`, calling
[`CTOperators.visit`](@ref) for every voxel, and returns the accumulated value of the action. The result is the value of data element `i` for a
[`CTOperators.Gather`](@ref). A
projector that traces through this function works on every geometry. Called in kernels.
"""
@inline trace_element(segment::F, action, grid, geometry, i) where {F} =
  segment(action, grid, ray(geometry, i)..., raytype(geometry))

# forward: the weighted mean of the line integrals of the rays
@inline function trace_element(segment::F, action::Gather, grid::VoxelGrid{T}, g::MultiRay, i) where {F,T}
  cell = element_cell(g, i)
  acc, wsum = zero(T), zero(T)
  for k in 1:nsubrays(g)
    a, b, w = cell_subray(g, cell, k)
    acc = muladd(w, segment(action, grid, a, b, raytype(g)), acc)
    wsum += w
  end
  acc / wsum
end

# back: every ray adds its share w / sum(w) of the value; the weights are summed first, so that the
# shares are known before any ray is traced
@inline function trace_element(segment::F, action::Scatter, grid::VoxelGrid{T}, g::MultiRay, i) where {F,T}
  cell = element_cell(g, i)
  wsum = zero(T)
  for k in 1:nsubrays(g)
    wsum += cell_weight(g, cell, k)
  end
  for k in 1:nsubrays(g)
    a, b, w = cell_subray(g, cell, k)
    segment(Scatter(action.image, action.value * (w / wsum)), grid, a, b, raytype(g))
  end
  zero(T)
end
