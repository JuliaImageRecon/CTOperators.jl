# The rays of data element i for the ray-driven methods: one ray for plain geometries, the weighted
# subrays of a MultiRay. `segment(action, grid, a, b, raytype)` traces one ray from a to b and
# returns the value of the action (Gather: the weighted sum of the image).

@inline trace_element(segment::F, action, grid, geometry, i) where {F} =
  segment(action, grid, ray(geometry, i)..., raytype(geometry))

# forward: the weighted mean of the subrays' line integrals
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

# back: every subray adds its share w / sum(w) of the value; the weights are summed first, so that
# the shares are known before any subray is traced
@inline function trace_element(segment::F, action::Scatter, grid::VoxelGrid{T}, g::MultiRay, i) where {F,T}
  cell = element_cell(g, i)
  wsum = zero(T)
  for k in 1:nsubrays(g)
    wsum += subray_weight(g, cell, k)
  end
  for k in 1:nsubrays(g)
    a, b, w = cell_subray(g, cell, k)
    segment(Scatter(action.image, action.value * (w / wsum)), grid, a, b, raytype(g))
  end
  zero(T)
end
