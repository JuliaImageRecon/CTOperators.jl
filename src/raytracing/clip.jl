# Clipping of rays to the volume, in continuous voxel coordinates (see `to_index`): the volume is
# the box [1/2, n + 1/2] per dimension and every voxel a unit cube.

"""
    clip(p, d, dims, ::RaySegment | ::RayLine) -> (amin, amax)

Parameters of the part of the ray `p + α d` (voxel coordinates) inside the volume of size `dims`:
the box [1/2, n + 1/2] per dimension, intersected with α ∈ [0, 1] for segments. The ray misses
the volume if `amin >= amax`.
"""
@inline function clip(p::SVector{3,T}, d::SVector{3,T}, dims, rt) where {T}
  amin, amax = segment_range(rt, T)
  for k in 1:3
    lo, hi = T(1 // 2), T(dims[k]) + T(1 // 2)
    if iszero(d[k])
      # parallel to the faces of this dimension: inside between them or nowhere
      (lo <= p[k] <= hi) || return one(T), zero(T)
    else
      a1, a2 = (lo - p[k]) / d[k], (hi - p[k]) / d[k]
      amin = max(amin, min(a1, a2))
      amax = min(amax, max(a1, a2))
    end
  end
  amin, amax
end

segment_range(::RaySegment, ::Type{T}) where {T} = zero(T), one(T)
segment_range(::RayLine, ::Type{T}) where {T} = typemin(T), typemax(T)
