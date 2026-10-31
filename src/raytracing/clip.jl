# Clipping of rays to the volume, in continuous voxel coordinates (see `to_index`): the volume is
# the box [1/2, n + 1/2] per dimension and every voxel a unit cube.

"""
    clip(p, d, dims, ::RaySegment | ::RayLine) -> (amin, amax)

Parameters of the part of the ray `p + α d` (voxel coordinates) inside the volume of size `dims`:
the box [1/2, n + 1/2] per dimension, intersected with α ∈ [0, 1] for segments. The ray misses
the volume if `amin >= amax`.
"""
@inline function clip(p::SVector{3,T}, d::SVector{3,T}, dims, rt) where {T}
  half = T(1 // 2)
  clip(p, d, SVector(half, half, half), SVector(map(n -> T(n) + half, dims)), rt)
end

"""
    clip(p, d, lo, hi, ::RaySegment | ::RayLine) -> (amin, amax)

Parameters of the part of the ray `p + α d` inside the box `[lo, hi]`, intersected with α ∈ [0, 1]
for segments.
"""
@inline function clip(p::SVector{3,T}, d::SVector{3,T}, lo::SVector{3,T}, hi::SVector{3,T}, rt) where {T}
  amin, amax = segment_range(rt, T)
  for k in 1:3
    if iszero(d[k])
      # parallel to the faces of this dimension: inside between them or nowhere
      (lo[k] <= p[k] <= hi[k]) || return one(T), zero(T)
    else
      a1, a2 = (lo[k] - p[k]) / d[k], (hi[k] - p[k]) / d[k]
      amin = max(amin, min(a1, a2))
      amax = min(amax, max(a1, a2))
    end
  end
  amin, amax
end

segment_range(::RaySegment, ::Type{T}) where {T} = zero(T), one(T)
segment_range(::RayLine, ::Type{T}) where {T} = typemin(T), typemax(T)
