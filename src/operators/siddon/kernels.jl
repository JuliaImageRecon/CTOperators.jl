# Siddon's method in the incremental form of Jacobs et al. 1998, in continuous voxel coordinates
# (`to_index`): every voxel is a unit cube, voxel v covers [v - 1/2, v + 1/2) per dimension, the
# planes between voxels sit at w + 1/2 for integer w. The ray is p + alpha d with alpha in
# [amin, amax] after clipping; a step in alpha is a length alpha * len in world units
# (len = |b - a|).
#
# Per ray: the first plane crossing after amin on each axis and the voxel of the first segment
# (from the midpoint of that segment, the one float -> int conversion of the ray), then repeatedly
# the nearest crossing: segment length, step that axis's voxel index by +-1. Plane crossings are
# computed from the plane index rather than accumulated, so rounding errors do not add up along the
# ray. Simultaneous crossings give zero-length segments; voxels outside the volume (possible in the
# first or last such segment through rounding) are skipped.
#
# No closures in this code: a closure capturing a variable that is reassigned (the loop state) is
# boxed and the kernel no longer compiles for GPUs (docs/memory/ka-lessons.md of the project).

# What a ray does with the voxels it visits: sum the weighted image values (forward projection)
# or add its weighted value to them (back projection).
struct Gather{A}
  image::A
end

struct Scatter{A,T}
  image::A
  value::T
end

@inline visit(g::Gather, acc, ix, iy, iz, w) = @inbounds muladd(w, g.image[ix, iy, iz], acc)

@inline function visit(s::Scatter, acc, ix, iy, iz, w)
  @inbounds Atomix.@atomic :monotonic s.image[ix, iy, iz] += w * s.value
  acc
end

@kernel inbounds = true function siddon_forward_kernel!(y, @Const(image), grid, geometry)
  i = @index(Global, Linear)
  y[i] = siddon_ray(Gather(image), grid, geometry, i)
end

@kernel inbounds = true function siddon_backward_kernel!(image, @Const(y), grid, geometry)
  i = @index(Global, Linear)
  v = y[i]
  if !iszero(v)
    siddon_ray(Scatter(image, v), grid, geometry, i)
  end
end

# The ray i of the geometry through the grid; returns the accumulated value of the action.
@inline function siddon_ray(action, grid::VoxelGrid{T}, geometry, i) where {T}
  a, b = ray(geometry, i)
  p = to_index(grid, a)
  d = (b - a) ./ grid.deltas
  amin, amax = clip(p, d, grid.dims, raytype(geometry))
  amin < amax || return zero(T)
  siddon_traverse(action, grid.dims, p, d, norm(b - a), amin, amax)
end

# index w of the next plane (at w + 1/2) after the coordinate q in direction s
@inline next_plane(q, s) = s > 0 ? floor(q - oftype(q, 0.5)) + 1 : s < 0 ? ceil(q - oftype(q, 0.5)) - 1 : zero(q)
# parameter of the crossing of the plane w + 1/2 by the ray coordinate p + alpha d
@inline crossing(w, p, d) = iszero(d) ? typemax(d) : (w + oftype(d, 0.5) - p) / d
# the voxel containing the coordinate q
@inline voxel_index(q) = unsafe_trunc(Int, floor(q + oftype(q, 0.5)))
@inline step_index(s) = unsafe_trunc(Int, s)

@inline function siddon_traverse(action, dims, p::SVector{3,T}, d::SVector{3,T}, len, amin, amax) where {T}
  s = sign.(d)
  step = map(step_index, s)
  plane = map(next_plane, p + amin * d, s)
  alpha = map(crossing, plane, p, d)
  # voxel of the first segment, from its midpoint
  amid = (amin + min(minimum(alpha), amax)) / 2
  vox = map(voxel_index, p + amid * d)
  acc = zero(T)
  acur = amin
  # every iteration crosses one plane: at most sum(dims) + 3 crossings (bounds NaN rays too)
  for _ in 1:sum(dims)+3
    k = alpha[1] <= alpha[2] ? (alpha[1] <= alpha[3] ? 1 : 3) : (alpha[2] <= alpha[3] ? 2 : 3)
    anext = min(alpha[k], amax)
    if 1 <= vox[1] <= dims[1] && 1 <= vox[2] <= dims[2] && 1 <= vox[3] <= dims[3]
      acc = visit(action, acc, vox[1], vox[2], vox[3], (anext - acur) * len)
    end
    anext >= amax && break
    acur = anext
    plane = Base.setindex(plane, plane[k] + s[k], k)
    alpha = Base.setindex(alpha, crossing(plane[k], p[k], d[k]), k)
    vox = Base.setindex(vox, vox[k] + step[k], k)
  end
  acc
end
