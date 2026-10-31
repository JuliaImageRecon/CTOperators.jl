# Siddon's method in the incremental form of Jacobs et al. 1998, in continuous voxel coordinates
# (`to_index`): every voxel is a unit cube, voxel v covers [v - 1/2, v + 1/2) per dimension, the
# planes between voxels sit at w + 1/2 for integer w. The ray is p + alpha d with alpha in
# [amin, amax] after clipping; a step in alpha is a length alpha * len in world units
# (len = |b - a|).
#
# Per ray: the voxel of the entry point (clamped into the volume, the one float -> int conversion of
# the ray) and, from it, the next plane on each axis; then repeatedly the nearest crossing: segment
# length, step that axis's voxel index (and the linear index) by +-1. Leaving the volume on the
# stepped axis ends the ray. The planes are derived from the voxel, never computed independently
# of it: in Float32 an entry point within rounding of a plane could otherwise put voxel and plane
# counter one apart, shifting all later voxels of that axis (found by comparing with Float64).
# Plane crossings are computed from the plane index rather than accumulated, so rounding errors do
# not add up along the ray. Simultaneous crossings give zero-length segments.
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

@inline visit(g::Gather, acc, lin, w) = @inbounds muladd(w, g.image[lin], acc)

@inline function visit(s::Scatter, acc, lin, w)
  @inbounds Atomix.@atomic :monotonic s.image[lin] += w * s.value
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

# the voxel containing the voxel coordinate q, clamped to 1:n
@inline voxel_index(q, n) = clamp(unsafe_trunc(Int, floor(q + oftype(q, 0.5))), 1, n)
# index w of the next plane (at w + 1/2) from voxel v in direction s
@inline first_plane(v, s) = s > 0 ? oftype(s, v) : s < 0 ? oftype(s, v - 1) : zero(s)
# The crossing of the plane w + 1/2 by the ray coordinate p + alpha d is at
# alpha = (w + 1/2 - p) / d = w * invd + offset with invd = 1 / d, offset = (1/2 - p) / d: one fused
# multiply-add per step instead of a division on the loop's critical path. Axes the ray is parallel
# to have no crossings (alpha = Inf; their invd is never used).
@inline inverse(d) = iszero(d) ? zero(d) : inv(d)
@inline crossing_offset(p, d) = iszero(d) ? zero(d) : (oftype(d, 0.5) - p) / d
@inline crossing(w, invd, offset, d) = iszero(d) ? typemax(d) : muladd(w, invd, offset)
@inline step_index(s) = unsafe_trunc(Int, s)

@inline function siddon_traverse(action, dims, p::SVector{3,T}, d::SVector{3,T}, len, amin, amax) where {T}
  s = sign.(d)
  step = map(step_index, s)
  vox = map(voxel_index, p + amin * d, SVector(dims))
  plane = map(first_plane, vox, s)
  invd = map(inverse, d)
  offset = map(crossing_offset, p, d)
  alpha = map(crossing, plane, invd, offset, d)
  # The loop state in scalars, one branch per axis: indexing the small vectors with a run-time axis
  # would keep them in memory instead of registers (measured: 9.0 ns per step with SVectors).
  nx, ny, nz = dims
  ix, iy, iz = vox[1], vox[2], vox[3]
  sx, sy, sz = step[1], step[2], step[3]
  wx, wy, wz = plane[1], plane[2], plane[3]
  fx, fy, fz = s[1], s[2], s[3]
  ax, ay, az = alpha[1], alpha[2], alpha[3]
  lx, ly, lz = sx, sy * nx, sz * nx * ny
  lin = ix + (iy - 1) * nx + (iz - 1) * nx * ny
  acc = zero(T)
  acur = amin
  # every iteration crosses one plane: at most sum(dims) + 3 crossings (bounds NaN rays too)
  for _ in 1:nx+ny+nz+3
    if ax <= ay && ax <= az
      anext = min(ax, amax)
      acc = visit(action, acc, lin, (anext - acur) * len)
      (anext >= amax || !(1 <= ix + sx <= nx)) && break
      ix += sx; lin += lx; wx += fx
      ax = muladd(wx, invd[1], offset[1])
    elseif ay <= az
      anext = min(ay, amax)
      acc = visit(action, acc, lin, (anext - acur) * len)
      (anext >= amax || !(1 <= iy + sy <= ny)) && break
      iy += sy; lin += ly; wy += fy
      ay = muladd(wy, invd[2], offset[2])
    else
      anext = min(az, amax)
      acc = visit(action, acc, lin, (anext - acur) * len)
      (anext >= amax || !(1 <= iz + sz <= nz)) && break
      iz += sz; lin += lz; wz += fz
      az = muladd(wz, invd[3], offset[3])
    end
    acur = anext
  end
  acc
end
