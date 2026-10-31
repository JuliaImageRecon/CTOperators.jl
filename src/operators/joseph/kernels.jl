# Joseph's method (Joseph 1982) in three dimensions, in continuous voxel coordinates (`to_index`):
# voxel centres at integer coordinates, voxel v covers [v - 1/2, v + 1/2) per dimension. The ray is
# p + alpha d, a step in alpha is a length alpha * len in world units (len = |b - a|).
#
# The image is a function that is bilinear between the voxel centres within every layer of voxels
# perpendicular to the principal axis k of the ray (the axis along which d is largest in voxel
# coordinates) and zero outside the volume. The ray crosses one slab [i - 1/2, i + 1/2) per layer i
# along k; it contributes the length of the ray inside the slab times the interpolated image at the
# middle of that part. A ray through the whole slab has the length len / |d_k| in it and its middle
# lies in the centre plane of the layer: Joseph's sum. Only the first and last slab of a segment
# that starts or ends inside the volume are partial, which is Joseph's treatment of the end points
# proportional to the length of the ray in them.
#
# Interpolation needs no plane crossings: the slabs are counted along k, the transverse coordinates
# follow from the slab index. The neighbours of the interpolation outside the volume count zero, so
# the image fades to zero within one voxel outside the outermost centres across the ray; the plane
# range is clipped to that support.
#
# No closures in this code (see siddon/kernels.jl).

@kernel inbounds = true function joseph_forward_kernel!(y, @Const(image), grid, geometry)
  i = @index(Global, Linear)
  y[i] = joseph_ray(Gather(image), grid, geometry, i)
end

@kernel inbounds = true function joseph_backward_kernel!(image, @Const(y), grid, geometry)
  i = @index(Global, Linear)
  v = y[i]
  if !iszero(v)
    joseph_ray(Scatter(image, v), grid, geometry, i)
  end
end

# The four neighbours of a sample inside the volume, with weights wb0, wb1 across b and 1 - fc, fc
# across c. The forward projection interpolates first and weights once.
@inline function visit_bilinear(g::Gather, acc, lin, sb, sc, wb0, wb1, fc)
  x = g.image
  @inbounds begin
    c0 = muladd(fc, x[lin+sc] - x[lin], x[lin])
    c1 = muladd(fc, x[lin+sb+sc] - x[lin+sb], x[lin+sb])
  end
  muladd(wb1, c1, muladd(wb0, c0, acc))
end

@inline function visit_bilinear(s::Scatter, acc, lin, sb, sc, wb0, wb1, fc)
  gc = one(fc) - fc
  acc = visit(s, acc, lin, wb0 * gc)
  acc = visit(s, acc, lin + sb, wb1 * gc)
  acc = visit(s, acc, lin + sc, wb0 * fc)
  visit(s, acc, lin + sb + sc, wb1 * fc)
end

# the component k of v without indexing with a run-time index (which would put v in memory)
@inline pick(v, k) = ifelse(k == 1, v[1], ifelse(k == 2, v[2], v[3]))

# the axis along which the ray advances most per unit of its parameter (ties: the lower axis)
@inline function principal_axis(d)
  ax, ay, az = abs(d[1]), abs(d[2]), abs(d[3])
  ifelse(ax >= ay && ax >= az, 1, ifelse(ay >= az, 2, 3))
end

# the region where the image can be nonzero for rays with principal axis k: the slabs of the
# layers along k, and up to the next voxel centre outside the volume across k
@inline function joseph_support(dims, k, ::Type{T}) where {T}
  lo = SVector(ntuple(m -> m == k ? T(1 // 2) : zero(T), 3))
  hi = SVector(ntuple(m -> m == k ? T(dims[m]) + T(1 // 2) : T(dims[m] + 1), 3))
  lo, hi
end

# The ray i of the geometry through the grid; returns the accumulated value of the action.
@inline function joseph_ray(action, grid::VoxelGrid{T}, geometry, i) where {T}
  a, b = ray(geometry, i)
  p = to_index(grid, a)
  d = (b - a) ./ grid.deltas
  k = principal_axis(d)
  iszero(pick(d, k)) && return zero(T)   # a point, not a ray
  lo, hi = joseph_support(grid.dims, k, T)
  amin, amax = clip(p, d, lo, hi, raytype(geometry))
  amin < amax || return zero(T)
  smin, smax = segment_range(raytype(geometry), T)
  joseph_traverse(action, grid.dims, p, d, k, norm(b - a), amin, amax, smin, smax)
end

@inline function joseph_traverse(action, dims, p::SVector{3,T}, d::SVector{3,T}, k, len, amin, amax, smin, smax) where {T}
  # the principal axis a = k and the transverse axes b < c, with their strides in the image
  nx, ny, nz = dims
  strides = (1, nx, nx * ny)
  kb = ifelse(k == 1, 2, 1)
  kc = ifelse(k == 3, 2, 3)
  na, nb, nc = pick(dims, k), pick(dims, kb), pick(dims, kc)
  sa, sb, sc = pick(strides, k), pick(strides, kb), pick(strides, kc)
  pa, pb, pc = pick(p, k), pick(p, kb), pick(p, kc)
  da, db, dc = pick(d, k), pick(d, kb), pick(d, kc)

  # the layers the clipped ray crosses, in the direction of the ray
  first = voxel_index(muladd(amin, da, pa), na)
  last = voxel_index(muladd(amax, da, pa), na)
  step = ifelse(da > 0, 1, -1)
  invd = inv(da)
  half = abs(invd) / 2        # half the parameter length of a slab
  one_t = one(T)
  acc = zero(T)
  for t in 0:abs(last - first)
    i = first + t * step
    centre = (i - pa) * invd    # the parameter of the centre plane of layer i
    lo = max(centre - half, smin)
    hi = min(centre + half, smax)
    if lo < hi
      w = (hi - lo) * len
      mid = (lo + hi) / 2
      qb = muladd(mid, db, pb)
      qc = muladd(mid, dc, pc)
      jb = unsafe_trunc(Int, floor(qb))
      jc = unsafe_trunc(Int, floor(qc))
      fb = qb - jb
      fc = qc - jc
      lin = 1 + (i - 1) * sa + (jb - 1) * sb + (jc - 1) * sc
      wb0, wb1 = w * (one_t - fb), w * fb
      if 1 <= jb < nb && 1 <= jc < nc
        # all four neighbours inside the volume: no bounds checks (most samples)
        acc = visit_bilinear(action, acc, lin, sb, sc, wb0, wb1, fc)
      else
        inb0, inb1 = 1 <= jb <= nb, 0 <= jb <= nb - 1
        inc0, inc1 = 1 <= jc <= nc, 0 <= jc <= nc - 1
        if inb0 && inc0
          acc = visit(action, acc, lin, wb0 * (one_t - fc))
        end
        if inb1 && inc0
          acc = visit(action, acc, lin + sb, wb1 * (one_t - fc))
        end
        if inb0 && inc1
          acc = visit(action, acc, lin + sc, wb0 * fc)
        end
        if inb1 && inc1
          acc = visit(action, acc, lin + sb + sc, wb1 * fc)
        end
      end
    end
  end
  acc
end
