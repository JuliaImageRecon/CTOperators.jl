# The distance-driven method (De Man & Basu 2004) in continuous voxel coordinates (`to_index`):
# voxel centers at integer coordinates, voxel v covers [v - 1/2, v + 1/2) per dimension.
#
# Per view, the slicing axis a is the axis along which the central ray of the view (through the
# detector center) advances most in voxels. For a ray of detector cell (iu, iv) and the layer of
# voxels at coordinate t along a, the rays through the midpoints of the four cell edges, (iu ± 1/2,
# iv) and (iu, iv ± 1/2), cross the plane through the layer's centers in four points; their
# bounding box in the two transverse axes b, c is the footprint [bmin, bmax] × [cmin, cmax]. The
# layer contributes the mean of the image over the footprint times the length of the cell's central
# ray in the slab of the layer:
#
#   y += slab length / ((bmax - bmin)(cmax - cmin)) * sum over voxels of o_b o_c x
#
# with o_b, o_c the overlaps of the footprint with the voxel along b and c. All rays of a view use
# the same slicing axis, so neighboring cells share their edge rays and their footprints tile each
# layer. Forward and back projection compute the same weights: the back projection is the exact
# adjoint. Along a transverse axis of a single voxel (a 2D image) the footprint is not averaged:
# the weight is 1 if the central ray lies within that voxel.
#
# The transverse coordinates of an edge ray are linear in t, q(t) = q0 + t q1; the coefficients are
# computed once per ray. The slabs are those of the central ray (segments: clipped to the segment,
# partial end slabs weighted by the part of the slab they cover, as in Joseph's method).
#
# No closures in this code (see siddon/kernels.jl).

@kernel inbounds = true function distance_driven_forward_kernel!(y, @Const(image), grid, geometry, order)
  i = ray_index(order, @index(Global, Linear))
  if i > 0
    y[i] = distance_driven_ray(Gather(image), grid, geometry, i)
  end
end

@kernel inbounds = true function distance_driven_backward_kernel!(image, @Const(y), grid, geometry, order)
  i = ray_index(order, @index(Global, Linear))
  if i > 0
    v = y[i]
    if !iszero(v)
      distance_driven_ray(Scatter(image, v), grid, geometry, i)
    end
  end
end

# The ray of cell (cu, cv) of view k in voxel coordinates: start point and direction.
@inline function voxel_ray(grid, geometry, cu, cv, k)
  a, b = cell_ray(geometry, cu, cv, k)
  to_index(grid, a), (b - a) ./ grid.deltas
end

# The transverse coordinates along axes kb, kc of the ray p + α d where it crosses the plane at
# coordinate t along axis ka, as linear functions of t: (b0, b1, c0, c1) with b(t) = b0 + t b1.
@inline function edge_line(p, d, ka, kb, kc)
  inv_a = inv(pick(d, ka))
  b1 = pick(d, kb) * inv_a
  c1 = pick(d, kc) * inv_a
  pa = pick(p, ka)
  pick(p, kb) - pa * b1, b1, pick(p, kc) - pa * c1, c1
end

# The ray i of a detector geometry through the grid; returns the accumulated value of the action.
@inline function distance_driven_ray(action, grid::VoxelGrid{T}, geometry, i) where {T}
  nu, nv = detectorsize(geometry)
  iu, iv, k = pixel_index(i, nu, nv)
  half = T(1 // 2)
  cu, cv = T(iu), T(iv)
  # the slicing axis of the view, from its central ray
  _, d0 = voxel_ray(grid, geometry, T(nu + 1) / 2, T(nv + 1) / 2, k)
  ka = principal_axis(d0)
  kb = ifelse(ka == 1, 2, 1)
  kc = ifelse(ka == 3, 2, 3)
  # the central ray of the cell and its four edge rays
  a, b = cell_ray(geometry, cu, cv, k)
  p = to_index(grid, a)
  d = (b - a) ./ grid.deltas
  pl, dl = voxel_ray(grid, geometry, cu - half, cv, k)
  pr, dr = voxel_ray(grid, geometry, cu + half, cv, k)
  pd, dd = voxel_ray(grid, geometry, cu, cv - half, k)
  pu, du = voxel_ray(grid, geometry, cu, cv + half, k)
  # rays parallel to the layers (a fan wider than 90 degrees) do not cross them
  (iszero(pick(d, ka)) || iszero(pick(dl, ka)) || iszero(pick(dr, ka)) || iszero(pick(dd, ka)) ||
   iszero(pick(du, ka))) && return zero(T)
  lines = (edge_line(pl, dl, ka, kb, kc), edge_line(pr, dr, ka, kb, kc), edge_line(pd, dd, ka, kb, kc),
           edge_line(pu, du, ka, kb, kc), edge_line(p, d, ka, kb, kc))
  # the slabs along the slicing axis that the central ray crosses
  na = pick(grid.dims, ka)
  lo = SVector(ntuple(m -> m == ka ? half : typemin(T), 3))
  hi = SVector(ntuple(m -> m == ka ? T(na) + half : typemax(T), 3))
  amin, amax = clip(p, d, lo, hi, raytype(geometry))
  amin < amax || return zero(T)
  smin, smax = segment_range(raytype(geometry), T)
  distance_driven_traverse(action, grid.dims, p, d, ka, kb, kc, lines, norm(b - a), amin, amax, smin, smax)
end

# the overlap of the interval [lo, hi] with voxel j, [j - 1/2, j + 1/2)
@inline overlap(lo, hi, j) = min(hi, j + oftype(hi, 1 // 2)) - max(lo, j - oftype(lo, 1 // 2))

# the voxels j in 1:n that the interval [lo, hi] overlaps
@inline function voxel_range(lo, hi, n)
  half = oftype(lo, 1 // 2)
  j0 = unsafe_trunc(Int, floor(clamp(lo + half, zero(lo), oftype(lo, n + 1))))
  j1 = unsafe_trunc(Int, floor(clamp(hi + half, zero(hi), oftype(hi, n + 1))))
  max(j0, 1), min(j1, n)
end

@inline function distance_driven_traverse(action, dims, p::SVector{3,T}, d::SVector{3,T}, ka, kb, kc, lines,
                                          len, amin, amax, smin, smax) where {T}
  nx, ny, nz = dims
  strides = (1, nx, nx * ny)
  na, nb, nc = pick(dims, ka), pick(dims, kb), pick(dims, kc)
  sa, sb, sc = pick(strides, ka), pick(strides, kb), pick(strides, kc)
  pa, da = pick(p, ka), pick(d, ka)
  l1, l2, l3, l4, l0 = lines
  half = T(1 // 2)

  # the layers the clipped central ray crosses, in the direction of the ray
  first = voxel_index(muladd(amin, da, pa), na)
  last = voxel_index(muladd(amax, da, pa), na)
  step = ifelse(da > 0, 1, -1)
  invd = inv(da)
  slab = abs(invd) / 2        # half the parameter length of a slab
  acc = zero(T)
  for s in 0:abs(last - first)
    n = first + s * step
    t = T(n)
    center = (t - pa) * invd  # the parameter of the center plane of layer n
    lo = max(center - slab, smin)
    hi = min(center + slab, smax)
    if lo < hi
      w = (hi - lo) * len     # the length of the central ray in the slab
      # the footprint: the bounding box of the four edge points in the plane of the layer
      b1, b2, b3, b4 = muladd(t, l1[2], l1[1]), muladd(t, l2[2], l2[1]), muladd(t, l3[2], l3[1]), muladd(t, l4[2], l4[1])
      c1, c2, c3, c4 = muladd(t, l1[4], l1[3]), muladd(t, l2[4], l2[3]), muladd(t, l3[4], l3[3]), muladd(t, l4[4], l4[3])
      bmin, bmax = min(min(b1, b2), min(b3, b4)), max(max(b1, b2), max(b3, b4))
      cmin, cmax = min(min(c1, c2), min(c3, c4)), max(max(c1, c2), max(c3, c4))
      # a single voxel across: no footprint, the central ray decides
      if nb == 1
        b0 = muladd(t, l0[2], l0[1])
        bmin, bmax = b0, b0
      end
      if nc == 1
        c0 = muladd(t, l0[4], l0[3])
        cmin, cmax = c0, c0
      end
      jb0, jb1 = voxel_range(bmin, bmax, nb)
      jc0, jc1 = voxel_range(cmin, cmax, nc)
      wb = nb == 1 ? one(T) : inv(bmax - bmin)
      wc = nc == 1 ? one(T) : inv(cmax - cmin)
      if jb0 <= jb1 && jc0 <= jc1 && isfinite(wb) && isfinite(wc)
        base = 1 + (n - 1) * sa
        for jc in jc0:jc1
          oc = nc == 1 ? one(T) : overlap(cmin, cmax, T(jc))
          for jb in jb0:jb1
            ob = nb == 1 ? one(T) : overlap(bmin, bmax, T(jb))
            acc = visit(action, acc, base + (jb - 1) * sb + (jc - 1) * sc, w * (wb * ob) * (wc * oc))
          end
        end
      end
    end
  end
  acc
end
