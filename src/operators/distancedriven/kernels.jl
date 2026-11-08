# The distance-driven method (De Man & Basu 2004) in continuous voxel coordinates (`to_index`):
# voxel centers at integer coordinates, voxel v covers [v - 1/2, v + 1/2) per dimension.
#
# Per view, the slicing axis a is the axis along which the central ray of the view (through the
# detector center) advances most in voxels. For a ray of detector cell (iu, iv) and the layer of
# voxels at coordinate t along a, the rays through the midpoints of the four cell edges, (iu ± 1/2,
# iv) and (iu, iv ± 1/2), cross the plane through the layer's centers; the two u-edge rays bound
# the footprint along the transverse axis that u runs along, the two v-edge rays along the other
# one: the footprint [bmin, bmax] × [cmin, cmax] (De Man & Basu's mapping of the cell boundaries).
# The layer contributes the mean of the image over the footprint times the length of the cell's central
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

@kernel inbounds = true function distance_driven_forward_kernel!(y, @Const(image), grid, geometry, order, raymask)
  i = ray_index(order, @index(Global, Linear))
  if i > 0
    y[i] = is_masked(raymask, i) ? zero(eltype(y)) : distance_driven_ray(Gather(image), grid, geometry, i)
  end
end

@kernel inbounds = true function distance_driven_backward_kernel!(image, @Const(y), grid, geometry, order, raymask)
  i = ray_index(order, @index(Global, Linear))
  if i > 0 && !is_masked(raymask, i)
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
@inline function distance_driven_ray(action, grid::VoxelGrid{T,I}, geometry, i) where {T,I}
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
  # The rays through the edges between cells along u bound the footprint along the transverse
  # axis that u runs along (b for rotating scanners), those along v the other one. Edge rays meet
  # only at the source, so their order along each axis is the same in every layer; it is decided
  # once, in the middle of the volume.
  el, er, ed, eu = edge_line(pl, dl, ka, kb, kc), edge_line(pr, dr, ka, kb, kc), edge_line(pd, dd, ka, kb, kc),
                   edge_line(pu, du, ka, kb, kc)
  tm = T(pick(grid.dims, ka) + 1) / 2
  at(e, m) = m == 1 ? muladd(tm, e[2], e[1]) : muladd(tm, e[4], e[3])
  u_along_b = abs(at(er, 1) - at(el, 1)) >= abs(at(er, 2) - at(el, 2))
  eb1, eb2 = u_along_b ? (el, er) : (ed, eu)   # the pair along b
  ec1, ec2 = u_along_b ? (ed, eu) : (el, er)   # the pair along c
  blo, bhi = at(eb1, 1) <= at(eb2, 1) ? (eb1, eb2) : (eb2, eb1)
  clo, chi = at(ec1, 2) <= at(ec2, 2) ? (ec1, ec2) : (ec2, ec1)
  lines = ((blo[1], blo[2]), (bhi[1], bhi[2]), (clo[3], clo[4]), (chi[3], chi[4]), edge_line(p, d, ka, kb, kc))
  # the slabs along the slicing axis that the central ray crosses, within the box of the image mask
  # (the volume without a mask): the layers outside it hold only masked voxels
  lo = SVector(ntuple(m -> m == ka ? grid.lo[m] : typemin(T), 3))
  hi = SVector(ntuple(m -> m == ka ? grid.hi[m] : typemax(T), 3))
  amin, amax = clip(p, d, lo, hi, raytype(geometry))
  amin < amax || return zero(T)
  smin, smax = segment_range(raytype(geometry), T)
  distance_driven_traverse(action, grid.dims, p, d, ka, kb, kc, lines, norm(b - a), amin, amax, smin, smax)
end

# The four voxels i, i + db, i + dc, i + db + dc with their weights; db or dc is 0 (and the weights
# of the repeated voxels 0) where the footprint covers a single voxel along that axis. The forward
# projection loads all four before it combines them; the back projection skips weights of 0.
@inline function visit_quad(g::Gather, acc, i, db, dc, w00, w10, w01, w11)
  x = g.image
  # values converted on reading: an image stored in lower precision is summed in that of the weights
  @inbounds v00, v10, v01, v11 = oftype(w00, x[i]), oftype(w00, x[i+db]), oftype(w00, x[i+dc]), oftype(w00, x[i+db+dc])
  muladd(w11, v11, muladd(w01, v01, muladd(w10, v10, muladd(w00, v00, acc))))
end

@inline function visit_quad(s::Scatter, acc, i, db, dc, w00, w10, w01, w11)
  acc = visit(s, acc, i, w00)
  iszero(w10) || (acc = visit(s, acc, i + db, w10))
  iszero(w01) || (acc = visit(s, acc, i + dc, w01))
  iszero(w11) || (acc = visit(s, acc, i + db + dc, w11))
  acc
end

# the voxels j in 1:n that the interval [lo, hi] overlaps, and the overlaps of the first and the
# last of them (voxel j covers [j - 1/2, j + 1/2); the voxels in between are covered completely);
# j in the integer type of n
@inline function voxel_overlaps(lo, hi, n::I) where {I}
  half = oftype(lo, 1 // 2)
  j0 = max(unsafe_trunc(I, floor(clamp(lo + half, zero(lo), oftype(lo, n + 1)))), one(I))
  j1 = min(unsafe_trunc(I, floor(clamp(hi + half, zero(hi), oftype(hi, n + 1)))), n)
  o0 = min(hi, j0 + half) - max(lo, j0 - half)
  o1 = min(hi, j1 + half) - max(lo, j1 - half)
  j0, j1, o0, o1
end

# the footprint along an axis of n voxels: [lo, hi], or for a single voxel (a 2D image) the voxel
# itself if the central ray q lies in it, else nothing: there the footprint is not averaged
@inline function footprint(lo, hi, q, n)
  half = oftype(q, 1 // 2)
  inside = half <= q < 3half
  n == 1 ? (ifelse(inside, half, 3half), ifelse(inside, 3half, half)) : (lo, hi)
end

@inline function distance_driven_traverse(action, dims::NTuple{3,I}, p::SVector{3,T}, d::SVector{3,T}, ka, kb, kc,
                                          lines, len, amin, amax, smin, smax) where {I,T}
  # all integer arithmetic in I: a literal 1 would promote it to Int64
  unit = one(I)
  nx, ny, nz = dims
  strides = (unit, nx, nx * ny)
  na, nb, nc = pick(dims, ka), pick(dims, kb), pick(dims, kc)
  sa, sb, sc = pick(strides, ka), pick(strides, kb), pick(strides, kc)
  pa, da = pick(p, ka), pick(d, ka)
  lb0, lb1, lc0, lc1, l0 = lines   # the bounds of the footprint along b and c, the central ray

  # the layers the clipped central ray crosses, in the direction of the ray
  first = voxel_index(muladd(amin, da, pa), na)
  last = voxel_index(muladd(amax, da, pa), na)
  step = ifelse(da > 0, unit, -unit)
  invd = inv(da)
  slab = abs(invd) / 2        # half the parameter length of a slab
  acc = zero(T)
  for s in zero(I):abs(last - first)
    n = first + s * step
    t = T(n)
    center = (t - pa) * invd  # the parameter of the center plane of layer n
    lo = max(center - slab, smin)
    hi = min(center + slab, smax)
    # the footprint in the plane of the layer
    bmin, bmax = footprint(muladd(t, lb0[2], lb0[1]), muladd(t, lb1[2], lb1[1]), muladd(t, l0[2], l0[1]), nb)
    cmin, cmax = footprint(muladd(t, lc0[2], lc0[1]), muladd(t, lc1[2], lc1[1]), muladd(t, l0[4], l0[3]), nc)
    jb0, jb1, ob0, ob1 = voxel_overlaps(bmin, bmax, nb)
    jc0, jc1, oc0, oc1 = voxel_overlaps(cmin, cmax, nc)
    area = (bmax - bmin) * (cmax - cmin)
    if lo < hi && jb0 <= jb1 && jc0 <= jc1 && area > 0
      # the length of the central ray in the slab over the area of the footprint
      w = (hi - lo) * len / area
      base = unit + (n - unit) * sa
      if jb1 - jb0 <= 1 && jc1 - jc0 <= 1
        # at most 2 × 2 voxels (the common case): four independent accesses, as in Joseph's method;
        # a missing second voxel along an axis gets the weight 0
        i00 = base + (jb0 - unit) * sb + (jc0 - unit) * sc
        db = (jb1 - jb0) * sb
        dc = (jc1 - jc0) * sc
        wb1 = ifelse(jb1 > jb0, ob1, zero(T))
        wc0, wc1 = w * oc0, w * ifelse(jc1 > jc0, oc1, zero(T))
        acc = visit_quad(action, acc, i00, db, dc, wc0 * ob0, wc0 * wb1, wc1 * ob0, wc1 * wb1)
      else
      for jc in jc0:jc1
        wc = w * ifelse(jc == jc0, oc0, ifelse(jc == jc1, oc1, one(T)))
        row = base + (jc - unit) * sc
        for jb in jb0:jb1
          ob = ifelse(jb == jb0, ob0, ifelse(jb == jb1, ob1, one(T)))
          acc = visit(action, acc, row + (jb - unit) * sb, wc * ob)
        end
      end
      end
    end
  end
  acc
end
