# The voxel-driven method: one thread per voxel (within the bounding box of the image mask), a loop
# over the views. Per view the voxel center p is projected onto the detector (`project_point`, at
# continuous cell coordinates (cu, cv)), and the voxel meets the 2 x 2 cells around that point with
# the bilinear weights times the scale of the voxel in that view (`voxel_scale`):
#
#   a_ij = scale * (1 - |cu - iu|) (1 - |cv - iv|)    for the cells with |cu - iu| < 1, |cv - iv| < 1
#
# Cells outside the detector and masked cells are skipped (weight 0). The actions are those of the
# ray-driven projectors with the roles swapped: they act on the data, not on the image. The back
# projection gathers the data (`Gather`) and writes the voxel once; the forward projection scatters
# the voxel value into the data (`Scatter`, atomic additions).
#
# For a 2D image (one voxel along z) the rows are not interpolated: every row whose ray at the voxel
# lies within the slab of the image gets weight 1 (as in the distance-driven method), see
# `LineIntegralScale`.
#
# No closures in this code (see siddon/kernels.jl).

@kernel inbounds = true function voxel_driven_forward_kernel!(y, @Const(image), grid, geometry, box, raymask, scale)
  j = @index(Global, Linear)
  if j <= box_length(box)
    lin, q = box_voxel(grid, box, j)
    v = image[lin]
    if !iszero(v)
      voxel_driven_voxel(Scatter(y, v), grid, geometry, raymask, scale, q)
    end
  end
end

@kernel inbounds = true function voxel_driven_backward_kernel!(image, @Const(y), grid, geometry, box, raymask, scale)
  j = @index(Global, Linear)
  if j <= box_length(box)
    lin, q = box_voxel(grid, box, j)
    image[lin] = voxel_driven_voxel(Gather(y), grid, geometry, raymask, scale, q)
  end
end

# The voxels of the bounding box of the image mask: first index and size per dimension, in the
# index type of the grid. Empty (a size of 0) for an empty mask.
function voxel_box(grid::VoxelGrid{T,I}) where {T,I}
  half = T(1 // 2)
  first = map(l -> I(round(Int, l + half)), Tuple(grid.lo))
  last = map(h -> I(round(Int, h - half)), Tuple(grid.hi))
  first, map((a, b) -> max(b - a + one(I), zero(I)), first, last)
end

@inline box_length((first, n)) = n[1] * n[2] * n[3]

# the linear index and the voxel coordinates (i, j, k) of thread j of the box
@inline function box_voxel(grid::VoxelGrid{T,I}, (first, n), j) where {T,I}
  unit = one(I)
  r = unsafe_trunc(I, j) - unit
  a, r = rem(r, n[1]), div(r, n[1])
  b, c = rem(r, n[2]), div(r, n[2])
  i1, i2, i3 = first[1] + a, first[2] + b, first[3] + c
  nx, ny, _ = grid.dims
  i1 + (i2 - unit) * nx + (i3 - unit) * nx * ny, (i1, i2, i3)
end

# All views of one voxel; returns the accumulated value of the action.
@inline function voxel_driven_voxel(action, grid::VoxelGrid{T}, geometry, raymask, scale, q) where {T}
  p = grid.first .+ (SVector{3,T}(q) .- one(T)) .* grid.deltas
  nu, nv = detectorsize(geometry)
  slab = grid.dims[3] == 1
  acc = zero(T)
  for k in 1:nviews(geometry)
    hit = project_point(geometry, p, k)
    if hit.ok
      s = voxel_scale(scale, geometry, k, grid, hit)
      base = (k - 1) * nu * nv
      acc = slab ? visit_rows(action, acc, raymask, base, nu, nv, hit, s, abs(grid.deltas[3])) :
            visit_quad_cells(action, acc, raymask, base, nu, nv, hit.cu, hit.cv, s)
    end
  end
  acc
end

# cell (iu, iv) of the view starting after `base`, with weight w; skipped outside the detector, if
# masked, or with weight 0
@inline function visit_cell(action, acc, raymask, base, nu, nv, iu, iv, w)
  if 1 <= iu <= nu && 1 <= iv <= nv && !iszero(w)
    i = base + iu + (iv - 1) * nu
    is_masked(raymask, i) || (acc = visit(action, acc, i, w))
  end
  acc
end

# the 2 x 2 cells around (cu, cv) with the bilinear weights times s
@inline function visit_quad_cells(action, acc, raymask, base, nu, nv, cu, cv, s)
  T = typeof(cu)
  # positions far off the detector would overflow the integer conversion; they meet no cell
  (-one(T) < cu < T(nu + 1) && -one(T) < cv < T(nv + 1)) || return acc
  fu, fv = floor(cu), floor(cv)
  iu, iv = unsafe_trunc(Int, fu), unsafe_trunc(Int, fv)
  wu, wv = cu - fu, cv - fv
  acc = visit_cell(action, acc, raymask, base, nu, nv, iu, iv, s * (1 - wu) * (1 - wv))
  acc = visit_cell(action, acc, raymask, base, nu, nv, iu + 1, iv, s * wu * (1 - wv))
  acc = visit_cell(action, acc, raymask, base, nu, nv, iu, iv + 1, s * (1 - wu) * wv)
  visit_cell(action, acc, raymask, base, nu, nv, iu + 1, iv + 1, s * wu * wv)
end

# A 2D image: interpolated along u; along v every row whose ray at the voxel lies within the slab of
# thickness dz (half its shadow: magnification dz / 2, in rows) gets the full weight.
@inline function visit_rows(action, acc, raymask, base, nu, nv, hit, s, dz)
  T = typeof(hit.cu)
  cu, cv = hit.cu, hit.cv
  half = hit.magnification * dz / (2 * hit.rowsize)
  (-one(T) < cu < T(nu + 1) && -half < cv < T(nv + 1) + half) || return acc
  fu = floor(cu)
  iu = unsafe_trunc(Int, fu)
  wu = cu - fu
  first = max(unsafe_trunc(Int, floor(cv - half)) + 1, 1)
  last = min(unsafe_trunc(Int, ceil(cv + half)) - 1, nv)
  for iv in first:last
    acc = visit_cell(action, acc, raymask, base, nu, nv, iu, iv, s * (1 - wu))
    acc = visit_cell(action, acc, raymask, base, nu, nv, iu + 1, iv, s * wu)
  end
  acc
end

# Launches `kernel(out, in, grid, geometry, box, raymask, scale)` over the voxels of the bounding box of
# the image mask.
function launch_voxels!(kernel, op::VoxelDrivenProjector, out, in)
  box = voxel_box(op.grid)
  n = Int(box_length(box))
  n == 0 && return out
  backend = KA.get_backend(op)
  wg = isnothing(op.launch) ? (backend isa KA.CPU ? nothing : 64) : op.launch.workgroupsize
  k = isnothing(wg) ? kernel(backend) : kernel(backend, wg)
  k(out, in, op.grid, geometry(op), box, ray_mask(op), op.scale; ndrange=n)
  out
end
