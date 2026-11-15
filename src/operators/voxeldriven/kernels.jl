# The voxel-driven method: one thread per run of R neighboring voxels along x (within the bounding box
# of the image mask; R from the layout of the launch, 1 on GPUs by default), a loop over the views and,
# within it, over the voxels of the run. Per view the voxel center p is projected onto the detector
# (`project_point` of the per-view table, projection.jl, at continuous cell coordinates (cu, cv)), and
# the voxel meets the 2 x 2 cells around that point with the bilinear weights times the scale of the
# voxel in that view (`voxel_scale`):
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

@kernel inbounds = true function voxel_driven_forward_kernel!(y, @Const(image), grid, geometry, projection, box, raymask,
                                                             scale, run::Val{R}, slab) where {R}
  j = @index(Global, Linear)
  if j <= run_count(box, run)
    lin, q, m = box_run(grid, box, j, run)
    values = zero(MVector{R,eltype(image)})
    for r in 1:R
      if r <= m
        values[r] = image[lin + r - 1]
      end
    end
    voxel_driven_run(ScatterRun(y, SVector(values)), grid, geometry, projection, raymask, scale, q, run, slab)
  end
end

@kernel inbounds = true function voxel_driven_backward_kernel!(image, @Const(y), grid, geometry, projection, box, raymask,
                                                              scale, run::Val{R}, slab) where {R}
  j = @index(Global, Linear)
  if j <= run_count(box, run)
    lin, q, m = box_run(grid, box, j, run)
    acc = voxel_driven_run(Gather(y), grid, geometry, projection, raymask, scale, q, run, slab)
    for r in 1:R
      if r <= m
        image[lin + r - 1] = acc[r]
      end
    end
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

# the number of runs of R voxels along x in the box: per row cld(n[1], R), the last one shorter
@inline run_count((first, n), ::Val{R}) where {R} = cld(n[1], oftype(n[1], R)) * n[2] * n[3]

# the linear index and the voxel coordinates (i, j, k) of the first voxel of run j of the box, and
# the number of voxels of the run (R, fewer for the last run of a row)
@inline function box_run(grid::VoxelGrid{T,I}, (first, n), j, ::Val{R}) where {T,I,R}
  unit = one(I)
  len = I(R)
  nr = cld(n[1], len)
  r = unsafe_trunc(I, j) - unit
  a, r = rem(r, nr), div(r, nr)
  b, c = rem(r, n[2]), div(r, n[2])
  i1, i2, i3 = first[1] + a * len, first[2] + b, first[3] + c
  nx, ny, _ = grid.dims
  i1 + (i2 - unit) * nx + (i3 - unit) * nx * ny, (i1, i2, i3), min(len, n[1] - a * len)
end

# The actions of a run: the back projection gathers into one sum per voxel; the forward projection
# scatters the value of each voxel (voxels with value 0 are skipped, also those beyond the end of a
# shorter run, whose values are 0).
struct ScatterRun{A,V}
  data::A
  values::V
end

@inline run_action(g::Gather, r) = g
@inline run_action(s::ScatterRun, r) = Scatter(s.data, s.values[r])
# (the gather computes all R sums, also beyond the end of a shorter run, where they are not written:
# no branch in the loop)
@inline run_skip(::Gather, r) = false
@inline run_skip(s::ScatterRun, r) = iszero(s.values[r])

# All views of the voxels of a run starting at voxel q; returns the accumulated value of the action
# of each voxel. The views are the outer loop: per view the run is prepared once (`view_run`). No
# branch per voxel: a voxel without hit (or skipped) gets the weight 0, which the cells skip (and a
# position that is not finite fails their range check). `slab`: Val(true) for a 2D image (one voxel
# along z), decided on the host.
@inline function voxel_driven_run(action, grid::VoxelGrid{T}, geometry, projection, raymask, scale, q, ::Val{R},
                                  ::Val{SLAB}) where {T,R,SLAB}
  nu, nv = detectorsize(geometry)
  p0 = grid.first .+ (SVector{3,T}(q) .- one(T)) .* grid.deltas
  step = SVector(grid.deltas[1], zero(T), zero(T))
  acc = zero(MVector{R,T})
  for k in 1:nviews(geometry)
    base = (k - 1) * nu * nv
    run = view_run(projection, p0, step, k)
    for r in 1:R
      hit = run_hit(run, r)
      s = ifelse(hit.ok && !run_skip(action, r), voxel_scale(scale, geometry, k, grid, hit), zero(T))
      a = run_action(action, r)
      acc[r] = SLAB ? visit_rows(a, acc[r], raymask, base, nu, nv, hit, s, abs(grid.deltas[3])) :
               visit_quad_cells(a, acc[r], raymask, base, nu, nv, hit.cu, hit.cv, s)
    end
  end
  SVector(acc)
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

# Launches `kernel(out, in, grid, geometry, projection, box, raymask, scale, Val(R), Val(slab))` over the
# runs of voxels of the bounding box of the image mask.
function launch_voxels!(kernel, op::VoxelDrivenProjector, out, in)
  box = voxel_box(op.grid)
  backend = KA.get_backend(op)
  config = voxel_launch(op.launch, backend)
  run = Val(voxels_per_thread(config))
  n = Int(run_count(box, run))
  n == 0 && return out
  wg = config.workgroupsize
  k = isnothing(wg) ? kernel(backend) : kernel(backend, wg)
  k(out, in, op.grid, geometry(op), op.projection, box, ray_mask(op), op.scale, run, Val(op.grid.dims[3] == 1); ndrange=n)
  out
end

# the projector's launch or the defaults of the device: on the CPU runs of 8 voxels, on GPUs runs of 2
# in workgroups of 64 (measured on one CPU and one GPU, benchmark/README.md)
voxel_launch(launch::LaunchConfig, backend) = launch
voxel_launch(::Nothing, backend) =
  backend isa KA.CPU ? LaunchConfig(; layout=(8,)) : LaunchConfig(; workgroupsize=64, layout=(2,))

voxels_per_thread(::LaunchConfig{0}) = 1
voxels_per_thread(c::LaunchConfig{1}) = c.layout[1]
