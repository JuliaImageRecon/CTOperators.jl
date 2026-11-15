# Per-view tables for `project_point` in the voxel-driven projector. Its kernels loop over the views
# for every voxel, so the parts of `project_point` that depend only on the view are computed here once
# per view: on the host in Float64 from the projector's geometry, then stored in its element type on
# its device. `project_point` of a table gives the `DetectorHit` of `project_point` of the geometry,
# with the arithmetic rearranged (equal up to rounding).
#
# Flat detector (ConeBeam), per view: source s, detector center c, cell vectors u, v; n = u × v,
# e = c - s, w = e · n, and D1, D2 the rows of G^-1 [u'; v'] with G = [u·u u·v; u·v v·v] (so that
# alpha = D1 · q, beta = D2 · q for q = alpha u + beta v). The ray from s through p meets the
# detector plane at s + t d (d = p - s, t = w / (d · n)), so
#
#   alpha = D1 · (t d - e) = ((w D1 - (D1 · e) n) · d) / (n · d),
#
# numerator and denominator linear in d, hence in [p; 1]. Divided by w, the denominator is
# n · d / w = 1 / t. The projection of a view is a 3 x 4 matrix P with rows a, b, c:
#
#   (a, b, c) = P [p; 1],  cu = a / c,  cv = b / c,  t = 1 / c   (the magnification R / r),
#   a = (w D1 - (D1 · e) n + (nu + 1)/2 n) / w and its last entry -(that row) · s, b the same with
#   D2 and nv, c = (n, -n · s) / w.
#
# The density (R/r)^2 / (|n| cos phi) = t^2 |d| / |d · n| = t^3 |d| / |w| (t > 0) needs |d|, so the
# table keeps the source and 1 / |w|. With w = 0 (the source in the plane of the detector) the rows
# are 0: c = 0, t is not finite and no point projects, as in `project_point` of the geometry.
#
# Parallel beam (direction e), per view: the point p moves along e into the detector plane,
# q = (p - c) - ((p - c) · n / (e · n)) e, so alpha = (D1 - (D1 · e / e · n) n) · (p - c): affine,
# (cu, cv) = P [p; 1] with a 2 x 4 matrix P, magnification 1, the density |e| / |e · n| per view.
# With e · n = 0 (rays parallel to the detector) the view has no hits.
#
# Other geometries (ArcConeBeam: the hit is an angle, no matrix) are their own table: `project_point`
# of the geometry.
#
# Runs of R voxels (p0, p0 + step, ...): `view_run` and `run_hit` give the hits of them in one view. For
# the matrices, a, b, c of the first voxel are computed once and the others follow by adding the
# matrix times the step (a sum of the same products, rounded differently); this is what makes runs
# faster on the CPU (measured: projecting every voxel of the run separately gains almost nothing).

struct FlatProjection{M}
  matrix::M     # 12 x nviews: the rows a (1:4), b (5:8), c (9:12) of P
  source::M     # 3 x nviews
  scalars::M    # 2 x nviews: the row spacing |v| and 1 / |w|
end

struct ParallelProjection{M}
  matrix::M     # 8 x nviews: the rows a (1:4), b (5:8) of P
  scalars::M    # 3 x nviews: the row spacing |v|, the density, and 1 if the view has hits, else 0
end

Adapt.@adapt_structure FlatProjection
Adapt.@adapt_structure ParallelProjection

# the table of `geometry`, on `backend`, in the element type of the geometry
point_projection(geometry, backend) = geometry

function point_projection(g::ConeBeam{T}, backend) where {T}
  h = adapt(Array{Float64}, g)
  n = nviews(h)
  matrix, source, scalars = zeros(12, n), zeros(3, n), zeros(2, n)
  for k in 1:n
    s, c, u, v = column(h.source, k), column(h.center, k), column(h.u, k), column(h.v, k)
    nrm = cross(u, v)
    e = c - s
    w = dot(e, nrm)
    D1, D2 = dual_rows(u, v)
    source[:, k] = s
    scalars[:, k] = [norm(v), iszero(w) ? 0.0 : 1 / abs(w)]
    iszero(w) && continue
    row(D, half) = (a = (w * D - dot(D, e) * nrm + half * nrm) / w; vcat(a, -dot(a, s)))
    matrix[:, k] = vcat(row(D1, (h.nu + 1) / 2), row(D2, (h.nv + 1) / 2), vcat(nrm / w, -dot(nrm, s) / w))
  end
  FlatProjection(device_array(backend, T, matrix), device_array(backend, T, source), device_array(backend, T, scalars))
end

function point_projection(g::ParallelBeam{T}, backend) where {T}
  h = adapt(Array{Float64}, g)
  n = nviews(h)
  matrix, scalars = zeros(8, n), zeros(3, n)
  for k in 1:n
    e, c, u, v = column(h.direction, k), column(h.center, k), column(h.u, k), column(h.v, k)
    nrm = cross(u, v)
    en = dot(e, nrm)
    scalars[:, k] = [norm(v), iszero(en) ? 0.0 : norm(e) / abs(en), iszero(en) ? 0.0 : 1.0]
    iszero(en) && continue
    D1, D2 = dual_rows(u, v)
    row(D, half) = (a = D - (dot(D, e) / en) * nrm; vcat(a, half - dot(a, c)))
    matrix[:, k] = vcat(row(D1, (h.nu + 1) / 2), row(D2, (h.nv + 1) / 2))
  end
  ParallelProjection(device_array(backend, T, matrix), device_array(backend, T, scalars))
end

# the rows of G^-1 [u'; v']: alpha = D1 · q and beta = D2 · q for q = alpha u + beta v
function dual_rows(u, v)
  uu, vv, uv = dot(u, u), dot(v, v), dot(u, v)
  det = uu * vv - uv * uv
  (vv * u - uv * v) / det, (uu * v - uv * u) / det
end

device_array(backend, T, a) = copyto!(KA.allocate(backend, T, size(a)), T.(a))

# row r (entries 4r - 3 : 4r) of the matrix of view k applied to [p; 1]
@inline matrix_row(m, r, k, p) =
  muladd(m[4r - 3, k], p[1], muladd(m[4r - 2, k], p[2], muladd(m[4r - 1, k], p[3], m[4r, k])))

@inline function project_point(f::FlatProjection, p, k)
  a, b, c = matrix_row(f.matrix, 1, k, p), matrix_row(f.matrix, 2, k, p), matrix_row(f.matrix, 3, k, p)
  t = inv(c)
  d = p - column(f.source, k)
  DetectorHit(a * t, b * t, t * t * t * norm(d) * f.scalars[2, k], t, f.scalars[1, k], t > 0 && isfinite(t))
end

@inline function project_point(f::ParallelProjection, p, k)
  T = eltype(f.matrix)
  DetectorHit(matrix_row(f.matrix, 1, k, p), matrix_row(f.matrix, 2, k, p), f.scalars[2, k], one(T), f.scalars[1, k],
              !iszero(f.scalars[3, k]))
end

# The run of view k: what the hits of the voxels p0 + (r - 1) step have in common, prepared once per
# view (`view_run`), and the hit of voxel r (`run_hit`), computed where it is used. The default
# projects every voxel of the run with `project_point`.
struct PointRun{P,V}
  projection::P
  p0::V
  step::V
  k::Int
end

@inline view_run(projection, p0, step, k) = PointRun(projection, p0, step, k)
@inline run_hit(v::PointRun, r) = project_point(v.projection, v.p0 + (r - 1) * v.step, v.k)

# the change of row r of the matrix of view k for a step of the point
@inline matrix_step(m, r, k, step) = muladd(m[4r - 3, k], step[1], muladd(m[4r - 2, k], step[2], m[4r - 1, k] * step[3]))

struct FlatRun{T,V}
  a0::T
  b0::T
  c0::T
  da::T
  db::T
  dc::T
  d0::V         # p0 - source
  step::V
  rowsize::T
  inv_w::T
end

@inline function view_run(f::FlatProjection, p0, step, k)
  m = f.matrix
  FlatRun(matrix_row(m, 1, k, p0), matrix_row(m, 2, k, p0), matrix_row(m, 3, k, p0), matrix_step(m, 1, k, step),
          matrix_step(m, 2, k, step), matrix_step(m, 3, k, step), p0 - column(f.source, k), step, f.scalars[1, k],
          f.scalars[2, k])
end

@inline function run_hit(v::FlatRun, r)
  t = inv(muladd(r - 1, v.dc, v.c0))
  d = v.d0 + (r - 1) * v.step
  DetectorHit(muladd(r - 1, v.da, v.a0) * t, muladd(r - 1, v.db, v.b0) * t, t * t * t * norm(d) * v.inv_w, t,
              v.rowsize, t > 0 && isfinite(t))
end

struct ParallelRun{T}
  a0::T
  b0::T
  da::T
  db::T
  density::T
  rowsize::T
  ok::Bool
end

@inline function view_run(f::ParallelProjection, p0, step, k)
  m = f.matrix
  ParallelRun(matrix_row(m, 1, k, p0), matrix_row(m, 2, k, p0), matrix_step(m, 1, k, step), matrix_step(m, 2, k, step),
              f.scalars[2, k], f.scalars[1, k], !iszero(f.scalars[3, k]))
end

@inline run_hit(v::ParallelRun{T}, r) where {T} =
  DetectorHit(muladd(r - 1, v.da, v.a0), muladd(r - 1, v.db, v.b0), v.density, one(T), v.rowsize, v.ok)
