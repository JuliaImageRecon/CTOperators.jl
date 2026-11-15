@testitem "Voxel driven: contract" tags = [:voxeldriven, :gpu] setup = [Backends, Contract] begin
  contract_tests(VoxelDrivenProjector, arrayTypes, floatTypes; raylists=false)
end

@testitem "Voxel driven: operator interface" tags = [:voxeldriven] setup = [Projections] begin
  using Adapt
  image = ImageGeom(; dims=(8, 8, 6))
  g = circular_cone_beam(; angles=range(0, 2pi, 9)[1:8], dso=25, dsd=50, nu=12, nv=9, du=1.0)
  op = VoxelDrivenProjector(image, g; eltype=Float64)
  @test size(op) == (12 * 9 * 8, 8 * 8 * 6) && datasize(op) == (12, 9, 8)
  @test image_geometry(op) === image && geometry(op) isa ConeBeam{Float64} && op.scale === LineIntegralScale()
  x = rand(8, 8, 6)
  @test same_projection(VoxelDrivenProjector, op * vec(x), vec(op * x))
  @test isapprox(subset(op, [2, 5]) * x, (op * x)[:, :, [2, 5]])
  op32 = adapt(Array{Float32}, op)
  @test op32 isa VoxelDrivenProjector{Float32} && image_geometry(op32) === image
  # the scale travels with subset and adapt
  unit = VoxelDrivenProjector(image, g; scale=UnitScale())
  @test subset(unit, [1]).scale === UnitScale() && adapt(Array{Float64}, unit).scale === UnitScale()
  # ray lists have no detector, a MultiRay has no meaning here
  @test_throws ArgumentError VoxelDrivenProjector(image, RayList(g))
  @test_throws ArgumentError VoxelDrivenProjector(image, MultiRay(g, RegularPoints(2)))
  masked = VoxelDrivenProjector(ImageGeom(MaskCircle(); dims=(8, 8, 6)), g; raymask=trues(datasize(g)))
  @test CTOperators.handles_raymask(masked) && !isnothing(CTOperators.image_mask(masked))
  @test CTOperators.ray_mask(subset(masked, [2, 5])) == trues(12, 9, 2)
  @test masked.grid isa CTOperators.VoxelGrid{Float32,Int32}
end

# A voxel that projects inside the detector puts its mass V (R/r)^2 / cos(phi) on the detector: the
# data of one view sum to the voxel volume times the density of the projection (the bilinear weights
# of the four cells sum to 1; times the cell area the sum is the mass).
@testitem "Voxel driven: mass of a voxel" tags = [:voxeldriven] begin
  using CTOperators: project_point, VoxelGrid
  image = ImageGeom(; dims=(7, 6, 5), deltas=(1.1, 0.9, 1.3), offsets=(0.3, -0.2, 0.1))
  grid = VoxelGrid{Float64}(image)
  for g in (circular_cone_beam(; angles=[0.3], dso=30, dsd=60, nu=40, nv=30, du=0.8, dv=0.7),
            circular_cone_beam(; angles=[0.3], dso=30, dsd=60, nu=40, nv=30, du=0.8, dv=0.7, detector=:arc),
            parallel_beam(; angles=[0.3], nu=40, nv=30, du=0.8, dv=0.7))
    op = VoxelDrivenProjector(image, g; eltype=Float64)
    for q in ((1, 1, 1), (4, 3, 2), (7, 6, 5))
      x = zeros(7, 6, 5)
      x[q...] = 1
      p = grid.first .+ (collect(q) .- 1) .* grid.deltas
      h = project_point(geometry(op), p, 1)
      @test isapprox(sum(op * x), prod(abs.(grid.deltas)) * h.density; rtol=1e-12)
    end
  end
end

# With cells as large as the voxels (or larger), the projection of a smooth object is close to the
# distance-driven one, as Siddon's is; with cells much smaller than the projected voxels the
# voxel-driven projection shows its known high-frequency artifacts (De Man & Basu 2004), but keeps
# the total.
@testitem "Voxel driven: ball" tags = [:voxeldriven] begin
  using ImagePhantoms, LinearAlgebra
  using ImageGeoms: axis
  image = ImageGeom(; dims=(24, 24, 24), deltas=(1.0, 1.0, 1.0))
  x = Float64.(phantom(axis(image, 1), axis(image, 2), axis(image, 3), [ellipsoid((0, 0, 0), (8, 8, 8), (0, 0, 0), 1.0)]))
  rel(a, b) = norm(a - b) / norm(b)
  for detector in (:flat, :arc), du in (1.0, 2.0)
    g = circular_cone_beam(; angles=range(0, 2pi, 9)[1:8], dso=60, dsd=120, nu=round(Int, 128 / du),
                           nv=round(Int, 128 / du), du, detector)
    yv = VoxelDrivenProjector(image, g; eltype=Float64) * x
    yd = DistanceDrivenProjector(image, g; eltype=Float64) * x
    @test isapprox(sum(yv), sum(yd); rtol=2e-3)
    # The detector models differ (a point interpolated between cells, a box averaged over a cell):
    # no bound from theory; the thresholds keep the measured values, 0.027 (cells of one voxel at
    # the center) and 0.23 (half a voxel), on their side.
    du == 2.0 ? (@test rel(yv, yd) < 0.04) : (@test rel(yv, yd) > 0.15)
  end
end

# A 2D image is a slab: in-plane geometries give the in-plane line integrals, with the total of the
# distance-driven projection.
@testitem "Voxel driven: 2D image" tags = [:voxeldriven] begin
  using ImagePhantoms, LinearAlgebra
  using ImageGeoms: axis
  image = ImageGeom(; dims=(64, 64), deltas=(1.0, 1.0))
  x = Float64.(phantom(axis(image, 1), axis(image, 2), [ellipse((3, -2), (20, 14), 0.3, 1.0)]))
  for g in (circular_cone_beam(; angles=range(0, 2pi, 37)[1:36], dso=120, dsd=240, nu=64, nv=1, du=2.0),
            parallel_beam(; angles=range(0, pi, 37)[1:36], nu=64, nv=1, du=1.0))
    yv = VoxelDrivenProjector(image, g; eltype=Float64) * x
    yd = DistanceDrivenProjector(image, g; eltype=Float64) * x
    @test isapprox(sum(yv), sum(yd); rtol=1e-5)
    # the detector models differ (see the ball): measured 0.008 (fan), 0.016 (parallel)
    @test norm(yv - yd) / norm(yd) < 0.03
  end
end

# A slab (2D image, or 3D with one voxel along z) and a detector with several rows: the rows whose
# rays pass through the slab get its in-plane line integrals, the others 0, as for the
# distance-driven projector, which uses the same rule (row sums equal; the detector models differ
# along u as above, measured 0.010 to 0.015).
@testitem "Voxel driven: slab with several rows" tags = [:voxeldriven] begin
  using ImagePhantoms, LinearAlgebra
  using ImageGeoms: axis
  for image in (ImageGeom(; dims=(64, 64), deltas=(1.0, 1.0)), ImageGeom(; dims=(64, 64, 1), deltas=(1.0, 1.0, 2.0)))
    x = reshape(phantom(axis(image, 1), axis(image, 2), [ellipse((3, -2), (20, 14), 0.3, 1.0)]), image.dims)
    for detector in (:flat, :arc)
      g = circular_cone_beam(; angles=range(0, 2pi, 37)[1:36], dso=120, dsd=240, nu=64, nv=9, du=2.0, dv=1.0, detector)
      yv = VoxelDrivenProjector(image, g; eltype=Float64) * x
      yd = DistanceDrivenProjector(image, g; eltype=Float64) * x
      rows_v, rows_d = vec(sum(yv; dims=(1, 3))), vec(sum(yd; dims=(1, 3)))
      @test count(>(0), rows_v) == (length(image.dims) == 2 ? 3 : 5) && iszero(rows_v[1:2]) && iszero(rows_v[8:9])
      @test isapprox(rows_v, rows_d; rtol=1e-3)
      @test norm(yv - yd) / norm(yd) < 0.03
    end
  end
end

# Scales of one's own: an isbits factor, and one weight per view in an array, which subset and adapt
# carry along.
@testitem "Voxel driven: custom scales" tags = [:voxeldriven] begin
  using Adapt
  using CTOperators: voxel_scale, VoxelGrid
  struct Doubled <: AbstractVoxelScale end
  CTOperators.voxel_scale(::Doubled, geometry, k, grid::VoxelGrid{T}, hit) where {T} =
    2 * voxel_scale(LineIntegralScale(), geometry, k, grid, hit)
  struct PerViewScale{V} <: AbstractVoxelScale
    weights::V
  end
  Adapt.@adapt_structure PerViewScale
  CTOperators.voxel_scale(s::PerViewScale, geometry, k, grid::VoxelGrid{T}, hit) where {T} = T(s.weights[k])
  CTOperators.subset(s::PerViewScale, views) = PerViewScale(s.weights[views])

  image = ImageGeom(; dims=(8, 8, 6))
  g = circular_cone_beam(; angles=range(0, 2pi, 9)[1:8], dso=25, dsd=50, nu=12, nv=9, du=1.0)
  x = rand(8, 8, 6)
  @test isapprox(VoxelDrivenProjector(image, g; eltype=Float64, scale=Doubled()) * x,
                 2 * (VoxelDrivenProjector(image, g; eltype=Float64) * x))
  w = collect(1.0:8.0)
  op = VoxelDrivenProjector(image, g; eltype=Float64, scale=PerViewScale(w))
  unit = VoxelDrivenProjector(image, g; eltype=Float64, scale=UnitScale())
  @test isapprox(op * x, (unit * x) .* reshape(w, 1, 1, 8))
  @test isapprox(subset(op, [2, 5]) * x, (op * x)[:, :, [2, 5]])
  op32 = adapt(Array{Float32}, op)
  @test op32.scale.weights isa Vector{Float32} && isapprox(op32 * Float32.(x), op * x; rtol=1e-5)
end

# UnitScale: the weights are the bilinear weights alone, which sum to 1 per voxel and view inside
# the detector.
@testitem "Voxel driven: unit scale" tags = [:voxeldriven] begin
  image = ImageGeom(; dims=(6, 6, 6))
  g = circular_cone_beam(; angles=range(0, 2pi, 5)[1:4], dso=30, dsd=60, nu=40, nv=40, du=1.0)
  op = VoxelDrivenProjector(image, g; eltype=Float64, scale=UnitScale())
  @test all(isapprox.(colsums(op), 4; rtol=1e-12))
end

# The per-view tables (src/operators/voxeldriven/projection.jl) give the hits of project_point of the
# geometry: the same rational function of the point, rearranged. In Float64, with coordinates up to
# about 100 and some 30 operations, the rounding is of order 1e-12 (in cells, and relative for the
# density and the magnification); the bounds leave a factor of 100. Runs (view_run, run_hit) add the
# step to the first voxel instead of projecting each one: the same bounds.
@testitem "Voxel driven: per-view tables" tags = [:voxeldriven] begin
  using CTOperators: KA, point_projection, project_point, view_run, run_hit
  using StaticArrays, Random
  Random.seed!(3)
  m(cols...) = reduce(hcat, cols)
  tilted = ConeBeam(m([-30.0, 0, 5], [-28, 3, 4]), m([30.0, 2, -1], [31, -1, 0]), m([0.1, 1.0, 0.2], [0.2, 0.9, -0.1]),
                    m([0.05, -0.1, 0.9], [0.0, 0.1, 1.1]); nu=20, nv=15)
  oblique = ParallelBeam(m([1.0, 0.2, 0.3], [0.8, -0.5, 0.1]), m([20.0, 1, 0], [-3, 18, 2]),
                         m([-0.1, 1.0, 0.1], [1.0, 0.3, 0.0]), m([0.1, 0.0, 1.2], [0.1, 0.2, 0.9]); nu=20, nv=15)
  # rays parallel to the detector: no hits
  flat_rays = ParallelBeam(m([0.0, 1.0, 0.0]), m([20.0, 0, 0]), m([0.0, 1.0, 0.0]), m([0.0, 0.0, 1.0]); nu=20, nv=15)
  geometries = (circular_cone_beam(; angles=range(0, 2pi, 7)[1:6], dso=40, dsd=80, nu=30, nv=20, du=1.3, dv=0.9,
                                   offset_u=1.5, offset_v=-0.7),
                helical_cone_beam(; angles=range(0, 4pi, 9), pitch=10, dso=40, dsd=80, nu=30, nv=20, du=1.3),
                tilted, parallel_beam(; angles=range(0, pi, 5), nu=30, nv=20, du=1.1), oblique, flat_rays,
                circular_cone_beam(; angles=range(0, 2pi, 5)[1:4], dso=40, dsd=80, nu=30, nv=20, du=1.3, detector=:arc))
  close_hits(a, b) = a.ok == b.ok && (!a.ok || (isapprox(a.cu, b.cu; atol=1e-10) && isapprox(a.cv, b.cv; atol=1e-10) &&
                     isapprox(a.density, b.density; rtol=1e-10) && isapprox(a.magnification, b.magnification; rtol=1e-10) &&
                     a.rowsize == b.rowsize))
  for g in geometries
    table = point_projection(g, KA.CPU())
    @test (table === g) == (g isa ArcConeBeam)
    step = SVector(0.7, 0.0, 0.0)
    for k in 1:nviews(g), _ in 1:20
      p = SVector{3}(30 .* (rand(3) .- 0.5))
      @test close_hits(project_point(table, p, k), project_point(g, p, k))
      run = view_run(table, p, step, k)
      @test all(r -> close_hits(run_hit(run, r), project_point(g, p + (r - 1) * step, k)), 1:5)
    end
    if g isa ConeBeam
      # behind the source: no hit
      s, c = g.source[:, 1], g.center[:, 1]
      p = SVector{3}(s + 0.5 * (s - c))
      @test !project_point(table, p, 1).ok && !project_point(g, p, 1).ok
    end
  end
  @test !project_point(point_projection(flat_rays, KA.CPU()), SVector(1.0, 2.0, 3.0), 1).ok
end

# Runs of voxels give the projections of one voxel per thread, also for boxes that are no multiple
# of the run (odd sizes, an image mask), with a ray mask, for 2D images and for every detector type.
@testitem "Voxel driven: runs of voxels" tags = [:voxeldriven] setup = [Projections] begin
  using LinearAlgebra, Random
  Random.seed!(4)
  cases = ((ImageGeom(MaskCircle(); dims=(13, 7, 5), deltas=(1.0, 1.1, 0.9)),
            circular_cone_beam(; angles=range(0, 2pi, 6)[1:5], dso=30, dsd=60, nu=17, nv=11, du=1.2)),
           (ImageGeom(; dims=(13, 7, 5)),
            circular_cone_beam(; angles=range(0, 2pi, 6)[1:5], dso=30, dsd=60, nu=17, nv=11, du=1.2, detector=:arc)),
           (ImageGeom(; dims=(11, 9, 4)), parallel_beam(; angles=range(0, pi, 4), nu=17, nv=11, du=1.0)),
           (ImageGeom(; dims=(21, 17)),
            circular_cone_beam(; angles=range(0, 2pi, 6)[1:5], dso=40, dsd=80, nu=31, nv=3, du=1.5)))
  for (image, g) in cases
    raymask = rand(datasize(g)...) .> 0.2
    ref = VoxelDrivenProjector(image, g; eltype=Float64, raymask, launch=LaunchConfig(; layout=(1,)))
    x, y = rand(imagesize(ref)...), rand(datasize(ref)...)
    for R in (2, 3, 8)
      op = VoxelDrivenProjector(image, g; eltype=Float64, raymask, launch=LaunchConfig(; layout=(R,)))
      @test isapprox(op * x, ref * x; rtol=1e-12)
      @test isapprox(adjoint(op) * y, adjoint(ref) * y; rtol=1e-12)
    end
  end
end
