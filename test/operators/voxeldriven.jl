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
    # measured: 0.027 (cells of one voxel at the center), 0.23 (half a voxel)
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
    @test norm(yv - yd) / norm(yd) < 0.03   # measured 0.008 (fan), 0.016 (parallel)
  end
end

# UnitScale: the weights are the bilinear weights alone, which sum to 1 per voxel and view inside
# the detector.
@testitem "Voxel driven: unit scale" tags = [:voxeldriven] begin
  image = ImageGeom(; dims=(6, 6, 6))
  g = circular_cone_beam(; angles=range(0, 2pi, 5)[1:4], dso=30, dsd=60, nu=40, nv=40, du=1.0)
  op = VoxelDrivenProjector(image, g; eltype=Float64, scale=UnitScale())
  @test all(isapprox.(colsums(op), 4; rtol=1e-12))
end
