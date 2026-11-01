@testitem "Distance driven: contract" tags = [:distancedriven, :gpu] setup = [Backends, Contract] begin
  contract_tests(DistanceDrivenProjector, arrayTypes, floatTypes; raylists=false)
end

@testitem "Distance driven: operator interface" tags = [:distancedriven] begin
  using Adapt
  image = ImageGeom(; dims=(8, 8, 6))
  g = circular_cone_beam(; angles=range(0, 2pi, 9)[1:8], dso=25, dsd=50, nu=12, nv=9, du=1.0)
  op = DistanceDrivenProjector(image, g; eltype=Float64)
  @test size(op) == (12 * 9 * 8, 8 * 8 * 6) && datasize(op) == (12, 9, 8)
  @test image_geometry(op) === image && geometry(op) isa ConeBeam{Float64}
  x = rand(8, 8, 6)
  @test op * vec(x) == vec(op * x)
  @test isapprox(subset(op, [2, 5]) * x, (op * x)[:, :, [2, 5]])
  op32 = adapt(Array{Float32}, op)
  @test op32 isa DistanceDrivenProjector{Float32} && image_geometry(op32) === image
  # ray lists have no detector cells
  @test_throws ArgumentError DistanceDrivenProjector(image, RayList(g))
end

# With detector cells that coincide with the voxels across the rays, the footprint of a cell is its
# voxel in every layer: the mean over the footprint is the voxel value, and the method equals
# Siddon's, in 3D and in 2D.
@testitem "Distance driven: cells aligned with the voxels" tags = [:distancedriven] begin
  image = ImageGeom(; dims=(8, 8, 6), deltas=(1.0, 1.0, 1.0))
  g = parallel_beam(; angles=[0.0, pi / 2, pi], nu=8, nv=6, du=1.0)
  x = rand(8, 8, 6)
  op = DistanceDrivenProjector(image, g; eltype=Float64)
  @test isapprox(op * x, SiddonProjector(image, g; eltype=Float64) * x; rtol=1e-12)
  image2 = ImageGeom(; dims=(8, 8), deltas=(1.0, 1.0))
  g2 = parallel_beam(; angles=[0.0, pi / 2], nu=8, nv=1, du=1.0)
  x2 = rand(8, 8)
  @test isapprox(DistanceDrivenProjector(image2, g2; eltype=Float64) * x2,
                 SiddonProjector(image2, g2; eltype=Float64) * x2; rtol=1e-12)
end

# The footprints of the cells of a view tile every layer, so with one cell per pixel every pixel
# inside the volume is covered exactly once, at every angle (Siddon: 0.83 to 1.34 at 45 degrees).
@testitem "Distance driven: coverage" tags = [:distancedriven] begin
  n = 32
  image = ImageGeom(; dims=(n, n), deltas=(1.0, 1.0))
  inner = 5:n-4
  for angle in (pi / 4, 0.3, 0.1, 2.0)
    g = parallel_beam(; angles=[angle], nu=2n, nv=1, du=1.0)
    c = colsums(DistanceDrivenProjector(image, g; eltype=Float64))[inner, inner]
    @test all(isapprox.(c, 1; atol=1e-12))
  end
  # cone beam: the coverage varies smoothly with the magnification across the volume, without the
  # jumps between neighboring voxels of ray-driven methods (measured: largest jump 2 % of the
  # maximum, Siddon 18 %, Joseph 8 %)
  image3 = ImageGeom(; dims=(16, 16, 16), deltas=(1.0, 1.0, 1.0))
  g = circular_cone_beam(; angles=[0.4], dso=40, dsd=80, nu=64, nv=64, du=1.0)
  jump(c) = maximum(abs.(diff(c; dims=1))) / maximum(c)
  c = colsums(DistanceDrivenProjector(image3, g; eltype=Float64))[4:13, 4:13, 4:13]
  cs = colsums(SiddonProjector(image3, g; eltype=Float64))[4:13, 4:13, 4:13]
  @test jump(c) < 0.04 && jump(c) < jump(cs) / 4
end

# A constant image: every footprint inside the volume has mean 1, so the projection is the length
# of the central ray inside the volume, as Siddon's for rays that do not graze the boundary.
@testitem "Distance driven: constant image" tags = [:distancedriven] begin
  image = ImageGeom(; dims=(10, 10, 10), deltas=(1.0, 1.0, 1.0))
  g = circular_cone_beam(; angles=range(0, 2pi, 5)[1:4], dso=40, dsd=80, nu=6, nv=6, du=1.0)
  dd = rowsums(DistanceDrivenProjector(image, g; eltype=Float64))
  si = rowsums(SiddonProjector(image, g; eltype=Float64))
  @test isapprox(dd, si; rtol=1e-12)
end
