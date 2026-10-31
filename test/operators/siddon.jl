@testitem "Siddon: contract" tags = [:siddon, :gpu] setup = [Backends, Contract] begin
  contract_tests(SiddonProjector, arrayTypes, floatTypes)
end

@testitem "Siddon: operator interface" tags = [:siddon] begin
  using LinearAlgebra
  image = ImageGeom(; dims=(6, 5, 4))
  g = circular_cone_beam(; angles=range(0, 2pi, 4)[1:3], dso=20, dsd=40, nu=7, nv=5, du=1.0)
  op = SiddonProjector(image, g; eltype=Float64)
  @test size(op) == (7 * 5 * 3, 6 * 5 * 4) && size(op, 1) == 105 && size(op, 3) == 1
  @test imagesize(op) == (6, 5, 4) && datasize(op) == (7, 5, 3)
  @test image_geometry(op) === image && geometry(op) isa ConeBeam{Float64}
  @test adjoint(adjoint(op)) === op && transpose(op) isa AdjointCTOperator
  @test size(adjoint(op)) == reverse(size(op))

  # vectors in, vectors out (RegularizedLeastSquares); shaped arrays in, shaped arrays out
  x = rand(6, 5, 4)
  @test op * vec(x) == vec(op * x) && size(op * x) == datasize(op)
  y = op * x
  @test mul!(similar(vec(y)), op, vec(x)) == vec(y)
  @test adjoint(op) * vec(y) == vec(adjoint(op) * y)
  @test_throws DimensionMismatch op * rand(10)
  @test_throws DimensionMismatch mul!(rand(3), op, x)

  # element type and storage of the stored geometry
  op32 = SiddonProjector(image, g)
  @test eltype(op32) == Float32 && eltype(geometry(op32)) == Float32
  @test eltype(op32 * Float32.(x)) == Float32
end

# Siddon's weights are exact intersection lengths, so the projection of an object that is a union
# of voxels equals its analytic line integrals (ImagePhantoms) to rounding error.
@testitem "Siddon: exact for voxel-aligned cuboids" tags = [:siddon] setup = [Rays] begin
  using ImagePhantoms: cuboid, radon
  using ImageGeoms: axis
  for (image, block) in ((ImageGeom(; dims=(12, 10, 8), deltas=(1.0, 0.8, 1.25), offsets=(0.5, 0, 0)),
                          (3:9, 2:6, 4:7)),
                         (ImageGeom(; dims=(14, 12), deltas=(1.0, 0.75)), (4:11, 3:10, 1:1)))
    # the cuboid covering the voxels of the block, value 2
    lims = [extrema(axis(image, d)[block[d]]) for d in 1:length(image.dims)]
    deltas = [image.deltas...]
    if length(image.dims) == 2   # a 2D image is one voxel of thickness 1 at z = 0
      push!(lims, (0.0, 0.0)); push!(deltas, 1.0)
    end
    centre = Tuple((l[1] + l[2]) / 2 for l in lims)
    width = Tuple(l[2] - l[1] + abs(d) for (l, d) in zip(lims, deltas))
    ob = cuboid(centre, width, (0, 0, 0), 2.0)
    x = zeros(image.dims)
    x[block[1:length(image.dims)]...] .= 2

    geometries = Any[parallel_beam(; angles=range(0, pi, 9)[1:8], nu=21, nv=length(image.dims) == 3 ? 15 : 1,
                                du=0.7, dv=0.7, offset_u=0.3)]
    length(image.dims) == 3 &&
      push!(geometries, helical_cone_beam(; angles=range(0, 2pi, 7), pitch=6, dso=40, dsd=80, nu=25, nv=19, du=1.1))
    for g in geometries
      op = SiddonProjector(image, g; eltype=Float64)
      exact = [radon(ob)(line_coordinates(a, b)...) for (a, b) in allrays(g)]
      @test isapprox(vec(op * x), exact; atol=1e-10 * maximum(exact))
    end
  end
end

@testitem "Siddon: segments and ray directions" tags = [:siddon] begin
  using LinearAlgebra
  image = ImageGeom(; dims=(10, 10, 10), deltas=(1.0, 1.0, 1.0))
  ones3 = ones(10, 10, 10)
  # a segment inside the volume integrates its own length; one leaving it the part inside
  # (the volume is [-5, 5]^3)
  starts = [-2.0 1.0 0.3; 0.5 -3.0 0.2; 1.0 2.0 0.1]
  ends = [3.0 1.5 9.0; 2.0 4.0 0.2; -1.0 -2.5 0.1]
  op = SiddonProjector(image, RayList(starts, ends); eltype=Float64)
  y = op * ones3
  @test isapprox(y[1], norm(ends[:, 1] - starts[:, 1]))
  @test isapprox(y[2], norm(ends[:, 2] - starts[:, 2]))
  @test isapprox(y[3], 5.0 - 0.3)

  # reversing the rays does not change the projection
  x = rand(10, 10, 10)
  rays = RayList(rand(3, 50) .* 30 .- 15, rand(3, 50) .* 30 .- 15)
  reversed = RayList(rays.ends, rays.starts)
  @test isapprox(SiddonProjector(image, rays; eltype=Float64) * x, SiddonProjector(image, reversed; eltype=Float64) * x)

  # axis-aligned rays through voxel centres sum the column, times the voxel size
  g = parallel_beam(; angles=[0.0], nu=10, nv=10, du=1.0)   # rays along +y
  opy = SiddonProjector(ImageGeom(; dims=(10, 10, 10), deltas=(1.0, 2.0, 1.0)), g; eltype=Float64)
  @test isapprox(opy * x, 2 * dropdims(sum(x; dims=2); dims=2))
end

# a flipped axis (negative voxel size) with the image reversed along it is the same object (no ray
# in a plane between voxels, see the contract tests)
@testitem "Siddon: flipped axes" tags = [:siddon] begin
  g = helical_cone_beam(; angles=range(0, 2pi, 5), pitch=3, dso=30, dsd=60, nu=11, nv=9, du=1.3, offset_u=0.23)
  x = rand(8, 7, 6)
  op = SiddonProjector(ImageGeom(; dims=(8, 7, 6), deltas=(1.0, 1.0, 1.0)), g; eltype=Float64)
  flipped = SiddonProjector(ImageGeom(; dims=(8, 7, 6), deltas=(-1.0, 1.0, 1.0)), g; eltype=Float64)
  @test isapprox(flipped * reverse(x; dims=1), op * x)
end

# The projection of a voxelised ellipsoid approaches its analytic line integrals as the voxels
# shrink: the error is that of the voxel representation of the object's boundary (first order in
# the voxel size), not of Siddon's weights. Measured: relative errors 0.103, 0.049, 0.024 for
# n = 16, 32, 64 (oversampling 4).
@testitem "Siddon: smooth phantom converges" tags = [:siddon] setup = [Rays] begin
  using ImagePhantoms: ellipsoid, phantom, radon
  using ImageGeoms: axis
  ob = [ellipsoid((1.0, -0.5, 0.5), (6.0, 4.0, 5.0), (0.3, 0, 0), 1.0)]
  g = helical_cone_beam(; angles=range(0, 2pi, 5), pitch=4, dso=40, dsd=80, nu=31, nv=25, du=0.8)
  exact = [radon(ob)((line_coordinates(a, b))...) for (a, b) in allrays(g)]
  errors = map((16, 32)) do n
    image = ImageGeom(; dims=(n, n, n), deltas=(16 / n, 16 / n, 16 / n))
    x = phantom(axis(image, 1), axis(image, 2), axis(image, 3), ob, 4)
    norm(vec(SiddonProjector(image, g; eltype=Float64) * x) - exact) / norm(exact)
  end
  @test errors[1] < 0.15
  @test errors[2] < errors[1] / 1.8
end

@testitem "Siddon: normalisations" tags = [:siddon] begin
  image = ImageGeom(; dims=(9, 8, 7))
  g = helical_cone_beam(; angles=range(0, 2pi, 6), pitch=3, dso=25, dsd=50, nu=15, nv=11, du=1.0)
  op = SiddonProjector(image, g; eltype=Float64)
  r, c = rowsums(op), colsums(op)
  # both sum all weights; weights are lengths, never negative; rays that miss have none
  @test isapprox(sum(r), sum(c))
  @test all(>=(0), r) && all(>=(0), c)
  @test any(iszero, r) && maximum(r) <= sqrt(9^2 + 8^2 + 7^2) + 1e-12
end

@testitem "Siddon: subset and adapt" tags = [:siddon] begin
  using Adapt
  image = ImageGeom(; dims=(8, 8, 6))
  g = circular_cone_beam(; angles=range(0, 2pi, 9)[1:8], dso=25, dsd=50, nu=12, nv=9, du=1.0)
  op = SiddonProjector(image, g; eltype=Float64)
  x = rand(8, 8, 6)
  views = [2, 5, 7]
  # the projector of a subset of views computes those views of the full projection
  @test isapprox(subset(op, views) * x, (op * x)[:, :, views])
  op32 = adapt(Array{Float32}, op)
  @test op32 isa SiddonProjector{Float32} && image_geometry(op32) === image
  @test_throws MethodError subset(SiddonProjector(image, RayList(rand(3, 4), rand(3, 4))), [1])
end

# Every ray in Float32 agrees with Float64 to Float32 accuracy. Regression: an entry point within
# rounding of a plane between voxels put the start voxel and the plane counter one apart, shifting
# all later voxels of that axis (errors of 10-30 % on single rays, invisible in the norm over all
# rays). The rays enter through the face x = min at heights y, z exactly on planes between voxels,
# with large voxel coordinates (in Float32 they land within rounding of the plane on either side).
@testitem "Siddon: Float32 rays entering on planes" tags = [:siddon] begin
  using ImageGeoms: axis
  using Random
  Random.seed!(1)
  image = ImageGeom(; dims=(64, 128, 128), offsets=(0.37, 0.11, -0.23))
  planes(d) = axis(image, d)[1:end-1] .+ 0.5   # world positions of the planes between voxels
  xmin = axis(image, 1)[1] - 0.5
  nr = 2000
  starts = zeros(3, nr); ends = zeros(3, nr)
  for r in 1:nr
    entry = [xmin, rand(planes(2)), rand(planes(3))]
    dir = [1.0, 2rand() - 1, 2rand() - 1]
    starts[:, r] = entry - 50dir
    ends[:, r] = entry + 300dir
  end
  g = RayList(starts, ends)
  x = rand(64, 128, 128)
  y64 = SiddonProjector(image, g; eltype=Float64) * x
  y32 = SiddonProjector(image, g; eltype=Float32) * Float32.(x)
  @test maximum(abs.(y32 .- y64)) < 1e-4 * maximum(y64)
end
