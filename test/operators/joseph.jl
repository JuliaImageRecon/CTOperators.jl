@testitem "Joseph: contract" tags = [:joseph, :gpu] setup = [Backends, Contract] begin
  contract_tests(JosephProjector, arrayTypes, floatTypes)
end

@testitem "Joseph: operator interface" tags = [:joseph] begin
  using Adapt
  image = ImageGeom(; dims=(8, 8, 6))
  g = circular_cone_beam(; angles=range(0, 2pi, 9)[1:8], dso=25, dsd=50, nu=12, nv=9, du=1.0)
  op = JosephProjector(image, g; eltype=Float64)
  @test size(op) == (12 * 9 * 8, 8 * 8 * 6) && datasize(op) == (12, 9, 8)
  @test image_geometry(op) === image && geometry(op) isa ConeBeam{Float64}
  x = rand(8, 8, 6)
  @test op * vec(x) == vec(op * x)
  @test isapprox(subset(op, [2, 5]) * x, (op * x)[:, :, [2, 5]])
  op32 = adapt(Array{Float32}, op)
  @test op32 isa JosephProjector{Float32} && image_geometry(op32) === image
end

# Joseph's sum is the midpoint rule on every layer of voxels with bilinear interpolation across the
# ray, so it is exact for an image that is a linear function of the position, along rays that cross
# the whole volume and stay within the voxel centres across the principal axis.
@testitem "Joseph: exact for linear images" tags = [:joseph] begin
  using LinearAlgebra, Random
  using ImageGeoms: axis
  Random.seed!(1)
  image = ImageGeom(; dims=(20, 18, 16), deltas=(1.0, 1.25, 0.8), offsets=(0.3, -0.1, 0.2))
  xs, ys, zs = axis(image, 1), axis(image, 2), axis(image, 3)
  f(p) = 1 + 0.3p[1] - 0.2p[2] + 0.5p[3]
  x = [f((a, b, c)) for a in xs, b in ys, c in zs]
  nr = 100
  starts = zeros(3, nr); ends = zeros(3, nr); exact = zeros(nr)
  inside(r) = r[2] + rand() * (r[end-1] - r[2])
  for r in 1:nr
    # through the faces x = min and x = max, principal axis x
    a = [xs[1] - 0.5, inside(ys), inside(zs)]
    b = [xs[end] + 0.5, inside(ys), inside(zs)]
    starts[:, r] = a - 3 * (b - a); ends[:, r] = b + 3 * (b - a)
    exact[r] = norm(b - a) * f((a + b) / 2)
  end
  op = JosephProjector(image, RayList(starts, ends); eltype=Float64)
  @test isapprox(op * x, exact; rtol=1e-12)
  # reversed rays give the same projection
  @test isapprox(JosephProjector(image, RayList(ends, starts); eltype=Float64) * x, exact; rtol=1e-12)
end

@testitem "Joseph: segments and ray directions" tags = [:joseph] begin
  using LinearAlgebra
  image = ImageGeom(; dims=(10, 10, 10), deltas=(1.0, 1.0, 1.0))
  ones3 = ones(10, 10, 10)
  # in a constant image, segments within the voxel centres integrate their own length (partial
  # first and last layers), a segment leaving the volume along its principal axis the part inside
  starts = [-2.0 1.0 0.3; 0.5 -3.0 0.2; 1.0 2.0 0.1]
  ends = [3.0 1.5 9.0; 2.0 4.0 0.2; -1.0 -2.5 0.1]
  y = JosephProjector(image, RayList(starts, ends); eltype=Float64) * ones3
  @test isapprox(y[1], norm(ends[:, 1] - starts[:, 1]))
  @test isapprox(y[2], norm(ends[:, 2] - starts[:, 2]))
  @test isapprox(y[3], 5.0 - 0.3)

  # axis-aligned rays through voxel centres sum the column, times the voxel size
  x = rand(10, 10, 10)
  g = parallel_beam(; angles=[0.0], nu=10, nv=10, du=1.0)   # rays along +y
  op = JosephProjector(ImageGeom(; dims=(10, 10, 10), deltas=(1.0, 2.0, 1.0)), g; eltype=Float64)
  @test isapprox(op * x, 2 * dropdims(sum(x; dims=2); dims=2))

  # a ray between two voxel centres takes their mean, and its integral varies continuously with
  # its position (Siddon's jumps between the voxels)
  shifted = parallel_beam(; angles=[0.0], nu=10, nv=10, du=1.0, offset_u=0.5)
  y = JosephProjector(ImageGeom(; dims=(10, 10, 10)), shifted; eltype=Float64) * x
  @test isapprox(y[2, 3], (sum(x[1, :, 3]) + sum(x[2, :, 3])) / 2)
end

# Away from 45 degrees the rays of a view cover the voxels more evenly than Siddon's (at 45 degrees
# both coincide). Measured spread (max - min) of the column sums of one view at 0.3 rad, one ray per
# pixel, inner pixels of 64^2: Siddon 15.8 %, Joseph 4.7 %.
@testitem "Joseph: coverage" tags = [:joseph] begin
  n = 64
  image = ImageGeom(; dims=(n, n), deltas=(1.0, 1.0))
  g = parallel_beam(; angles=[0.3], nu=2n, nv=1, du=1.0)
  spread(c) = (maximum(c) - minimum(c)) / (sum(c) / length(c))
  inner = 20:44
  cj = colsums(JosephProjector(image, g; eltype=Float64))[inner, inner]
  cs = colsums(SiddonProjector(image, g; eltype=Float64))[inner, inner]
  @test spread(cj) < 0.06 && spread(cj) < spread(cs) / 3
end
