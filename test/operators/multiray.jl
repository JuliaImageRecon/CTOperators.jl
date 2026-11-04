# Projectors on a MultiRay average the line integrals of the subrays of each detector cell with
# normalized weights, with the exact adjoint, on every device (random positions from a hash, so the
# same everywhere).

@testsnippet MultiRayProblem begin
  using LinearAlgebra
  # a small volume, detector cells larger than the voxels
  image = ImageGeom(; dims=(12, 10, 6), deltas=(1.0, 1.1, 0.9), offsets=(0.2, -0.1, 0.3))
  cone(; nu=6, nv=3, du=2.6, dv=2.4) =
    circular_cone_beam(; angles=range(0, 2pi, 6)[1:5], dso=40, dsd=70, nu, nv, du, dv)
  parallel(; nu=6, nv=3, du=2.2, dv=2.0) = parallel_beam(; angles=range(0, pi, 6)[1:5], nu, nv, du, dv)
  arc(; nu=6, nv=3, du=2.6, dv=2.4) =
    circular_cone_beam(; angles=range(0, 2pi, 6)[1:5], dso=40, dsd=70, nu, nv, du, dv, detector=:arc)
  # the mean over blocks of su x sv cells of data of a detector of su nu x sv nv cells
  function block_mean(y, su, sv)
    nu, nv, nk = div(size(y, 1), su), div(size(y, 2), sv), size(y, 3)
    dropdims(sum(reshape(y, su, nu, sv, nv, nk); dims=(1, 3)); dims=(1, 3)) ./ (su * sv)
  end
end

@testitem "MultiRay: regular points are a finer detector" tags = [:operators, :multiray] setup = [MultiRayProblem] begin
  # the subcell centers of RegularPoints(su, sv) are the cell centers of a detector with su x sv
  # times the cells, so with uniform weights the projection is the block mean of that detector's
  # (on the arc detector the angle about the source is linear in the position, so also there)
  su, sv = 3, 2
  for (coarse, fine) in ((cone(), cone(; nu=18, nv=6, du=2.6 / 3, dv=1.2)),
                         (arc(), arc(; nu=18, nv=6, du=2.6 / 3, dv=1.2)),
                         (parallel(), parallel(; nu=18, nv=6, du=2.2 / 3, dv=1.0))),
      P in (SiddonProjector, JosephProjector)
    op = P(image, MultiRay(coarse, RegularPoints(su, sv)); eltype=Float64)
    x = rand(image.dims...)
    @test isapprox(op * x, block_mean(P(image, fine; eltype=Float64) * x, su, sv))
  end
end

@testitem "MultiRay: weights" tags = [:operators, :multiray] setup = [MultiRayProblem] begin
  x = rand(image.dims...)
  for P in (SiddonProjector, JosephProjector)
    plain = P(image, cone(); eltype=Float64)
    # one point at the center (any weight) is the plain projector
    @test isapprox(P(image, MultiRay(cone(), Points([0.0; 0.0;;]); weights=GaussianWeights(0.2)); eltype=Float64) * x, plain * x)
    # weights are normalized: equal points with Gaussian weights give the plain projection
    @test isapprox(P(image, MultiRay(cone(), Points([0.0 0.0; 0.0 0.0]); weights=GaussianWeights(0.3)); eltype=Float64) * x, plain * x)
    # Gaussian weights with a large width are uniform
    uniform = P(image, MultiRay(cone(), JitteredPoints(2, 2; seed=5)); eltype=Float64)
    wide = P(image, MultiRay(cone(), JitteredPoints(2, 2; seed=5); weights=GaussianWeights(1e4)); eltype=Float64)
    @test isapprox(wide * x, uniform * x)
  end
end

@testitem "MultiRay: adjoint and devices" tags = [:operators, :multiray, :gpu] setup = [Backends, MultiRayProblem] begin
  using Adapt
  for P in (SiddonProjector, JosephProjector),
      (g, pts) in ((cone(), JitteredPoints(2, 3; seed=1)), (arc(), RegularPoints(2)), (parallel(), RandomPoints(5; seed=2)), (cone(), GaussianPoints(4, 0.3; seed=6))),
      arrayType in arrayTypes, T in floatTypes
    @testset "$P, $(nameof(typeof(pts))), $arrayType{$T}" begin
      m = MultiRay(g, pts; weights=GaussianWeights(0.4))
      op = P(image, m; eltype=T, storage=arrayType)
      ref = P(image, m; eltype=Float64)
      x, y = rand(image.dims...), rand(datasize(g)...)
      # the same subrays on every device and in both precisions
      @test isapprox(Array(op * adapt(arrayType{T}, x)), ref * x; rtol=sqrt(eps(T)))
      @test isapprox(Array(adjoint(op) * adapt(arrayType{T}, y)), adjoint(ref) * y; rtol=sqrt(eps(T)))
      xd, yd = adapt(arrayType{T}, rand(T, image.dims...)), adapt(arrayType{T}, rand(T, datasize(g)...))
      @test isapprox(sum(vec(op * xd) .* vec(yd)), sum(vec(xd) .* vec(adjoint(op) * yd)); rtol=100eps(T))
    end
  end
end

@testitem "MultiRay: subset and ray mask" tags = [:operators, :multiray] setup = [MultiRayProblem] begin
  using CTOperators: ray_mask
  g = cone()
  raymask = rand(datasize(g)...) .> 0.3
  op = JosephProjector(image, MultiRay(g, JitteredPoints(2; seed=9)); eltype=Float64, raymask)
  x = rand(image.dims...)
  y = op * x
  # a subset of views gives the same rows (same random positions)
  @test isapprox(subset(op, [2, 4]) * x, y[:, :, [2, 4]])
  # masked data elements skip all their subrays
  @test all(iszero, y[.!raymask])
  # the default launch tiles the detector also with several rays per cell; results do not depend on it
  tiled = JosephProjector(image, MultiRay(g, JitteredPoints(2; seed=9)); eltype=Float64, raymask,
                          launch=LaunchConfig(; tile=(4, 2)))
  @test isapprox(tiled * x, y)
end

# A 2D image is one voxel layer along z: the projectors keep the subrays in the image plane (offsets
# along u only), for flat and arc detectors; 3D images keep offsets along v
@testitem "MultiRay: 2D images" tags = [:operators, :multiray] setup = [MultiRayProblem] begin
  using CTOperators: subray, cell_ray, pixel_index, point_offset, InPlane, FullCell
  image2 = ImageGeom(; dims=(24, 20), deltas=(1.0, 1.0))
  fan(detector) =
    circular_cone_beam(; angles=range(0, 2pi, 9)[1:8], dso=40, dsd=70, nu=10, nv=1, du=3.0, dv=3.0, detector)
  pts = JitteredPoints(4, 1; seed=3)
  for g in (fan(:flat), fan(:arc), parallel_beam(; angles=range(0, pi, 9)[1:8], nu=10, nv=1, du=3.0, dv=3.0))
    m = MultiRay(g, pts)
    s = geometry(SiddonProjector(image2, m; eltype=Float64))
    @test s.span isa InPlane && geometry(SiddonProjector(image, m; eltype=Float64)).span isa FullCell
    # every subray ends at its jittered position along u, in the row of the cell centers
    for i in 1:nrays(g), k in 1:4
      iu, iv, view = pixel_index(i, 10, 1)
      du, _ = point_offset(pts, Float64, iu, iv, view, k)
      a, b, _ = subray(s, i, k)
      @test all(isapprox.((a, b), cell_ray(g, iu + du, 1.0, view)))
    end
  end
  # the projection is the mean of the projections of the subrays as single rays (a ray list): the
  # in-plane subrays see the image like any ray in the plane
  x = rand(24, 20)
  for g in (fan(:flat), fan(:arc)), P in (SiddonProjector, JosephProjector)
    s = geometry(P(image2, MultiRay(g, pts); eltype=Float64))
    rays = [subray(s, i, k) for k in 1:4, i in 1:nrays(g)]
    list = RayList(stack(vec([r[1] for r in rays])), stack(vec([r[2] for r in rays])); datasize=(4, nrays(g)))
    expected = reshape(sum(P(image2, list; eltype=Float64) * x; dims=1) ./ 4, datasize(g))
    @test isapprox(P(image2, MultiRay(g, pts); eltype=Float64) * x, expected)
  end
  # subcells along v would collapse onto the plane
  @test_throws ArgumentError SiddonProjector(image2, MultiRay(fan(:flat), RegularPoints(2, 2)))
  # a 3D image of one layer is treated like a 2D image
  layer = ImageGeom(; dims=(24, 20, 1), deltas=(1.0, 1.0, 1.0))
  @test geometry(SiddonProjector(layer, MultiRay(fan(:flat), pts))).span isa InPlane
  @test isapprox(JosephProjector(layer, MultiRay(fan(:flat), pts); eltype=Float64) * reshape(x, 24, 20, 1),
                 JosephProjector(image2, MultiRay(fan(:flat), pts); eltype=Float64) * x)
end

# Strategies are open: types of other packages only subtype AbstractPoints and AbstractWeights and
# add a method each
@testitem "MultiRay: own strategies" tags = [:operators, :multiray] setup = [MultiRayProblem] begin
  struct Cross{K} <: AbstractPoints{K}
    reach::Float64
  end
  CTOperators.point_offset(p::Cross, ::Type{T}, iu, iv, view, k) where {T} =
    isodd(k) ? (T(p.reach) * (k - 2), zero(T)) : (zero(T), T(p.reach) * (k - 3))
  struct Linear <: AbstractWeights end
  CTOperators.subray_weight(::Linear, du, dv) = 1 - abs(du) - abs(dv)

  g = cone()
  m = MultiRay(g, Cross{4}(0.25); weights=Linear())
  equal = MultiRay(g, Points([-0.25 0.0 0.25 0.0; 0.0 -0.25 0.0 0.25]))
  x, y = rand(image.dims...), rand(datasize(g)...)
  for P in (SiddonProjector, JosephProjector)
    op = P(image, m; eltype=Float64)
    # the weights are equal at equal distance from the center: the same as given points
    @test isapprox(op * x, P(image, equal; eltype=Float64) * x)
    @test isapprox(sum((op * x) .* y), sum(x .* (adjoint(op) * y)))
  end
  @test_throws MethodError MultiRay(g, [0.0; 0.0;;])
end
