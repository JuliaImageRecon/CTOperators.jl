# MultiRay (src/geometry/multiray.jl): the subrays of a detector cell end inside the cell at the
# positions of the point strategy; random positions depend only on seed, cell, view (numbered as in
# the full geometry) and subray, so subsets and other element types draw the same ones.

@testitem "MultiRay: points and subrays" tags = [:geometry, :multiray] begin
  using CTOperators: subray, nsubrays, point_offset, pixel_index
  g = circular_cone_beam(; angles=range(0, 2pi, 6)[1:5], dso=50, dsd=100, nu=7, nv=3, du=1.5, dv=2.0)
  # one subray at the cell center is the ray of the cell, with weight 1
  m = MultiRay(g, RegularPoints(1))
  @test nsubrays(m) == 1 && datasize(m) == datasize(g)
  @test all(subray(m, i, 1) == (ray(g, i)..., 1.0) for i in 1:nrays(g))
  # offsets: regular points at the subcell centers, jittered ones inside their subcell, random ones
  # inside the cell
  su, sv = 3, 2
  for i in 1:nrays(g)
    iu, iv, view = pixel_index(i, 7, 3)
    for k in 1:su*sv
      ju, jv = rem(k - 1, su), div(k - 1, su)
      @test all(isapprox.(point_offset(RegularPoints(su, sv), Float64, iu, iv, view, k), ((ju + 0.5) / su - 0.5, (jv + 0.5) / sv - 0.5)))
      du, dv = point_offset(JitteredPoints(su, sv; seed=7), Float64, iu, iv, view, k)
      @test ju / su - 0.5 <= du < (ju + 1) / su - 0.5 && jv / sv - 0.5 <= dv < (jv + 1) / sv - 0.5
      du, dv = point_offset(RandomPoints(su * sv; seed=7), Float64, iu, iv, view, k)
      @test -0.5 <= du < 0.5 && -0.5 <= dv < 0.5
    end
  end
  # different seeds, cells and subrays give different positions
  offsets = [point_offset(RandomPoints(4; seed=s), Float64, iu, 1, 1, k) for s in 0:1, iu in 1:7, k in 1:4]
  @test allunique(offsets)
  # given points; Gaussian weights fall off from the center
  m = MultiRay(g, Points([0.0 0.25; 0.0 0.0]); weights=GaussianWeights(0.5))
  @test subray(m, 1, 1)[3] == 1 && isapprox(subray(m, 1, 2)[3], exp(-0.125))
end

# Gaussian points: normally distributed around the cell center with the given widths, identical
# rays counting equally; the seed fixes them
@testitem "MultiRay: Gaussian points" tags = [:geometry, :multiray] begin
  using Adapt
  using CTOperators: point_offset
  mean(x) = sum(x) / length(x)
  std(x) = sqrt(mean((x .- mean(x)) .^ 2))
  sigma_u, sigma_v = 0.3, 0.6
  p = GaussianPoints(16, sigma_u, sigma_v; seed=4)
  offsets = [point_offset(p, Float64, iu, iv, 1, k) for iu in 1:200, iv in 1:200, k in 1:16]
  du, dv = vec(first.(offsets)), vec(last.(offsets))
  N = length(du)   # 640000 samples: tolerances of 5 standard errors
  @test abs(mean(du)) < 5 * sigma_u / sqrt(N) && abs(mean(dv)) < 5 * sigma_v / sqrt(N)
  @test isapprox(std(du), sigma_u; rtol=5 / sqrt(2N)) && isapprox(std(dv), sigma_v; rtol=5 / sqrt(2N))
  # fraction of a normal distribution within one standard deviation
  @test isapprox(count(abs.(du) .< sigma_u) / N, 0.6827; atol=0.002)
  @test abs(mean(du .* dv)) < 5 * sigma_u * sigma_v / sqrt(N)   # the two axes are independent
  @test all(isfinite, du) && all(isfinite, dv)
  @test point_offset(p, Float64, 3, 2, 1, 5) == point_offset(GaussianPoints(16, sigma_u, sigma_v; seed=4), Float64, 3, 2, 1, 5)
  @test point_offset(p, Float64, 3, 2, 1, 5) != point_offset(GaussianPoints(16, sigma_u, sigma_v; seed=5), Float64, 3, 2, 1, 5)
  g = circular_cone_beam(; angles=range(0, 2pi, 5)[1:4], dso=50, dsd=100, nu=5, nv=3, du=1.5)
  @test adapt(Array{Float32}, MultiRay(g, GaussianPoints(4, 0.3))).points isa GaussianPoints{4,Float32}
  @test_throws ArgumentError GaussianPoints(4, -0.1)
  @test_throws ArgumentError GaussianPoints(0, 0.1)
end

@testitem "MultiRay: subset, adapt, errors" tags = [:geometry, :multiray] begin
  using Adapt
  using CTOperators: subray
  g = helical_cone_beam(; angles=range(0, 4pi, 9), pitch=10, dso=50, dsd=100, nu=5, nv=4, du=1.5)
  m = MultiRay(g, JitteredPoints(2, 3; seed=11); weights=GaussianWeights(0.3, 0.6))
  # the subrays of a subset (also of a subset of a subset) are those of the same views of m
  views = [2, 5, 9]
  s = subset(subset(m, [1, 2, 5, 8, 9]), [2, 3, 5])
  li, ls = LinearIndices(datasize(m)), LinearIndices(datasize(s))
  @test all(all(isapprox.(subray(s, ls[iu, iv, j], k), subray(m, li[iu, iv, views[j]], k)))
            for iu in 1:5, iv in 1:4, j in 1:3, k in 1:6)
  # adapt to Float32 keeps points, weights and the view numbers
  s32 = adapt(Array{Float32}, s)
  @test eltype(s32) == Float32 && s32.weights isa GaussianWeights{Float32}
  @test all(isapprox.(subray(s32, 7, 4), subray(s, 7, 4); rtol=1e-5))
  # ray lists have no cells; a MultiRay has no single ray per element
  @test_throws ArgumentError MultiRay(RayList(g), RegularPoints(2))
  @test_throws MethodError ray(m, 1)
  @test_throws ArgumentError RegularPoints(0, 2)
  @test_throws ArgumentError RandomPoints(0)
  @test_throws ArgumentError GaussianWeights(0.5, -1.0)
  @test_throws DimensionMismatch Points(zeros(3, 2))
end
