# The sampling descriptions (src/geometry/sampling.jl) and the shifted geometries of their samples
# (src/geometry/shifted.jl).

@testitem "Sampling: disk points and offsets" tags = [:geometry, :sampling] begin
  using CTOperators: point_offset, varies_per_cell, sample_offsets, spot_offsets, FullCell, InPlane
  using LinearAlgebra
  # inside the disk of radius 1/2, centered (the angles of a ring are evenly spaced: their mean is 0
  # up to rounding), the same in every cell and view
  p = DiskPoints(3, 8)
  o = [point_offset(p, Float64, 1, 1, 1, k) for k in 1:24]
  @test all(d -> hypot(d...) < 0.5, o)
  @test norm(sum(collect.(o)) / 24) < 1e-15
  @test all(k -> point_offset(p, Float64, 4, 2, 9, k) == o[k], 1:24)
  # rings of equal area: the radii are sqrt((i - 1/2)/rings)/2
  @test isapprox(sort(unique(round.(hypot.(first.(o), last.(o)); digits=12))), [sqrt((i - 0.5) / 3) / 2 for i in 1:3])
  @test_throws ArgumentError DiskPoints(0)
  @test !varies_per_cell(RegularPoints(2)) && !varies_per_cell(p) && !varies_per_cell(Points([0.0 0.1; 0.0 0.0]))
  @test varies_per_cell(JitteredPoints(2)) && varies_per_cell(RandomPoints(3)) && varies_per_cell(GaussianPoints(3, 0.2))

  # weights sum to 1 in every view (a sum of K terms: rounding K eps); jittered offsets differ
  # between views, and depend on the view number given
  spot = FocalSpot(1.0; points=JitteredPoints(2, 2; seed=3), weights=GaussianWeights(0.3))
  offsets, weights = sample_offsets(spot, 1:5, FullCell())
  @test all(isapprox.(sum(weights; dims=2), 1; atol=4eps()))
  @test offsets[1][:, 1] != offsets[1][:, 2]
  @test sample_offsets(spot, 3:5, FullCell())[1][1] == offsets[1][:, 3:5]
  # an image with one layer: offsets along u only
  @test all(o -> all(iszero, o[2, :]), sample_offsets(FocalSpot(1.0; points=RandomPoints(4)), 1:5, InPlane())[1])

  # the spot of a flat detector lies in the plane of u and v and spans fu x fv
  g = circular_cone_beam(; angles=range(0, 2pi, 6)[1:5], dso=40, dsd=80, nu=24, nv=20, du=1.5, dv=1.2)
  spot = FocalSpot(0.8, 0.4; points=RegularPoints(2, 2))
  offsets, _ = sample_offsets(spot, 1:5, FullCell())
  deltas = [spot_offsets(spot, g, o) for o in offsets]
  for k in 1:5
    n = normalize(cross(g.u[:, k], g.v[:, k]))
    @test all(d -> abs(dot(d[:, k], n)) < 1e-15, deltas)
    span_u = maximum(d -> dot(d[:, k], normalize(g.u[:, k])), deltas) - minimum(d -> dot(d[:, k], normalize(g.u[:, k])), deltas)
    span_v = maximum(d -> dot(d[:, k], normalize(g.v[:, k])), deltas) - minimum(d -> dot(d[:, k], normalize(g.v[:, k])), deltas)
    @test isapprox(span_u, 0.8 / 2) && isapprox(span_v, 0.4 / 2)      # the points sit at the subcell centers, +-1/4
  end
  @test_throws ArgumentError FocalSpot(-1.0; points=RegularPoints(2))
  @test_throws UndefKeywordError FocalSpot(1.0)
end

# A shifted geometry: zero offsets give the geometry itself; source offsets move the start of the
# rays, not the detector; detector offsets move their end by whole multiples of u and v. On an arc,
# project_point of the shifted source finds where its ray through p meets the detector.
@testitem "Sampling: shifted geometries" tags = [:geometry, :sampling] begin
  using CTOperators: Shifted, shift, cell_ray, project_point
  using LinearAlgebra, StaticArrays, Adapt
  arc = circular_cone_beam(; angles=range(0, 2pi, 6)[1:5], dso=40, dsd=80, nu=24, nv=20, du=1.5, detector=:arc)
  flat = circular_cone_beam(; angles=range(0, 2pi, 6)[1:5], dso=40, dsd=80, nu=24, nv=20, du=1.5)
  zero_shift = Shifted(arc, zeros(3, 5), zeros(2, 5))
  @test all(i -> ray(zero_shift, i) == ray(arc, i), 1:nrays(arc))
  @test datasize(zero_shift) == datasize(arc) && raytype(zero_shift) == raytype(arc)
  delta, epsilon = randn(3, 5), randn(2, 5)
  for g in (arc, flat)
    s = shift(g; source=delta)
    @test (s isa Shifted) == (g isa ArcConeBeam)
    for k in 1:5, (cu, cv) in ((3.0, 4.0), (12.5, 10.0))
      a, b = cell_ray(s, cu, cv, k)
      a0, b0 = cell_ray(g, cu, cv, k)
      @test isapprox(a, a0 + delta[:, k]) && isapprox(b, b0)           # the detector does not move with the source
      a1, b1 = cell_ray(shift(g; detector=epsilon), cu, cv, k)
      a2, b2 = cell_ray(g, cu + epsilon[1, k], cv + epsilon[2, k], k)
      @test isapprox(a1, a2) && isapprox(b1, b2)
    end
  end
  @test shift(flat) === flat && shift(arc) === arc
  @test_throws ArgumentError shift(parallel_beam(; angles=range(0, pi, 4), nu=8, nv=4, du=1.0); source=delta[:, 1:4])
  # subset and adapt keep the offsets of the views
  s = Shifted(arc, delta, epsilon)
  @test all(i -> ray(subset(s, [2, 4]), i) == ray(subset(arc, [2, 4]) |> g -> Shifted(g, delta[:, [2, 4]], epsilon[:, [2, 4]]), i), 1:nrays(subset(arc, [2, 4])))
  s32 = adapt(Array{Float32}, s)
  @test s32 isa Shifted{Float32} && s32.source == Float32.(delta)

  # arc, source shifted: the ray from the shifted source through the hit passes through p, and the
  # magnification is the ratio of the distances. Coordinates up to about 100, some 40 operations of
  # relative rounding eps(T) each: bounds 100 * 40 * eps(T), and 1e-12 in Float64.
  for T in (Float64, Float32)
    st = adapt(Array{T}, Shifted(arc, 0.5 .* randn(3, 5), nothing))
    tol = T === Float64 ? 1e-10 : 100 * 40 * eps(T)
    for k in 1:5, _ in 1:20
      p = SVector{3,T}(10 .* (rand(3) .- 0.5))
      h = project_point(st, p, k)
      a, b = cell_ray(st, h.cu, h.cv, k)
      d = normalize(b - a)
      @test h.ok && norm((p - a) - dot(p - a, d) * d) < tol
      @test isapprox(h.magnification, norm(b - a) / norm(p - a); rtol=tol)
    end
  end
  # without offsets the same hit as the arc itself (two forms of the same formula: rounding)
  for k in 1:5, _ in 1:10
    p = SVector{3}(10 .* (rand(3) .- 0.5))
    h, r = project_point(Shifted(arc, zeros(3, 5), nothing), p, k), project_point(arc, p, k)
    @test isapprox(h.cu, r.cu; atol=1e-12) && isapprox(h.cv, r.cv; atol=1e-12) && isapprox(h.density, r.density; rtol=1e-12)
  end
end
