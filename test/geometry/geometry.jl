@testitem "RayList" tags = [:geometry] setup = [Rays] begin
  starts, ends = rand(3, 6), rand(3, 6)
  g = RayList(starts, ends; datasize=(2, 3))
  @test datasize(g) == (2, 3) && nrays(g) == 6 && raytype(g) == RaySegment()
  @test all(ray(g, i) == (starts[:, i], ends[:, i]) for i in 1:6)
  @test_throws DimensionMismatch RayList(starts, ends; datasize=(4,))
  @test_throws DimensionMismatch RayList(rand(2, 6), rand(2, 6))
  @test_throws DimensionMismatch RayList(starts, rand(3, 5))

  # the rays of a segment geometry as a list
  c = circular_cone_beam(; angles=range(0, 2pi, 5), dso=500, dsd=900, nu=4, nv=3, du=1.0)
  r = RayList(c)
  @test datasize(r) == datasize(c)
  @test allrays(r) == allrays(c)
  @test_throws ArgumentError RayList(parallel_beam(; angles=[0.0], nu=2, nv=2, du=1.0))
end

@testitem "circular cone beam" tags = [:geometry] setup = [Rays] begin
  angles = range(0, 2pi, 9)[1:8]
  dso, dsd, nu, nv, du, dv = 500.0, 900.0, 7, 5, 1.5, 2.0
  offset_u, offset_v = 0.25, -0.5
  g = circular_cone_beam(; angles, dso, dsd, nu, nv, du, dv, offset_u, offset_v)
  @test datasize(g) == (nu, nv, 8) && nviews(g) == 8 && detectorsize(g) == (nu, nv)
  @test raytype(g) == RaySegment()

  # source on the circle at angle β counter-clockwise from the y axis; ray ends on the detector
  # plane at distance dsd − dso beyond the axis, at the cell positions of Sinograms' convention
  us = ((0:nu-1) .- (nu - 1) / 2 .- offset_u) .* du
  vs = ((0:nv-1) .- (nv - 1) / 2 .- offset_v) .* dv
  rays = reshape(allrays(g), nu, nv, :)
  for (k, beta) in enumerate(angles), iv in 1:nv, iu in 1:nu
    a, b = rays[iu, iv, k]
    @test isapprox(a, SVector(-dso * sin(beta), dso * cos(beta), 0))
    @test isapprox(b, SVector((dsd - dso) * sin(beta), -(dsd - dso) * cos(beta), 0) +
                      us[iu] * SVector(cos(beta), sin(beta), 0) + vs[iv] * SVector(0, 0, 1))
  end

  # without offsets and with an odd detector the central ray crosses the rotation axis
  g0 = circular_cone_beam(; angles, dso, dsd, nu, nv, du)
  for k in eachindex(angles)
    a, b = ray(g0, LinearIndices((nu, nv, length(angles)))[div(nu + 1, 2), div(nv + 1, 2), k])
    @test line_distance(a, b) < 1e-12 * dsd
  end

  @test_throws ArgumentError circular_cone_beam(; angles, dso=900, dsd=500, nu, nv, du)
  @test_throws DimensionMismatch circular_cone_beam(; angles, dso, dsd, nu, nv, du, source_z=zeros(3))
end

# Arc detector: the cells lie on a cylinder of radius dsd about the source, equally spaced in fan
# angle (du / dsd per cell, offset along the arc), at the heights of the flat detector.
@testitem "circular cone beam, arc detector" tags = [:geometry] setup = [Rays] begin
  angles = range(0, 2pi, 9)[1:8]
  dso, dsd, nu, nv, du, dv = 500.0, 900.0, 7, 5, 1.5, 2.0
  offset_u, offset_v = 0.25, -0.5
  g = circular_cone_beam(; angles, dso, dsd, nu, nv, du, dv, offset_u, offset_v, detector=:arc)
  @test g isa ArcConeBeam && datasize(g) == (nu, nv, 8) && raytype(g) == RaySegment()
  gammas = ((0:nu-1) .- (nu - 1) / 2 .- offset_u) .* du ./ dsd
  vs = ((0:nv-1) .- (nv - 1) / 2 .- offset_v) .* dv
  rays = reshape(allrays(g), nu, nv, :)
  for (k, beta) in enumerate(angles), iv in 1:nv, iu in 1:nu
    a, b = rays[iu, iv, k]
    towards_axis = SVector(sin(beta), -cos(beta), 0)
    across = SVector(cos(beta), sin(beta), 0)
    @test isapprox(a, SVector(-dso * sin(beta), dso * cos(beta), 0))
    @test isapprox(b, a + dsd * (cos(gammas[iu]) * towards_axis + sin(gammas[iu]) * across) +
                      vs[iv] * SVector(0, 0, 1))
  end
  # the central ray of an odd detector without offsets crosses the rotation axis
  g0 = circular_cone_beam(; angles, dso, dsd, nu, nv, du, detector=:arc)
  a, b = ray(g0, LinearIndices((nu, nv, 8))[4, 3, 2])
  @test line_distance(a, b) < 1e-12 * dsd
  @test subset(g, 2:3) isa ArcConeBeam && nviews(subset(g, 2:3)) == 2
  @test_throws ArgumentError circular_cone_beam(; angles, dso, dsd, nu, nv, du, detector=:curved)
  @test_throws ArgumentError ArcConeBeam(g.source, g.centre, g.u .+ [0.0, 0.0, 1.0], g.v; nu, nv)
end

@testitem "helical cone beam" tags = [:geometry] setup = [Rays] begin
  angles = range(0, 4pi, 17)
  pitch, z0 = 30.0, -10.0
  g = helical_cone_beam(; angles, pitch, z0, dso=500, dsd=900, nu=4, nv=3, du=1.0)
  # source and detector rise by pitch per turn; per view the rays equal those of the circular
  # geometry shifted along z
  gc = circular_cone_beam(; angles, dso=500, dsd=900, nu=4, nv=3, du=1.0)
  for (k, beta) in enumerate(angles)
    dz = SVector(0, 0, z0 + pitch * beta / 2pi)
    for i in LinearIndices((4, 3, length(angles)))[:, :, k]
      (a, b), (ac, bc) = ray(g, i), ray(gc, i)
      @test isapprox(a, ac + dz) && isapprox(b, bc + dz)
    end
  end
  @test_throws ArgumentError helical_cone_beam(; angles, pitch, source_z=0, dso=500, dsd=900, nu=4, nv=3, du=1.0)
end

@testitem "parallel beam" tags = [:geometry] setup = [Rays] begin
  angles = range(0, pi, 7)[1:6]
  nu, nv, du, dv = 5, 3, 0.8, 1.2
  g = parallel_beam(; angles, nu, nv, du, dv)
  @test datasize(g) == (nu, nv, 6) && raytype(g) == RayLine()
  # the rays are ImagePhantoms' lines (u, v, ϕ, θ = 0) at the cell positions
  us = ((0:nu-1) .- (nu - 1) / 2) .* du
  vs = ((0:nv-1) .- (nv - 1) / 2) .* dv
  rays = reshape(allrays(g), nu, nv, :)
  for (k, phi) in enumerate(angles), iv in 1:nv, iu in 1:nu
    u, v, phir, theta = line_coordinates(rays[iu, iv, k]...)
    @test isapprox(SVector(u, v, phir, theta), SVector(us[iu], vs[iv], phi, 0); atol=1e-12)
  end
end

@testitem "subset" tags = [:geometry] setup = [Rays] begin
  views = [2, 5, 6]
  for g in (helical_cone_beam(; angles=range(0, 2pi, 8), pitch=10, dso=500, dsd=900, nu=4, nv=3, du=1.0),
            parallel_beam(; angles=range(0, pi, 8), nu=4, nv=3, du=1.0))
    s = subset(g, views)
    @test typeof(s) == typeof(g) && datasize(s) == (4, 3, 3)
    # view j of the subset is view views[j] of the geometry
    rs, rg = reshape(allrays(s), 12, :), reshape(allrays(g), 12, :)
    @test rs == rg[:, views]
  end
end

# line_coordinates is the bridge to ImagePhantoms' exact line integrals in the operator tests:
# chords of a ball at the origin depend only on the distance of the line from the origin
@testitem "line integrals of a ball" tags = [:geometry] setup = [Rays] begin
  using ImagePhantoms: ellipsoid, radon
  R = 40.0
  ball = ellipsoid((0, 0, 0), (R, R, R), (0, 0, 0), 1.0)
  g = helical_cone_beam(; angles=range(0, 2pi, 6), pitch=20, dso=200, dsd=400, nu=9, nv=7, du=10.0)
  for (a, b) in allrays(g)
    d = line_distance(a, b)
    @test isapprox(radon(ball)(line_coordinates(a, b)...), 2sqrt(max(R^2 - d^2, 0)); atol=1e-9)
  end
end

# geometries move to the device in precision T, and the rays computed there by a kernel agree with
# the rays of the CPU geometry
@testitem "geometry on the device" tags = [:geometry, :gpu] setup = [Backends] begin
  for arrayType in arrayTypes, T in floatTypes
    @testset "$arrayType{$T} $(nameof(typeof(g)))" for g in (
        circular_cone_beam(; angles=range(0, 2pi, 7)[1:6], dso=500, dsd=900, nu=5, nv=4, du=1.5),
        helical_cone_beam(; angles=range(0, 4pi, 12), pitch=20, dso=500, dsd=900, nu=5, nv=4, du=1.5),
        helical_cone_beam(; angles=range(0, 4pi, 12), pitch=20, dso=500, dsd=900, nu=5, nv=4, du=1.5,
                          offset_v=0.3, detector=:arc),
        RayList(rand(3, 10), rand(3, 10)))
      gd = adapt(arrayType{T}, g)
      @test eltype(gd) == T
      @test CTOperators.storage_array(gd) isa arrayType{T}
      @test datasize(gd) == datasize(g)
      rd, r = RayList(gd), RayList(g)
      @test rd.starts isa arrayType{T}
      @test isapprox(Array(rd.starts), r.starts) && isapprox(Array(rd.ends), r.ends)
    end
    @testset "$arrayType{$T} subset" begin
      g = parallel_beam(; angles=range(0, pi, 6), nu=4, nv=2, du=1.0)
      gd = subset(adapt(arrayType{T}, g), 2:3)
      @test isapprox(Array(gd.direction), g.direction[:, 2:3])
    end
  end
end

@testitem "display" tags = [:geometry] begin
  g = circular_cone_beam(; angles=range(0, 2pi, 5)[1:4], dso=50, dsd=100, nu=6, nv=3, du=1.0, detector=:arc)
  @test repr(MIME("text/plain"), g) == "ArcConeBeam{Float64}: 4 views, detector of 6 x 3 cells, stored in Array"
  @test repr(MIME("text/plain"), RayList(rand(3, 5), rand(3, 5))) ==
        "RayList{Float64}: 5 rays, data size (5,), stored in Array"
end
