using ImagePhantoms: ellipsoid, radon

@testset "Geometries" begin
  @testset "RayList" begin
    starts, ends = rand(3, 6), rand(3, 6)
    g = RayList(starts, ends; datasize=(2, 3))
    @test datasize(g) == (2, 3) && nrays(g) == 6 && raytype(g) == RaySegment()
    @test all(ray(g, i) == (starts[:, i], ends[:, i]) for i in 1:6)
    @test_throws DimensionMismatch RayList(starts, ends; datasize=(4,))
    @test_throws DimensionMismatch RayList(rand(2, 6), rand(2, 6))
    @test_throws DimensionMismatch RayList(starts, rand(3, 5))
  end

  @testset "circular cone beam" begin
    angles = range(0, 2π, 9)[1:8]
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
    for (k, β) in enumerate(angles), iv in 1:nv, iu in 1:nu
      a, b = rays[iu, iv, k]
      @test a ≈ SVector(-dso * sin(β), dso * cos(β), 0)
      @test b ≈ SVector((dsd - dso) * sin(β), -(dsd - dso) * cos(β), 0) +
                us[iu] * SVector(cos(β), sin(β), 0) + vs[iv] * SVector(0, 0, 1)
    end

    # without offsets and with an odd detector the central ray crosses the rotation axis
    g0 = circular_cone_beam(; angles, dso, dsd, nu, nv, du)
    for k in eachindex(angles)
      a, b = ray(g0, LinearIndices((nu, nv, length(angles)))[(nu + 1) ÷ 2, (nv + 1) ÷ 2, k])
      @test line_distance(a, b) < 1e-12 * dsd
    end

    @test_throws ArgumentError circular_cone_beam(; angles, dso=900, dsd=500, nu, nv, du)
    @test_throws DimensionMismatch circular_cone_beam(; angles, dso, dsd, nu, nv, du, source_z=zeros(3))
  end

  @testset "helical cone beam" begin
    angles = range(0, 4π, 17)
    pitch, z0 = 30.0, -10.0
    g = helical_cone_beam(; angles, pitch, z0, dso=500, dsd=900, nu=4, nv=3, du=1.0)
    # source and detector rise by pitch per turn; per view the rays equal those of the circular
    # geometry shifted along z
    gc = circular_cone_beam(; angles, dso=500, dsd=900, nu=4, nv=3, du=1.0)
    for (k, β) in enumerate(angles)
      dz = SVector(0, 0, z0 + pitch * β / 2π)
      for i in LinearIndices((4, 3, length(angles)))[:, :, k]
        @test all(ray(g, i) .≈ ray(gc, i) .+ Ref(dz))
      end
    end
    @test_throws ArgumentError helical_cone_beam(; angles, pitch, source_z=0, dso=500, dsd=900, nu=4, nv=3, du=1.0)
  end

  @testset "parallel beam" begin
    angles = range(0, π, 7)[1:6]
    nu, nv, du, dv = 5, 3, 0.8, 1.2
    g = parallel_beam(; angles, nu, nv, du, dv)
    @test datasize(g) == (nu, nv, 6) && raytype(g) == RayLine()
    # the rays are ImagePhantoms' lines (u, v, ϕ, θ = 0) at the cell positions
    us = ((0:nu-1) .- (nu - 1) / 2) .* du
    vs = ((0:nv-1) .- (nv - 1) / 2) .* dv
    rays = reshape(allrays(g), nu, nv, :)
    for (k, ϕ) in enumerate(angles), iv in 1:nv, iu in 1:nu
      u, v, ϕr, θ = line_coordinates(rays[iu, iv, k]...)
      @test isapprox(SVector(u, v, ϕr, θ), SVector(us[iu], vs[iv], ϕ, 0); atol=1e-12)
    end
  end

  @testset "subset" begin
    views = [2, 5, 6]
    for g in (helical_cone_beam(; angles=range(0, 2π, 8), pitch=10, dso=500, dsd=900, nu=4, nv=3, du=1.0),
              parallel_beam(; angles=range(0, π, 8), nu=4, nv=3, du=1.0))
      s = subset(g, views)
      @test typeof(s) == typeof(g) && datasize(s) == (4, 3, 3)
      # view j of the subset is view views[j] of the geometry
      rs, rg = reshape(allrays(s), 12, :), reshape(allrays(g), 12, :)
      @test rs == rg[:, views]
    end
  end

  # line_coordinates is the bridge to ImagePhantoms' exact line integrals in the operator tests:
  # chords of a ball at the origin depend only on the distance of the line from the origin
  @testset "line integrals of a ball" begin
    R = 40.0
    ball = ellipsoid((0, 0, 0), (R, R, R), (0, 0, 0), 1.0)
    g = helical_cone_beam(; angles=range(0, 2π, 6), pitch=20, dso=200, dsd=400, nu=9, nv=7, du=10.0)
    for (a, b) in allrays(g)
      d = line_distance(a, b)
      @test radon(ball)(line_coordinates(a, b)...) ≈ 2sqrt(max(R^2 - d^2, 0)) atol = 1e-9
    end
  end

  @testset "RayList of a geometry" begin
    g = circular_cone_beam(; angles=range(0, 2π, 5), dso=500, dsd=900, nu=4, nv=3, du=1.0)
    r = RayList(g)
    @test datasize(r) == datasize(g)
    @test allrays(r) == allrays(g)
    @test_throws ArgumentError RayList(parallel_beam(; angles=[0.0], nu=2, nv=2, du=1.0))
  end
end
