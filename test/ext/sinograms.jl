# The geometries of Sinograms.jl converted to ours describe the same rays: the analytic line
# integrals of an off-centre ellipsoid (ImagePhantoms) along Sinograms' own rays (`rays(rg)`, in
# the coordinates of `radon`) and along our rays agree. This checks directions, offsets and angles
# end to end.
@testsnippet SinogramsRays begin
  using Sinograms, ImagePhantoms
  using Sinograms: rays, dims
  ob = [ellipsoid((3.0, -2.0, 1.5), (9.0, 6.0, 7.0), (0.4, 0.0, 0.0), 1.0)]   # radon supports rotations about z only
  ours(g) = [radon(ob)(line_coordinates(a, b)...) for (a, b) in allrays(g)]
end

@testitem "Sinograms: 3D geometries" tags = [:sinograms] setup = [Rays, SinogramsRays] begin
  for rg in (CtFanFlat(; ns=24, nt=10, ds=1.1, dt=1.3, offset_s=0.25, offset_t=-0.4, na=12, orbit=360,
                       orbit_start=15, dsd=120, dod=40),
             CtFanArc(; ns=24, nt=10, ds=1.1, dt=1.3, offset_s=0.25, offset_t=-0.4, na=12, orbit=360,
                      orbit_start=15, dsd=120, dod=40),
             CtPar(; ns=24, nt=10, ds=1.1, dt=1.3, offset_s=0.25, offset_t=-0.4, na=12, orbit=180, orbit_start=10))
    g = projection_geometry(rg)
    @test datasize(g) == dims(rg)
    # Sinograms 0.5 returns the polar angle of flat-panel rays (cb_flat_to_par) with the wrong sign:
    # its u and v belong to the ray with the opposite polar angle, and its arc version (computed
    # from the positions of source and detector cell) agrees with us. Corrected here.
    fix(r) = rg isa CtFanFlat ? (r[1], r[2], r[3], -r[4]) : r
    theirs = [radon(ob)(fix(r)...) for r in rays(rg)]
    @test isapprox(vec(ours(g)), vec(theirs); rtol=1e-5, atol=1e-5 * maximum(theirs))
  end
  @test projection_geometry(CtFanFlat(; ns=8, nt=4, na=3)) isa ConeBeam
  @test projection_geometry(CtFanArc(; ns=8, nt=4, na=3)) isa ArcConeBeam
  @test ParallelBeam(CtPar(; ns=8, nt=4, na=3)) isa ParallelBeam
end

@testitem "Sinograms: 2D geometries" tags = [:sinograms] setup = [Rays, SinogramsRays] begin
  for rg in (SinoFanFlat(; nb=30, d=1.2, offset=0.25, na=16, orbit=360, orbit_start=20, dsd=150, dod=50),
             SinoFanArc(; nb=30, d=1.2, offset=0.25, na=16, orbit=360, orbit_start=20, dsd=150, dod=50),
             SinoPar(; nb=30, d=1.2, offset=0.25, na=16, orbit=180, orbit_start=20))
    g = projection_geometry(rg)
    @test datasize(g) == (rg.nb, 1, rg.na)
    # Sinograms' 2D rays (r, phi) are the 3D lines (u = r, v = 0, phi, theta = 0)
    theirs = [radon(ob)(r, 0.0, phi, 0.0) for (r, phi) in rays(rg)]
    @test isapprox(vec(ours(g)), vec(theirs); rtol=1e-5, atol=1e-5 * maximum(theirs))
  end
end

# Sinograms' `rays` covers circular orbits only: the helix is checked through the source heights
# (Sinograms' internal `_source_zs`) and against our own helical_cone_beam.
@testitem "Sinograms: helix and source offset" tags = [:sinograms] setup = [Rays] begin
  using Sinograms
  rg = CtFanFlat(; ns=16, nt=8, ds=1.0, dt=1.0, na=24, orbit=720, dsd=100, dod=40,
                 src=CtSourceHelix(; pitch=0.8, source_z0=-5))
  g = projection_geometry(rg)
  @test isapprox(g.source[3, :], Sinograms._source_zs(rg))
  @test isapprox(g.centre[3, :], Sinograms._source_zs(rg))
  ref = helical_cone_beam(; angles=deg2rad.(collect(Sinograms.angles(rg))), z0=-5,
                          pitch=Sinograms._source_dz_per_view(rg) * 12, dso=60, dsd=100, nu=16, nv=8, du=1.0)
  # Sinograms stores the angles in Float32
  @test isapprox(g.source, ref.source; rtol=1e-5) && isapprox(g.centre, ref.centre; rtol=1e-5)

  # source_offset moves source and detector sideways: the detector cells of view 1 (beta = 0) are
  # at Sinograms' detector positions _xds, _yds
  for shifted in (CtFanFlat(; ns=16, nt=1, ds=1.0, na=4, dsd=100, dod=40, source_offset=1.5),
                  CtFanArc(; ns=16, nt=1, ds=1.0, na=4, dsd=100, dod=40, source_offset=1.5))
    gs = projection_geometry(shifted)
    cells = [ray(gs, i)[2] for i in 1:16]
    @test isapprox(first.(cells), Sinograms._xds(shifted); atol=1e-5)
    @test isapprox(getindex.(cells, 2), Sinograms._yds(shifted); atol=1e-5)
    @test isapprox(gs.source[1:2, 1], [1.5, 60]; atol=1e-12)
  end
end

@testitem "Sinograms: projectors take the geometry" tags = [:sinograms] begin
  using Sinograms
  rg = CtFanArc(; ns=20, nt=8, na=10, dsd=120, dod=40)
  image = ImageGeom(; dims=(16, 16, 8))
  x = rand(16, 16, 8)
  for P in (SiddonProjector, JosephProjector)
    op = P(image, rg; eltype=Float64)
    @test datasize(op) == Sinograms.dims(rg)
    @test isapprox(op * x, P(image, projection_geometry(rg); eltype=Float64) * x)
  end
  @test_throws ArgumentError projection_geometry(SinoMoj(; nb=8, na=4))
end
