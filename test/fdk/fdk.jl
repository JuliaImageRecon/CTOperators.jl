# FDK and filtered back projection: the steps on their own (filter, scan), then reconstructions of
# phantoms from their exact line integrals (ImagePhantoms.jl) and the comparison with Sinograms.jl.

@testitem "FDK: ramp filter" tags = [:fdk] begin
  using CTOperators: filter_response, ramp_tap, padded_length
  nu, du = 64, 0.7
  npad = padded_length(nu)
  @test npad >= 2 * nu - 1
  f = (0:div(npad, 2)) ./ (npad * du)       # frequencies in cycles per unit length, up to Nyquist
  ramp = filter_response(n -> ramp_tap(n, du), nu, du, npad, :ramp, 1.0) .* npad
  # the band-limited ramp |f|, up to the truncation of the kernel to 2 nu - 1 taps (at f = 0 about
  # 1 / (pi^2 nu du))
  @test maximum(abs.(ramp .- f)) <= 1 / (nu * du)
  # windows, as a fraction of the cutoff; nothing above the cutoff
  fc = 0.5 / (2 * du)
  x = f ./ fc
  hann = filter_response(n -> ramp_tap(n, du), nu, du, npad, :hann, 0.5) .* npad
  @test isapprox(hann, ramp .* [t <= 1 ? (1 + cospi(t)) / 2 : 0.0 for t in x])
  @test all(iszero, hann[x .> 1])
  custom = filter_response(n -> ramp_tap(n, du), nu, du, npad, t -> 1 - t, 1.0) .* npad
  @test isapprox(custom, ramp .* (1 .- 2 .* f .* du))
  # any callable
  struct Linear end
  (::Linear)(t) = 1 - t
  @test isapprox(filter_response(n -> ramp_tap(n, du), nu, du, npad, Linear(), 1.0) .* npad, custom)
  for w in (:hamming, :cosine, :shepplogan)
    h = filter_response(n -> ramp_tap(n, du), nu, du, npad, w, 1.0) .* npad
    @test isapprox(h[1], ramp[1]) && all(abs.(h) .<= abs.(ramp) .+ 1e-12)
  end
  # the equiangular kernel of an arc detector becomes the flat one for a large radius
  arc = filter_response(n -> ramp_tap(n, du, 1e6), nu, du, npad, :ramp, 1.0) .* npad
  @test isapprox(arc, ramp; rtol=1e-6)
  @test_throws ArgumentError filter_response(n -> ramp_tap(n, du), nu, du, npad, :unknown, 1.0)
  @test_throws ArgumentError filter_response(n -> ramp_tap(n, du), nu, du, npad, :ramp, 0.0)
  @test_throws ArgumentError filter_response(n -> ramp_tap(n, du), nu, du, npad, :ramp, 1.5)
end

@testitem "FDK: scan" tags = [:fdk] begin
  using CTOperators: circular_scan
  angles = range(0, 2pi, 13)[1:12]
  for detector in (:flat, :arc)
    s = circular_scan(circular_cone_beam(; angles, dso=100, dsd=150, nu=8, nv=4, du=1.0, offset_u=0.3,
                                         offset_v=-0.5, source_z=2.0, detector))
    @test isapprox(s.angles, angles) && isapprox(s.dso, fill(100, 12)) && isapprox(s.dsd, fill(150, 12)) &&
          isapprox(s.axis, [0, 0, 1])
    # at angle 0 the source is on +y: the central ray runs along -y
    @test isapprox(s.central[:, 1], [0, -1, 0]; atol=1e-12)
  end
  # unwrapped: monotonic across pi, in either direction
  @test isapprox(circular_scan(circular_cone_beam(; angles=angles .+ 3.0, dso=100, dsd=150, nu=8, nv=4, du=1.0)).angles,
                 rem2pi(3.0, RoundNearest) .+ (angles .- angles[1]))
  @test issorted(circular_scan(circular_cone_beam(; angles=reverse(angles), dso=100, dsd=150, nu=8, nv=4, du=1.0)).angles;
                 rev=true)
  # not a circular scan
  cone(angles; kw...) = circular_cone_beam(; angles, dso=100, dsd=150, nu=8, nv=4, du=1.0, kw...)
  @test_throws ArgumentError circular_scan(helical_cone_beam(; angles, pitch=10, dso=100, dsd=150, nu=8, nv=4, du=1.0))
  @test_throws ArgumentError circular_scan(cone(angles[[1, 3, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]]))
  @test_throws ArgumentError circular_scan(cone(angles[1:2]))
  g = cone(angles)
  tilted = ConeBeam(g.source, g.center, g.u .+ 0.2 .* (g.center .- g.source) ./ 150, g.v; nu=8, nv=4)
  @test_throws ArgumentError circular_scan(tilted)
  @test_throws ArgumentError FDK(ImageGeom(; dims=(8, 8, 4)), RayList(g))
  @test_throws ArgumentError FDK(ImageGeom(; dims=(8, 8)), g)                      # 2D image, 4 rows
  @test_throws ArgumentError FDK(ImageGeom(; dims=(8, 8, 1)), g)                   # one layer along z, 4 rows
  # cells of different width, arcs of different radius
  wide = copy(g.u)
  wide[:, 3] .*= 1.01
  @test_throws ArgumentError circular_scan(ConeBeam(g.source, g.center, wide, g.v; nu=8, nv=4))
  a = cone(angles; detector=:arc)
  far = copy(a.center)
  far[:, 3] .+= 0.01 .* (a.center[:, 3] .- a.source[:, 3])
  @test_throws ArgumentError circular_scan(ArcConeBeam(a.source, far, a.u, a.v; nu=8, nv=4))
  @test_throws ArgumentError circular_scan(parallel_beam(; angles=[0.0], nu=8, nv=1, du=1.0))
end

@testitem "FDK: full scan" tags = [:fdk] begin
  using CTOperators: circular_scan, redundancy_weights, redundancy_weight
  views(g) = redundancy_weights(FullScan(), circular_scan(g), g)[2]
  cone(angles) = circular_cone_beam(; angles, dso=100, dsd=150, nu=8, nv=4, du=1.0)
  # a full scan measures every line twice: half the angle step per view
  @test isapprox(views(cone(range(0, 2pi, 13)[1:12])), fill(pi / 12, 12))
  # unequal steps, either direction; the gap that closes the circle is the missing neighbor
  a = [0.0, 0.3, 0.5, 1.4, 2.0, 3.1, 4.0, 4.5, 5.5, 6.0]
  for b in (a, reverse(a), a .+ 1.0)
    d = diff(sort(b))
    gap = 2pi - sum(d)
    steps = [(gap + d[1]) / 2; (d[1:end-1] .+ d[2:end]) ./ 2; (d[end] + gap) / 2]
    @test isapprox(views(cone(b)), (issorted(b) ? steps : reverse(steps)) ./ 2)
  end
  # a closed range counts the last view as the first
  @test isapprox(views(cone(range(0, 2pi, 13))), [pi / 24; fill(pi / 12, 11); pi / 24])
  # parallel beams over half the circle measure every line once: the full angle step
  @test isapprox(views(parallel_beam(; angles=range(0, pi, 7)[1:6], nu=8, nv=1, du=1.0)), fill(pi / 6, 6))
  @test isapprox(views(parallel_beam(; angles=range(0, 2pi, 7)[1:6], nu=8, nv=1, du=1.0)), fill(pi / 6, 6))
  # the same weight for every data value
  g = cone(range(0, 2pi, 13)[1:12])
  weight = redundancy_weights(FullScan(), circular_scan(g), g)[1]
  @test redundancy_weight(weight, g, 3, 2, 5) === 1.0
  # short scans are not a full scan
  @test_throws ArgumentError views(cone(range(0, pi + 0.5, 20)))
  @test_throws ArgumentError views(cone(range(0, 1.5pi, 20)))
  @test_throws ArgumentError FDK(ImageGeom(; dims=(8, 8, 4)), cone(range(0, pi + 0.5, 20)))
end

# A redundancy weighting of one's own: a weight per data value from an array (it travels to the
# device), with the view weights scaled down by the same factor, reconstructs as the full scan.
@testsnippet CustomRedundancy begin
  using Adapt
  using CTOperators: CircularScan, redundancy_weights, redundancy_weight
  struct PerView{V}
    w::V
  end
  Adapt.@adapt_structure PerView
  CTOperators.redundancy_weight(p::PerView, geometry::AbstractDetectorGeometry{T}, iu, iv, k) where {T} = T(p.w[k])
  struct Scaled <: AbstractRedundancy
    factor::Float64
  end
  function CTOperators.redundancy_weights(r::Scaled, scan::CircularScan, geometry)
    _, views = redundancy_weights(FullScan(), scan, geometry)
    PerView(fill(Float32(r.factor), length(views))), views ./ r.factor
  end
end

@testitem "FDK: custom redundancy" tags = [:fdk] setup = [CustomRedundancy] begin
  image = ImageGeom(; dims=(16, 16, 8))
  g = circular_cone_beam(; angles=range(0, 2pi, 25)[1:24], dso=60, dsd=100, nu=24, nv=12, du=1.2)
  y = rand(datasize(g)...)
  full = FDK(image, g; eltype=Float64) * y
  @test isapprox(FDK(image, g; eltype=Float64, redundancy=Scaled(4.0)) * y, full)
  @test FDK(image, g; eltype=Float64).redundancy isa CTOperators.UniformRedundancy
end

# Weights given as arrays: the full scan as arrays reconstructs as FullScan(), dimensions of length
# 1 stand for all cells or views along them.
@testitem "FDK: array redundancy" tags = [:fdk] begin
  image = ImageGeom(; dims=(16, 16, 8))
  n = 24
  g = circular_cone_beam(; angles=range(0, 2pi, n + 1)[1:n], dso=60, dsd=100, nu=24, nv=12, du=1.2)
  nu, nv = detectorsize(g)
  y = rand(datasize(g)...)
  full = FDK(image, g; eltype=Float64) * y
  @test isapprox(FDK(image, g; eltype=Float64, redundancy=ArrayRedundancy(ones(1, 1, 1), fill(pi / n, n))) * y, full)
  @test isapprox(FDK(image, g; eltype=Float64, redundancy=ArrayRedundancy(fill(2.0, nu, nv, n), fill(pi / 2n, n))) * y, full)
  r = rand(nu, 1, n) .+ 0.5
  views = rand(n)
  @test isapprox(FDK(image, g; eltype=Float64, redundancy=ArrayRedundancy(r, views)) * y,
                 FDK(image, g; eltype=Float64, redundancy=ArrayRedundancy(repeat(r, 1, nv, 1), views)) * y)
  @test_throws DimensionMismatch FDK(image, g; redundancy=ArrayRedundancy(ones(nu + 1, 1, n), views))
  @test_throws DimensionMismatch FDK(image, g; redundancy=ArrayRedundancy(ones(1, nv, n - 1), views))
  @test_throws DimensionMismatch FDK(image, g; redundancy=ArrayRedundancy(ones(1, 1, 1), views[1:end-1]))
end

@testitem "FDK: parallel beam" tags = [:fdk] begin
  using ImagePhantoms
  using ImageGeoms: axis
  image = ImageGeom(; dims=(64, 64), deltas=(1.0, 1.0))
  disk = [ellipse(0, 0, 20, 20, 0, 1.0)]
  nu = 96
  us = ((0:nu-1) .- (nu - 1) / 2)
  inner = [hypot(a, b) < 16 for a in axis(image, 1), b in axis(image, 2)]
  for angles in (range(0, pi, 181)[1:180], range(0, 2pi, 361)[1:360])
    y = reshape([radon(disk)(u, phi) for u in us, phi in angles], nu, 1, :)
    g = parallel_beam(; angles, nu, nv=1, du=1.0)
    x = CTOperators.fdk(image, g, y; eltype=Float64, window=:ramp)
    @test size(x) == (64, 64)
    # the data are exact: the error is the discretization of filter and back projection
    @test maximum(abs.(x[inner] .- 1)) < 0.01
    xh = CTOperators.fdk(image, g, y; eltype=Float64)                    # Hann: smoother edge, same interior
    @test abs(sum(xh[inner]) / count(inner) - 1) < 0.01
  end
end

# Reconstructions of an ellipsoid from its exact line integrals. Error model:
# - central slices (z = +-0.5, half a voxel from the plane of the orbit, where FDK is the fan-beam
#   formula): the discretization of filter and back projection; measured 0.2 % at most, every
#   window, also with detector offsets;
# - volume, against the exact volume 4/3 pi a b c (the sampled phantom is 0.9 % short of it):
#   Feldkamp et al. (1984) show that FDK conserves the integral along lines parallel to the axis for
#   a flat detector, so only the discretization remains (measured 0.07 % at most); that derivation
#   does not cover rows filtered along an arc (measured -0.7 %, bound 1 %).
@testitem "FDK: circular cone beam" tags = [:fdk] setup = [Rays] begin
  using ImagePhantoms
  using ImageGeoms: axis
  image = ImageGeom(; dims=(40, 40, 24), deltas=(1.0, 1.0, 1.0))
  ob = [ellipsoid((0, 0, 0), (12, 12, 7), (0, 0, 0), 1.0)]
  exact = 4 / 3 * pi * 12 * 12 * 7
  inner = [hypot(a, b) < 9 for a in axis(image, 1), b in axis(image, 2)]
  angles = range(0, 2pi, 181)[1:180]
  for (g, volume_bound) in ((circular_cone_beam(; angles, dso=120, dsd=240, nu=96, nv=48, du=1.5), 0.005),
                            (circular_cone_beam(; angles, dso=120, dsd=240, nu=96, nv=48, du=1.5, detector=:arc), 0.01),
                            (circular_cone_beam(; angles, dso=120, dsd=240, nu=96, nv=48, du=1.5, offset_u=3.3,
                                                offset_v=-2.2), 0.005),
                            (parallel_beam(; angles=range(0, pi, 181)[1:180], nu=64, nv=32, du=1.0), 0.005))
    y = reshape([radon(ob)(line_coordinates(ray(g, i)...)...) for i in 1:nrays(g)], datasize(g))
    for window in (:ramp, :hann, :hamming, :cosine, :shepplogan)
      x = CTOperators.fdk(image, g, y; eltype=Float64, window)
      @test abs(sum(x[:, :, 12][inner]) / count(inner) - 1) < 0.005
      @test abs(sum(x) / exact - 1) < volume_bound
    end
    plan = FDK(image, g; eltype=Float64, window=:ramp)
    x = plan * y
    @test isapprox(plan * vec(y), vec(x)) && isapprox(mul!(similar(x), plan, y), x)
    # the image mask: voxels outside get 0, inside the same values
    masked_image = ImageGeom(MaskCircle(); dims=(40, 40, 24), deltas=(1.0, 1.0, 1.0))
    inside = falses(size(x)) .| masked_image.mask
    xm = FDK(masked_image, g; eltype=Float64, window=:ramp) * y
    @test !all(inside) && all(iszero, xm[.!inside]) && isapprox(xm[inside], x[inside])
  end
end

# Sinograms.jl's FDK (MIT) as reference: on its own test case both reconstruct with errors of the
# same size; the reconstructions differ by about as much as each differs from the phantom at its edge.
@testitem "FDK: Sinograms" tags = [:fdk, :sinograms] begin
  using Sinograms, ImagePhantoms, LinearAlgebra
  using Sinograms: plan_fbp, rays, CtFanArc, CtFanFlat
  ob = ellipsoid((2.0, 3.0, 4.0), (8.0, 7.0, 6.0))
  ig = ImageGeom((30, 28, 26), (1.0, 1.0, 1.0))
  truth = phantom(axes(ig)..., [ob])
  for geo in (CtFanArc, CtFanFlat)
    rg = geo(; ns=64, nt=40, ds=1.0, dt=1.2, na=32)
    proj = radon(rays(rg), [ob])
    xs = Sinograms.fdk(plan_fbp(rg, ig), proj)
    xo = CTOperators.fdk(ig, rg, proj; eltype=Float64, window=:ramp)
    @test abs(sum(xo) / sum(truth) - 1) < 0.01
    @test norm(xo - truth) <= 1.1 * norm(xs - truth)
    @test norm(xo - xs) <= 0.1 * norm(xs)
    # Sinograms' fdk takes the plan or a geometry of this package, and keeps its own method
    g = projection_geometry(rg)
    @test isapprox(Sinograms.fdk(FDK(ig, rg; eltype=Float64, window=:ramp), proj), xo)
    @test isapprox(Sinograms.fdk(ig, g, proj; eltype=Float64, window=:ramp), xo)
    @test Sinograms.fdk(plan_fbp(rg, ig), proj) == xs
  end
end

@testitem "FDK: devices" tags = [:fdk, :gpu] setup = [Backends, CustomRedundancy] begin
  using Adapt, LinearAlgebra
  image = ImageGeom(; dims=(24, 24, 12))
  for g in (circular_cone_beam(; angles=range(0, 2pi, 41)[1:40], dso=60, dsd=100, nu=40, nv=20, du=1.2),
            circular_cone_beam(; angles=range(0, 2pi, 41)[1:40], dso=60, dsd=100, nu=40, nv=20, du=1.2, detector=:arc),
            parallel_beam(; angles=range(0, pi, 31)[1:30], nu=36, nv=12, du=1.0))
    y = rand(Float32, datasize(g))
    reference = CTOperators.fdk(image, g, y)
    # Float32 on the device against Float32 on the CPU: only the order of the operations differs
    # (FFT, sums), a few roundings of eps(Float32) = 1.2e-7 per value (measured 1.5e-7)
    for arrayType in arrayTypes
      x = Array(CTOperators.fdk(image, g, adapt(arrayType, y); storage=arrayType))
      @test norm(x - reference) <= 1e-5 * norm(reference)
      x = Array(CTOperators.fdk(image, g, adapt(arrayType, y); storage=arrayType, redundancy=Scaled(4.0)))
      @test norm(x - reference) <= 1e-5 * norm(reference)
      # weights as arrays, given on the host: twice the cell weights, half the view weights
      _, views = CTOperators.redundancy_weights(FullScan(), CTOperators.circular_scan(g), g)
      redundancy = ArrayRedundancy(fill(2.0f0, detectorsize(g)[1], 1, nviews(g)), Float32.(views ./ 2))
      x = Array(CTOperators.fdk(image, g, adapt(arrayType, y); storage=arrayType, redundancy))
      @test norm(x - reference) <= 1e-5 * norm(reference)
      # a plan moved to the device and back
      plan = adapt(arrayType, FDK(image, g))
      @test CTOperators.KA.get_backend(plan.padded) == CTOperators.KA.get_backend(adapt(arrayType, y))
      @test norm(Array(plan * adapt(arrayType, y)) - reference) <= 1e-5 * norm(reference)
      @test norm(adapt(Array, plan) * y - reference) <= 1e-5 * norm(reference)
    end
  end
end
