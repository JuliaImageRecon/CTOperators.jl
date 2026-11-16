# Operators of sampled beams (src/operators/sampled.jl): which projector supports what, the fused and
# the sequential path, the exact adjoint, and the blur of a focal spot against its exact form.

@testsnippet SamplingProblem begin
  using LinearAlgebra, Adapt
  sampling_image() = ImageGeom(; dims=(16, 14, 12), deltas=(1.0, 1.1, 0.9))
  sampling_scan(detector) = circular_cone_beam(; angles=range(0, 2pi, 6)[1:5], dso=40, dsd=80, nu=24, nv=20, du=1.5,
                                               detector)
  rel(a, b) = norm(a - b) / norm(b)
end

@testitem "Sampled: supported combinations" tags = [:operators, :sampling] setup = [SamplingProblem] begin
  using CTOperators: supports_sampling, fuses_sampling, Summed
  image = sampling_image()
  g = sampling_scan(:flat)
  spot, cell = FocalSpot(1.0; points=RegularPoints(2)), DetectorSampling(RegularPoints(2))
  par = parallel_beam(; angles=range(0, pi, 4), nu=24, nv=20, du=1.0)
  # no source in a parallel beam, no cells in a ray list, the footprint covers the cell
  @test_throws ArgumentError sampled(JosephProjector(image, par); source=spot)
  @test_throws ArgumentError sampled(JosephProjector(image, RayList(g)); detector=cell)
  @test_throws ArgumentError sampled(DistanceDrivenProjector(image, g); detector=cell)
  @test sampled(DistanceDrivenProjector(image, g); source=spot) isa SampleSum
  @test sampled(JosephProjector(image, par); detector=cell) isa JosephProjector
  # Siddon and Joseph sample the detector in their kernel (one projector), the others sequentially
  @test fuses_sampling(SiddonProjector(image, g), g, cell, Summed()) && !fuses_sampling(VoxelDrivenProjector(image, g), g, cell, Summed())
  @test sampled(JosephProjector(image, g); detector=cell) isa JosephProjector
  @test sampled(VoxelDrivenProjector(image, g); detector=cell) isa SampleSum
  @test sampled(JosephProjector(image, g); detector=cell, execution=:sequential) isa SampleSum
  # :fused needs a kernel that samples; offsets that vary per cell need one too
  @test_throws ArgumentError sampled(JosephProjector(image, g); source=spot, execution=:fused)
  @test_throws ArgumentError sampled(VoxelDrivenProjector(image, g); detector=cell, execution=:fused)
  @test_throws ArgumentError sampled(JosephProjector(image, g); detector=DetectorSampling(JitteredPoints(2)), execution=:sequential)
  @test_throws ArgumentError sampled(VoxelDrivenProjector(image, g); detector=DetectorSampling(RandomPoints(3)))
  @test_throws ArgumentError sampled(JosephProjector(image, g); detector=cell, execution=:fast)
  # images with one layer: offsets along u only, strategies with several rows refused
  slab = ImageGeom(; dims=(16, 14))
  g2 = circular_cone_beam(; angles=range(0, 2pi, 6)[1:5], dso=40, dsd=80, nu=24, nv=1, du=1.5)
  @test_throws ArgumentError sampled(JosephProjector(slab, g2); source=FocalSpot(1.0; points=RegularPoints(2, 2)))
  @test_throws ArgumentError sampled(JosephProjector(slab, g2); source=FocalSpot(1.0; points=DiskPoints(2)))
  @test sampled(JosephProjector(slab, g2); source=FocalSpot(1.0; points=RegularPoints(3, 1))) isa SampleSum
end

# Fused and sequential detector sampling compute the same mean. The fused path ends a ray at
# cell_ray(g, cu + du, cv + dv), the sequential one at the cell of a geometry with a shifted center:
# the end points differ by the rounding of coordinates up to about 100 (100 eps), which changes the
# length in each crossed voxel by at most twice that. Per data element the difference is then at
# most 2 * 100 eps times the voxels crossed (at most the sum of the image dimensions) times the
# largest voxel value (1 here), plus the rounding of the sum over the rays. A spot of size zero and
# one detector point give the projector itself (the rounding of the sum).
@testitem "Sampled: fused and sequential agree" tags = [:operators, :sampling] setup = [SamplingProblem] begin
  image = sampling_image()
  x = rand(image.dims...)
  cell = DetectorSampling(RegularPoints(2, 2); weights=GaussianWeights(0.4))
  spot = FocalSpot(1.5; points=DiskPoints(2, 4))
  bound = 2 * 100 * eps() * sum(image.dims) * maximum(x) + 32 * eps() * maximum(x) * sum(image.dims)
  for detector in (:flat, :arc), P in (SiddonProjector, JosephProjector)
    op = P(image, sampling_scan(detector); eltype=Float64)
    @test maximum(abs.(sampled(op; detector=cell, execution=:fused) * x .- sampled(op; detector=cell, execution=:sequential) * x)) <= bound
    @test maximum(abs.(sampled(op; source=spot, detector=cell) * x .-
                       sampled(op; source=spot, detector=cell, execution=:sequential) * x)) <= bound
    @test isapprox(sampled(op; source=FocalSpot(0.0; points=RegularPoints(3))) * x, op * x; rtol=9 * eps())
    @test isapprox(sampled(op; detector=DetectorSampling(RegularPoints(1))) * x, op * x; rtol=eps())
  end
end

# SampleSum and SampleStack are exact adjoints of their products (dot-product test, relative bound
# from the rounding of the sums: 1e-12 in Float64); the stack weighted with its weights is the sum;
# subsets are the rows of the views; masks and adapt carry over.
@testitem "Sampled: operators" tags = [:operators, :sampling] setup = [SamplingProblem] begin
  image = sampling_image()
  x = rand(image.dims...)
  spot = FocalSpot(1.5; points=RegularPoints(2), weights=GaussianWeights(0.5))
  cell = DetectorSampling(RegularPoints(2))
  for detector in (:flat, :arc), P in (SiddonProjector, JosephProjector, DistanceDrivenProjector, VoxelDrivenProjector)
    op = P(image, sampling_scan(detector); eltype=Float64)
    A = sampled(op; source=spot, detector=P === DistanceDrivenProjector ? nothing : cell)
    y = rand(datasize(A)...)
    @test isapprox(dot(A * x, y), dot(x, adjoint(A) * y); rtol=1e-12)
    B = stacked(op; source=spot)
    yb = rand(datasize(B)...)
    @test size(B * x) == (datasize(op)..., 4) && isapprox(dot(B * x, yb), dot(x, adjoint(B) * yb); rtol=1e-12)
    w = CTOperators.sample_weights(B)
    @test isapprox(sum(reshape(w[k], 1, 1, :) .* (B * x)[:, :, :, k] for k in 1:4), sampled(op; source=spot) * x; rtol=1e-12)
    @test isapprox(subset(A, [2, 4]) * x, (A * x)[:, :, [2, 4]]; rtol=1e-12)
  end
  masked = JosephProjector(ImageGeom(MaskCircle(); dims=image.dims, deltas=image.deltas), sampling_scan(:flat); eltype=Float64,
                           raymask=rand(24, 20, 5) .> 0.3)
  A = sampled(masked; source=spot)
  rays = CTOperators.ray_mask(masked)
  @test iszero((A * x)[.!rays])
  @test iszero((adjoint(A) * rand(datasize(A)...))[.!CTOperators.image_mask(masked)])
  # Float32 operator: the projections agree within the Float32 rounding of the projector (sqrt(eps))
  A32 = adapt(Array{Float32}, A)
  @test eltype(A32) == Float32 && isapprox(A32 * Float32.(x), A * x; rtol=sqrt(eps(Float32)))
end

# The blur of a box spot on one voxel. With the voxel-driven projector and UnitScale the response of
# a voxel in a view is the bilinear hat around its projection (cu, cv); the spot parallel to the
# detector shifts it by -(t - 1) times the source offset, t the magnification, the same for all
# source points. So the exact response averages the hat over a box of b = f (t - 1) / |u| cells,
# A(x) = (H(x + b/2) - H(x - b/2)) / b with H the integral of the hat, per direction. The sequential
# path with n x n regular points is the midpoint rule for that average: exact where the hat is
# linear, each of its three kinks adds at most h^2 |slope jump| / 8 (h = b / n; jumps 1, 2, 1), so a
# factor differs by at most h^2 / (2b), the product by the sum of both plus their product.
@testitem "Sampled: focal spot blur" tags = [:operators, :sampling] setup = [SamplingProblem] begin
  using CTOperators: project_point, VoxelGrid
  H(x) = x <= -1 ? 0.0 : x <= 0 ? (x + 1)^2 / 2 : x <= 1 ? 1 - (1 - x)^2 / 2 : 1.0
  A(x, b) = (H(x + b / 2) - H(x - b / 2)) / b
  image = ImageGeom(; dims=(9, 9, 9), deltas=(1.0, 1.0, 1.0))
  g = circular_cone_beam(; angles=range(0, 2pi, 4)[1:3], dso=40, dsd=80, nu=40, nv=40, du=1.5)
  grid = VoxelGrid{Float64}(image)
  fu, fv = 3.0, 2.0
  op = VoxelDrivenProjector(image, g; eltype=Float64, scale=UnitScale())
  for q in ((5, 5, 5), (2, 7, 3))
    x = zeros(9, 9, 9)
    x[q...] = 1
    p = grid.first .+ (collect(q) .- 1) .* grid.deltas
    for n in (4, 8, 16)
      y = sampled(op; source=FocalSpot(fu, fv; points=RegularPoints(n))) * x
      for k in 1:3
        h = project_point(geometry(op), p, k)
        bu, bv = fu * (h.magnification - 1) / 1.5, fv * (h.magnification - 1) / 1.5
        exact = [A(iu - h.cu, bu) * A(iv - h.cv, bv) for iu in 1:40, iv in 1:40]
        eu, ev = (bu / n)^2 / (2bu), (bv / n)^2 / (2bv)
        @test maximum(abs.(y[:, :, k] .- exact)) <= eu + ev + eu * ev + 1e-12
        @test isapprox(sum(y[:, :, k]), 1; atol=1e-12)      # the weights of one voxel and view sum to 1
      end
    end
  end
end

# On every device and element type under test: the sampled operators give the projections of the
# Float64 operator on the CPU within the Float32 rounding of the projectors (sqrt(eps(T))), on the
# device of the projector.
@testitem "Sampled: devices" tags = [:operators, :sampling, :gpu] setup = [Backends, SamplingProblem] begin
  image = sampling_image()
  x = rand(image.dims...)
  spot = FocalSpot(1.5; points=DiskPoints(2, 3), weights=GaussianWeights(0.5))
  cell = DetectorSampling(RegularPoints(2))
  for arrayType in arrayTypes, T in floatTypes, detector in (:flat, :arc),
      P in (JosephProjector, VoxelDrivenProjector)
    @testset "$P, $detector, $arrayType{$T}" begin
      g = sampling_scan(detector)
      ref = sampled(P(image, g; eltype=Float64); source=spot, detector=cell)
      A = sampled(P(image, g; eltype=T, storage=arrayType); source=spot, detector=cell)
      y = rand(datasize(ref)...)
      fx = A * adapt(arrayType{T}, x)
      @test fx isa arrayType && eltype(fx) == T && isapprox(Array(fx), ref * x; rtol=sqrt(eps(T)))
      @test isapprox(Array(adjoint(A) * adapt(arrayType{T}, y)), adjoint(ref) * y; rtol=sqrt(eps(T)))
      B = stacked(P(image, g; eltype=T, storage=arrayType); source=spot)
      @test isapprox(Array(B * adapt(arrayType{T}, x)), stacked(P(image, g; eltype=Float64); source=spot) * x; rtol=sqrt(eps(T)))
      # the intensity model on the device (log-sum-exp by broadcasts and reductions)
      m = IntensityModel(P(image, g; eltype=T, storage=arrayType); source=spot)
      mref = IntensityModel(P(image, g; eltype=Float64); source=spot)
      xs = 0.05 .* x
      pm = m(adapt(arrayType{T}, xs))
      @test pm isa arrayType && isapprox(Array(pm), mref(xs); rtol=sqrt(eps(T)))
    end
  end
end

# The intensity model p(x) = -log sum_k w_k exp(-(A_k x)): never above the mean of the line
# integrals (Jensen, up to rounding); its second-order term is half the weighted variance of the
# samples; its Jacobian at 0 is the linear model, elsewhere the derivative of p (finite
# differences); the gradient is J(x)ᵀ (p(x) - y).
@testitem "Sampled: intensity model" tags = [:operators, :sampling] setup = [SamplingProblem] begin
  using CTOperators: jacobian, gradient!, sample_weights
  image = sampling_image()
  op = JosephProjector(image, sampling_scan(:flat); eltype=Float64)
  spot = FocalSpot(3.0; points=RegularPoints(2), weights=GaussianWeights(0.6))
  cell = DetectorSampling(RegularPoints(2, 1))
  m = IntensityModel(op; source=spot, detector=cell)
  lin = sampled(op; source=spot, detector=cell, execution=:sequential)
  x = 0.05 .* rand(image.dims...)       # line integrals of order 1
  p = m(x)
  @test size(p) == datasize(op) && all(p .<= lin * x .+ 1e-14 .* abs.(lin * x))
  @test any(p .< lin * x .- 1e-6)        # the effect is there
  @test vec(p) == m(vec(x))

  # p_lin - p = Var/2 - kappa3/6 + O(eps^4) for the projections eps q of the samples, so
  # (p_lin - p)/eps^2 - Var/2 is at most eps * max|q - mean|^3 / 6 in magnitude, plus rounding: the
  # log-sum-exp forms p as -(M + log S) with M and log S of the size of |log w|, so p is off by a
  # few eps(Float64) * (|log w| + 1), divided by eps^2
  e = 1e-3
  B = stacked(op; source=spot, detector=cell)
  q = B * x
  w = cat((reshape(v, 1, 1, :) for v in sample_weights(B))...; dims=4)
  mean = sum(w .* q; dims=4)
  var = dropdims(sum(w .* (q .- mean) .^ 2; dims=4); dims=4)
  dev = dropdims(maximum(abs.(q .- mean); dims=4); dims=4)
  lhs = (lin * (e .* x) .- m(e .* x)) ./ e^2
  logw = maximum(abs, log.(w))
  @test all(abs.(lhs .- var ./ 2) .<= e .* dev .^ 3 ./ 6 .+ 8 * eps() * (logw + 1) / e^2)

  @test isapprox(jacobian(m, zero(x)) * x, lin * x; rtol=1e-14)
  # central differences: truncation h^2/6 |p'''| with |p'''| <= max_k |A_k v|^3 (derivatives of a
  # log-sum-exp of linear forms), rounding about 4 eps |p| / h
  v = 0.05 .* randn(image.dims...)
  h = 1e-4
  Jv = jacobian(m, x) * v
  av = maximum(abs, B * v)
  fd = (m(x .+ h .* v) .- m(x .- h .* v)) ./ (2h)
  @test maximum(abs.(fd .- Jv)) <= h^2 / 6 * av^3 + 4 * eps() * maximum(abs, p) / h
  # the Jacobian is an exact adjoint pair; the gradient is its adjoint applied to the residual
  J = jacobian(m, x)
  yd = rand(datasize(op)...)
  @test isapprox(dot(J * v, yd), dot(v, adjoint(J) * yd); rtol=1e-12)
  y = lin * x
  g = similar(x)
  gradient!(g, m, x, y)
  @test isapprox(g, adjoint(J) * (p .- y); rtol=1e-12)
  G = IntensityGradient(m, y)
  @test size(G) == (prod(image.dims), prod(image.dims)) && eltype(G) == Float64
  @test isapprox(G * vec(x), vec(g); rtol=1e-14)
  # the detector alone; jittered points need a projector that stacks them in its kernel
  @test IntensityModel(op; detector=DetectorSampling(DiskPoints(1, 4)))(x) isa Array{Float64,3}
  @test_throws ArgumentError IntensityModel(op; detector=DetectorSampling(JitteredPoints(2)))
end
