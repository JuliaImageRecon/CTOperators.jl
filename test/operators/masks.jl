# Masks (src/operators/masks.jl): a projector with an image mask (from its ImageGeom) and a ray
# mask (keyword `raymask`) applies M_rays A M_image, the exact adjoint M_image Aᵀ M_rays, on every
# device. The projectors of the package skip masked rays in their kernels and clip the rays to the
# box of the image mask; the results must equal those of the unmasked projector applied to masked
# inputs.

@testsnippet MaskProblem begin
  using LinearAlgebra, Adapt
  using CTOperators: image_mask, ray_mask, handles_raymask

  # a 3D image with the circular field of view plus a box-shaped region of interest, a helical
  # cone beam and a random ray mask with about 20 % masked rays
  function mask_problem()
    dims = (10, 9, 7)
    circle = ImageGeom(MaskCircle(); dims, deltas=(1.0, 1.25, 0.8), offsets=(0.37, 0.11, -0.23))
    roi = falses(dims...)
    roi[3:7, 2:6, 2:5] .= true
    box = ImageGeom(; dims, deltas=(1.0, 1.25, 0.8), offsets=(0.37, 0.11, -0.23), mask=roi)
    g = helical_cone_beam(; angles=range(0, 3pi, 7), pitch=4, dso=30, dsd=60, nu=13, nv=9, du=1.17, dv=0.93)
    raymask = rand(datasize(g)...) .> 0.2
    circle, box, g, raymask
  end
end

@testitem "masks: semantics and adjoint" tags = [:operators, :gpu] setup = [Backends, MaskProblem] begin
  using Random
  Random.seed!(3)
  circle, box, g, raymask = mask_problem()
  for P in (SiddonProjector, JosephProjector), image in (circle, box), arrayType in arrayTypes, T in floatTypes
    @testset "$P, mask of $(count(image.mask)) voxels, $arrayType{$T}" begin
      op = P(image, g; eltype=T, storage=arrayType, raymask)
      plain = P(ImageGeom(; dims=image.dims, deltas=image.deltas, offsets=image.offsets), g; eltype=Float64)
      m = Array(image.mask)
      x, y = rand(image.dims...), rand(datasize(g)...)
      fx = Array(op * adapt(arrayType{T}, x))
      by = Array(adjoint(op) * adapt(arrayType{T}, y))
      # equal to the unmasked projector on masked inputs (also checks the clipping to the mask's box)
      @test isapprox(fx, raymask .* (plain * (m .* x)); rtol=sqrt(eps(T)))
      @test isapprox(by, m .* (adjoint(plain) * (raymask .* y)); rtol=sqrt(eps(T)))
      # exact zeros outside the masks
      @test all(iszero, fx[.!raymask]) && all(iszero, by[.!m])
      # exact adjoint
      xd, yd = adapt(arrayType{T}, rand(T, image.dims...)), adapt(arrayType{T}, rand(T, datasize(g)...))
      @test isapprox(sum(vec(op * xd) .* vec(yd)), sum(vec(xd) .* vec(adjoint(op) * yd)); rtol=100eps(T))
    end
  end
end

@testitem "masks: interface, subset, adapt" tags = [:operators] setup = [MaskProblem] begin
  circle, box, g, raymask = mask_problem()
  op = JosephProjector(circle, g; eltype=Float64, raymask)
  @test image_mask(op) == circle.mask && ray_mask(op) == raymask && handles_raymask(op)
  plain = JosephProjector(ImageGeom(; dims=circle.dims), g)
  @test isnothing(image_mask(plain)) && isnothing(ray_mask(plain))
  # subset takes the views of the ray mask; adapt and similar projectors keep the masks
  sub = subset(op, [2, 5])
  @test ray_mask(sub) == raymask[:, :, [2, 5]]
  x = rand(circle.dims...)
  @test sub * x ≈ (op * x)[:, :, [2, 5]]
  op32 = adapt(Array{Float32}, op)
  @test ray_mask(op32) == raymask && image_mask(op32) == circle.mask
  # the ray mask also as a vector, and with the wrong size
  @test ray_mask(JosephProjector(circle, g; raymask=vec(raymask))) == raymask
  @test_throws DimensionMismatch JosephProjector(circle, g; raymask=trues(3))
  # an empty image mask: no voxel, all projections zero
  none = ImageGeom(; dims=(6, 5, 4), mask=falses(6, 5, 4))
  @test all(iszero, SiddonProjector(none, g; eltype=Float64) * rand(6, 5, 4))
end

# A projector of another package that knows nothing about masks gets the semantics of the image mask
# from the generic layer (the default `projector_masks` takes the mask of its ImageGeom).
@testitem "masks: generic layer for other projectors" tags = [:operators] setup = [MaskProblem] begin
  using CTOperators: AbstractProjector
  import CTOperators: geometry, image_geometry, forward!, backward!
  struct WrappedProjector{T,G,P} <: AbstractProjector{T,G}
    image::ImageGeom
    inner::P    # a projector without mask, doing the work
  end
  WrappedProjector(image, g) =
    WrappedProjector{Float64,typeof(g),Any}(image, SiddonProjector(ImageGeom(; dims=image.dims), g; eltype=Float64))
  geometry(op::WrappedProjector) = geometry(op.inner)
  image_geometry(op::WrappedProjector) = op.image
  CTOperators.KA.get_backend(op::WrappedProjector) = CTOperators.KA.get_backend(op.inner)
  forward!(y, op::WrappedProjector, x) = forward!(y, op.inner, x)
  backward!(x, op::WrappedProjector, y) = backward!(x, op.inner, y)

  circle, box, g, raymask = mask_problem()
  op = WrappedProjector(box, g)
  @test !handles_raymask(op) && image_mask(op) == box.mask
  x, y = rand(box.dims...), rand(datasize(g)...)
  @test op * x ≈ op.inner * (box.mask .* x)
  @test adjoint(op) * y ≈ box.mask .* (adjoint(op.inner) * y)
end
