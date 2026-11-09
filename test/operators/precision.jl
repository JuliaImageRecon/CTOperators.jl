# The element types follow the arrays: the operator's element type is that of the ray geometry and of
# the computation, the input array's that of the storage (values are converted when read), the
# output array's that of the result (the forward projection rounds each data value once, the back
# projection accumulates in it). Every combination runs; below Float32 the results lose precision,
# which these tests do not check.

@testitem "precision: Float16 and BFloat16 images and data" tags = [:operators, :gpu] setup = [Backends] begin
  using Adapt, LinearAlgebra, BFloat16s
  image = ImageGeom(; dims=(12, 10, 6), deltas=(1.0, 1.1, 0.9))
  g = circular_cone_beam(; angles=range(0, 2pi, 7)[1:6], dso=40, dsd=70, nu=14, nv=8, du=1.6)
  # distance driven integrates over the cell itself: no MultiRay
  multi = MultiRay(g, JitteredPoints(2; seed=1))
  cases = ((SiddonProjector, g), (SiddonProjector, multi), (JosephProjector, g), (JosephProjector, multi),
           (DistanceDrivenProjector, g))
  for S in (Float16, BFloat16), (P, geometry) in cases, arrayType in arrayTypes
    @testset "$S, $P, $(nameof(typeof(geometry))), $arrayType" begin
      xs = S.(rand(image.dims...))
      ys = S.(rand(datasize(g)...))
      op = P(image, geometry; storage=arrayType)
      ds, d32 = adapt(arrayType, xs), adapt(arrayType, Float32.(xs))
      # an image stored in S is projected like the same values in Float32
      y = op * ds
      @test eltype(y) == Float32 && Array(y) == Array(op * d32)
      # data in S: the Float32 projection, rounded once per value
      out = adapt(arrayType, zeros(S, datasize(g)...))
      mul!(out, op, ds)
      @test Array(out) == S.(Array(y))
      # back projection of data in S into a Float32 image (atomic additions on GPUs run in no fixed
      # order, so equal up to rounding)
      bs, b32 = adapt(arrayType, ys), adapt(arrayType, Float32.(ys))
      @test isapprox(Array(adjoint(op) * bs), Array(adjoint(op) * b32); rtol=sqrt(eps(Float32)))
      # the back projection may also accumulate in S, at the user's choice
      acc = adapt(arrayType, zeros(S, image.dims...))
      mul!(acc, adjoint(op), b32)
      @test eltype(acc) == S && all(isfinite, Array(acc))
    end
  end
end

@testitem "precision: any element type for the operator" tags = [:operators] begin
  using Adapt, BFloat16s
  image = ImageGeom(; dims=(8, 8))
  g = parallel_beam(; angles=range(0, pi, 5)[1:4], nu=10, nv=1, du=1.0)
  for P in (SiddonProjector, JosephProjector, DistanceDrivenProjector), T in (Float16, BFloat16)
    # the geometry and the computation in T: runs, the precision is the user's choice
    op = P(image, g; eltype=T)
    @test eltype(op) == T && eltype(adapt(Array{T}, P(image, g))) == T
    y = op * T.(rand(8, 8))
    @test eltype(y) == T && all(isfinite, y)
    @test all(isfinite, adjoint(op) * y)
  end
end
