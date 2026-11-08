# LinearOperatorCollection.jl: the normal operator of the projectors is fused (one pass over the
# rays, no buffer for the projection data) and equals the back projection of the weighted forward
# projection, with masks and several rays per cell (Siddon, Joseph), also for a weighted projector
# ProdOp(WeightingOp(w), op). Other weighting operators keep the generic normal operator.

@testitem "LinearOperatorCollection: fused normal operator" tags = [:operators, :gpu] setup = [Backends] begin
  using Adapt, LinearAlgebra, LinearOperatorCollection, LinearOperators
  ext = Base.get_extension(CTOperators, :CTOperatorsLinearOperatorCollectionExt)
  image = ImageGeom(MaskCircle(); dims=(12, 10, 6), deltas=(1.0, 1.1, 0.9))
  g = circular_cone_beam(; angles=range(0, 2pi, 7)[1:6], dso=40, dsd=70, nu=14, nv=8, du=1.6)
  raymask = rand(datasize(g)...) .> 0.2
  multi = MultiRay(g, JitteredPoints(2; seed=1))
  cases = ((SiddonProjector, g), (SiddonProjector, multi), (JosephProjector, g), (JosephProjector, multi),
           (DistanceDrivenProjector, g))
  for (P, geometry) in cases, arrayType in arrayTypes, T in floatTypes
    @testset "$P, $(nameof(typeof(geometry))), $arrayType{$T}" begin
      op = P(image, geometry; eltype=T, storage=arrayType, raymask)
      x = adapt(arrayType{T}, rand(T, prod(image.dims)))
      w = adapt(arrayType{T}, rand(T, prod(datasize(g))))
      N, Nw = normalOperator(op), normalOperator(op, w)
      @test N isa ext.FusedNormalOp && Nw isa ext.FusedNormalOp && N isa LinearOperatorCollection.NormalOp
      @test isapprox(Array(N * x), Array(vec(adjoint(op) * (op * x))); rtol=sqrt(eps(T)))
      @test isapprox(Array(Nw * x), Array(vec(adjoint(op) * (w .* vec(op * x)))); rtol=sqrt(eps(T)))
      @test LinearOperators.storage_type(N) == typeof(x) && size(N) == (prod(image.dims), prod(image.dims))
      # with real weights Hermitian and symmetric: the adjoint and the transpose are the operator
      @test ishermitian(N) && issymmetric(N) && ishermitian(Nw) && issymmetric(Nw)
      @test isapprox(Array(adjoint(Nw) * x), Array(Nw * x); rtol=sqrt(eps(T)))
      @test isapprox(Array(transpose(Nw) * x), Array(Nw * x); rtol=sqrt(eps(T)))
      # a weighted projector W A: LinearOperatorCollection passes the squared weights on
      Np = normalOperator(ProdOp(WeightingOp(w), op))
      @test Np isa ext.FusedNormalOp
      @test isapprox(Array(Np * x), Array(vec(adjoint(op) * (w .^ 2 .* vec(op * x)))); rtol=sqrt(eps(T)))
    end
  end
  # weights as an operator other than a WeightingOp: the generic normal operator
  op = SiddonProjector(image, g)
  W = opEye(Float32, prod(datasize(g)); S=Vector{Float32})
  @test !(normalOperator(op, W) isa ext.FusedNormalOp)
  @test_throws DimensionMismatch normalOperator(op, rand(Float32, 3)) * rand(Float32, prod(image.dims))
end
