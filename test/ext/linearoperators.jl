# The storage type of CT operators for LinearOperators.jl (used by LinearOperatorCollection.jl, e.g.
# for `normalOperator(op)`): the vector type of the operator's device and element type. The normal
# operator itself is exercised by the documentation examples; LinearOperatorCollection restricts
# KernelAbstractions to 0.9 and is therefore no test dependency.
@testitem "LinearOperators: storage type" tags = [:operators, :gpu] setup = [Backends] begin
  using LinearOperators
  image = ImageGeom(; dims=(6, 5, 4))
  g = circular_cone_beam(; angles=range(0, 2pi, 6)[1:5], dso=15, dsd=30, nu=7, nv=5, du=1.2)
  for P in (SiddonProjector, JosephProjector), arrayType in arrayTypes, T in floatTypes
    op = P(image, g; eltype=T, storage=arrayType)
    S = typeof(adapt(arrayType{T}, zeros(2)))
    @test LinearOperators.storage_type(op) == S
    @test LinearOperators.storage_type(adjoint(op)) == S
  end
end
