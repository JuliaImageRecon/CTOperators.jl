# The pullback of op * x is adjoint(op) * ybar: ChainRulesTestUtils compares it with finite
# differences and checks the shapes, for the projector and its adjoint.
@testitem "ChainRules" tags = [:chainrules] begin
  using ChainRulesCore, ChainRulesTestUtils
  image = ImageGeom(; dims=(5, 4, 3))
  g = circular_cone_beam(; angles=range(0, 2pi, 4)[1:3], dso=15, dsd=30, nu=6, nv=4, du=1.2)
  op = SiddonProjector(image, g; eltype=Float64)
  x, y = rand(5, 4, 3), rand(datasize(op)...)
  test_rrule(*, op ⊢ NoTangent(), x)
  test_rrule(*, op ⊢ NoTangent(), vec(x))
  test_rrule(*, adjoint(op) ⊢ NoTangent(), y)
end
