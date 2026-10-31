# The generic operator layer, through the Siddon projector.

@testitem "five-argument mul!" tags = [:operators] begin
  using LinearAlgebra
  image = ImageGeom(; dims=(6, 6, 4))
  op = SiddonProjector(image, parallel_beam(; angles=range(0, pi, 4), nu=8, nv=5, du=1.0); eltype=Float64)
  x, y = rand(6, 6, 4), rand(datasize(op)...)
  @test isapprox(mul!(copy(y), op, x, 2.0, -0.5), 2 * (op * x) - 0.5 * y)
  @test isapprox(mul!(vec(copy(x)), adjoint(op), vec(y), 1.0, 1.0), vec(adjoint(op) * y + x))
end
