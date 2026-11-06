# The generic operator layer, through the Siddon projector.

@testitem "five-argument mul!" tags = [:operators] begin
  using LinearAlgebra
  image = ImageGeom(; dims=(6, 6, 4))
  op = SiddonProjector(image, parallel_beam(; angles=range(0, pi, 4), nu=8, nv=5, du=1.0); eltype=Float64)
  x, y = rand(6, 6, 4), rand(datasize(op)...)
  @test isapprox(mul!(copy(y), op, x, 2.0, -0.5), 2 * (op * x) - 0.5 * y)
  @test isapprox(mul!(vec(copy(x)), adjoint(op), vec(y), 1.0, 1.0), vec(adjoint(op) * y + x))
end

# The blocks of subset_blocks reproduce the projector: every block's projection is the full
# projection in its rows, the rows partition the data, and the back projections of all blocks add
# up to the full back projection; on every device (index vectors on the device).
@testitem "subset_blocks" tags = [:operators, :gpu] setup = [Backends] begin
  image = ImageGeom(; dims=(6, 6, 4))
  g = circular_cone_beam(; angles=range(0, 2pi, 8)[1:7], dso=25, dsd=50, nu=8, nv=5, du=1.0)
  for arrayType in arrayTypes, mode in (:interleaved, :contiguous), n in (1, 3, 7, 20)
    op = JosephProjector(image, g; eltype=Float64, storage=arrayType)
    x, y = rand(6, 6, 4), rand(datasize(op)...)
    full, back = vec(Array(op * adapt(arrayType, x))), zeros(6, 6, 4)
    ops, rows = subset_blocks(op, n; mode)
    @test length(ops) == length(rows) == min(n, 7)
    @test sort(reduce(vcat, [collect(Array(r)) for r in rows])) == 1:length(full)
    for (o, r) in zip(ops, rows)
      @test vec(Array(o * adapt(arrayType, x))) ≈ full[collect(Array(r))]
      back .+= Array(adjoint(o) * adapt(arrayType, reshape(vec(y)[collect(Array(r))], datasize(o))))
    end
    @test back ≈ Array(adjoint(op) * adapt(arrayType, y))
    mode == :contiguous && @test all(r -> r isa AbstractUnitRange, rows)
  end
  @test_throws ArgumentError subset_blocks(JosephProjector(image, g), 2; mode=:random)
end
