# Images and data may be stored in Float16, while the projector computes and accumulates in its own
# element type: values are converted when read, and the forward projection rounds each data value
# once. The image of a back projection and the ray geometry need at least Float32.

@testitem "precision: Float16 images and data" tags = [:operators, :gpu] setup = [Backends] begin
  using Adapt, LinearAlgebra
  image = ImageGeom(; dims=(12, 10, 6), deltas=(1.0, 1.1, 0.9))
  g = circular_cone_beam(; angles=range(0, 2pi, 7)[1:6], dso=40, dsd=70, nu=14, nv=8, du=1.6)
  x16 = Float16.(rand(image.dims...))
  y16 = Float16.(rand(datasize(g)...))
  # distance driven integrates over the cell itself: no MultiRay
  multi = MultiRay(g, JitteredPoints(2; seed=1))
  cases = ((SiddonProjector, g), (SiddonProjector, multi), (JosephProjector, g), (JosephProjector, multi),
           (DistanceDrivenProjector, g))
  for (P, geometry) in cases, arrayType in arrayTypes
    @testset "$P, $(nameof(typeof(geometry))), $arrayType" begin
      op = P(image, geometry; storage=arrayType)
      d16, d32 = adapt(arrayType, x16), adapt(arrayType, Float32.(x16))
      # a Float16 image is projected like the same values in Float32
      y = op * d16
      @test eltype(y) == Float32 && Array(y) == Array(op * d32)
      # Float16 data: the Float32 projection, rounded once per value
      out = adapt(arrayType, zeros(Float16, datasize(g)...))
      mul!(out, op, d16)
      @test Array(out) == Float16.(Array(y))
      # back projection of Float16 data into a Float32 image (atomic additions on GPUs run in no fixed
      # order, so equal up to rounding)
      b16, b32 = adapt(arrayType, y16), adapt(arrayType, Float32.(y16))
      @test isapprox(Array(adjoint(op) * b16), Array(adjoint(op) * b32); rtol=sqrt(eps(Float32)))
      # the image of a back projection accumulates: no Float16
      @test_throws ArgumentError mul!(adapt(arrayType, zeros(Float16, image.dims...)), adjoint(op), b32)
    end
  end
end

@testitem "precision: ray geometry needs Float32" tags = [:operators] begin
  using Adapt
  image = ImageGeom(; dims=(8, 8))
  g = parallel_beam(; angles=range(0, pi, 5)[1:4], nu=10, nv=1, du=1.0)
  for P in (SiddonProjector, JosephProjector, DistanceDrivenProjector)
    @test_throws ArgumentError P(image, g; eltype=Float16)
    @test_throws ArgumentError adapt(Array{Float16}, P(image, g))
    @test eltype(P(image, g; eltype=Float64)) == Float64
  end
end
