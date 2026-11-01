# The operator layer on every device and element type under test (see the snippet `Backends`): the
# interface, moving operators between devices, Float32 accuracy per ray, and the ChainRulesCore
# rule. On the CPU these run on Array; with test/gpu/run.jl also on the GPU (or JLArrays). Results
# are compared with the Float64 operator on the CPU and must stay on the operator's device in its
# element type. The projections themselves are checked by the contract tests.

@testsnippet DeviceProblem begin
  using LinearAlgebra, Adapt
  using CTOperators: KA, storage_array

  const PROJECTORS = (SiddonProjector, JosephProjector)

  device_problem() = (ImageGeom(; dims=(8, 7, 6), deltas=(1.0, 1.25, 0.8), offsets=(0.37, 0.11, -0.23)),
                      helical_cone_beam(; angles=range(0, 3pi, 7), pitch=4, dso=30, dsd=60, nu=13, nv=9,
                                        du=1.17, dv=0.93))

  on_device(a, arrayType) = a isa arrayType && KA.get_backend(a) == KA.get_backend(adapt(arrayType, zeros(1)))
  agree(a, b, T) = isapprox(Array(a), b; rtol=sqrt(eps(T)))
end

@testitem "device: operator interface" tags = [:operators, :gpu] setup = [Backends, DeviceProblem] begin
  image, g = device_problem()
  for P in PROJECTORS, arrayType in arrayTypes, T in floatTypes
    @testset "$P, $arrayType{$T}" begin
      op = P(image, g; eltype=T, storage=arrayType)
      ref = P(image, g; eltype=Float64)
      x, y = rand(imagesize(ref)...), rand(datasize(ref)...)
      xd, yd = adapt(arrayType{T}, x), adapt(arrayType{T}, y)
      @test eltype(op) == T && on_device(storage_array(geometry(op)), arrayType)

      # shaped arrays in, shaped arrays out; vectors in, vectors out; on the device, in T
      fx = op * xd
      @test on_device(fx, arrayType) && eltype(fx) == T && size(fx) == datasize(op)
      @test agree(fx, ref * x, T)
      fv = op * vec(xd)
      @test on_device(fv, arrayType) && fv isa AbstractVector && agree(fv, vec(ref * x), T)
      by = adjoint(op) * yd
      @test on_device(by, arrayType) && eltype(by) == T && size(by) == imagesize(op)
      @test agree(by, adjoint(ref) * y, T)

      # in place, three and five arguments
      out = similar(fx)
      @test agree(mul!(out, op, xd), ref * x, T)
      @test agree(mul!(copy(yd), op, xd, T(2), T(-0.5)), 2 * (ref * x) - 0.5 * y, T)
      @test agree(mul!(copy(xd), adjoint(op), yd, T(1), T(1)), adjoint(ref) * y + x, T)

      # normalisations
      r, c = rowsums(op), colsums(op)
      @test on_device(r, arrayType) && on_device(c, arrayType) && eltype(r) == eltype(c) == T
      @test agree(r, rowsums(ref), T) && agree(c, colsums(ref), T)

      # subsets of views, of the device operator
      views = [2, 5, 6]
      sub = subset(op, views)
      @test on_device(storage_array(geometry(sub)), arrayType)
      @test agree(sub * xd, (ref * x)[:, :, views], T)
    end
  end
end

# adapt moves an operator to a device and back, and converts its precision
@testitem "device: adapt" tags = [:operators, :gpu] setup = [Backends, DeviceProblem] begin
  image, g = device_problem()
  for P in PROJECTORS, arrayType in arrayTypes, T in floatTypes
    @testset "$P, $arrayType{$T}" begin
      cpu = P(image, g; eltype=T)
      ref = P(image, g; eltype=Float64)
      x = rand(imagesize(ref)...)

      dev = adapt(arrayType, cpu)                     # to the device, element type kept
      @test dev isa P && eltype(dev) == T && on_device(storage_array(geometry(dev)), arrayType)
      @test image_geometry(dev) === image
      @test agree(dev * adapt(arrayType{T}, x), ref * x, T)

      back = adapt(Array, dev)                        # and back to the CPU
      @test back isa P && eltype(back) == T && storage_array(geometry(back)) isa Array{T}
      @test agree(back * T.(x), ref * x, T)

      conv = adapt(arrayType{T}, ref)                 # from Float64 on the CPU to T on the device
      @test eltype(conv) == T && on_device(storage_array(geometry(conv)), arrayType)
      @test agree(conv * adapt(arrayType{T}, x), ref * x, T)
    end
  end
end

# Every ray in Float32 agrees with Float64 to Float32 accuracy, on every device. Rays entering
# through the face x = min at heights exactly on planes between voxels, with large voxel coordinates:
# in Float32 they land within rounding of the plane on either side (the regression of the Siddon
# test "Float32 rays entering on planes", here per device and for every projector). Single-ray
# errors are invisible in the norm over all rays, so the test takes the maximum.
@testitem "device: Float32 rays entering on planes" tags = [:operators, :gpu] setup = [Backends, DeviceProblem] begin
  using ImageGeoms: axis
  using Random
  Random.seed!(1)
  image = ImageGeom(; dims=(64, 128, 128), offsets=(0.37, 0.11, -0.23))
  planes(d) = axis(image, d)[1:end-1] .+ 0.5
  xmin = axis(image, 1)[1] - 0.5
  nr = 2000
  starts = zeros(3, nr); ends = zeros(3, nr)
  for r in 1:nr
    entry = [xmin, rand(planes(2)), rand(planes(3))]
    dir = [1.0, 2rand() - 1, 2rand() - 1]
    starts[:, r] = entry - 50dir
    ends[:, r] = entry + 300dir
  end
  g = RayList(starts, ends)
  x = rand(64, 128, 128)
  for P in PROJECTORS, arrayType in arrayTypes
    @testset "$P, $arrayType{Float32}" begin
      y64 = P(image, g; eltype=Float64) * x
      y32 = Array(P(image, g; eltype=Float32, storage=arrayType) * adapt(arrayType{Float32}, x))
      @test maximum(abs.(y32 .- y64)) < 1e-4 * maximum(y64)
    end
  end
end

# The pullback of op * x with device arrays: adjoint(op) * ybar on the device, as on the CPU
@testitem "device: ChainRules" tags = [:chainrules, :gpu] setup = [Backends, DeviceProblem] begin
  using ChainRulesCore
  image, g = device_problem()
  for P in PROJECTORS, arrayType in arrayTypes, T in floatTypes
    @testset "$P, $arrayType{$T}" begin
      op = P(image, g; eltype=T, storage=arrayType)
      ref = P(image, g; eltype=Float64)
      x, ybar = rand(imagesize(ref)...), rand(datasize(ref)...)
      y, pullback = rrule(*, op, adapt(arrayType{T}, x))
      @test on_device(y, arrayType) && agree(y, ref * x, T)
      xbar = unthunk(pullback(adapt(arrayType{T}, ybar))[3])
      @test on_device(xbar, arrayType) && eltype(xbar) == T && agree(xbar, adjoint(ref) * ybar, T)
    end
  end
end
