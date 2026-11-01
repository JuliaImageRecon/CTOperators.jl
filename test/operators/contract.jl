# Tests every projector must pass, run by the test items of each operator.

@testsnippet Contract begin
  using LinearAlgebra, Adapt

  # Small problems covering the geometry types and ray types: helical cone beam with flat and arc
  # detector (segments, anisotropic voxels, offset image), parallel beam (lines), rays that start or
  # end inside the volume or miss it, an image with a flipped axis (negative voxel size), and a 2D
  # image.
  #
  # No ray lies in a plane between voxels (the offsets and cell sizes are chosen for that): along
  # such a plane the line integral of a voxel image is ambiguous, and different precisions may
  # take the voxels of either side.
  function contract_problems()
    image3 = ImageGeom(; dims=(9, 8, 7), deltas=(1.0, 1.25, 0.8), offsets=(0.37, 0.11, -0.23))
    image2 = ImageGeom(; dims=(10, 9), deltas=(1.0, 1.0), offsets=(0.29, -0.13))
    flipped = ImageGeom(; dims=(9, 8, 7), deltas=(-1.0, 1.25, -0.8), offsets=(0.37, 0.11, -0.23))
    helix = helical_cone_beam(; angles=range(0, 3pi, 7), pitch=4, dso=30, dsd=60, nu=13, nv=9, du=1.17, dv=0.93)
    arc = helical_cone_beam(; angles=range(0, 3pi, 7), pitch=4, dso=30, dsd=60, nu=13, nv=9, du=1.17, dv=0.93,
                            offset_u=0.13, detector=:arc)
    parallel = parallel_beam(; angles=range(0, pi, 6)[1:5], nu=11, nv=7, du=1.07, dv=0.97, offset_u=0.21)
    starts = [-20 -2 0 1.5 30 3.1; 0.3 -1 0.5 2 30 0.2; 0 1 -0.2 0.4 30 20]
    ends = [20 2.5 3 -1 31 3.1; 0.1 1 0.5 -2 32 0.2; 0.5 -1 0.3 0.1 33 -20]
    raylist = RayList(starts, ends)   # through, inside, from inside, inside, missing, axis-aligned
    inplane = parallel_beam(; angles=range(0, pi, 8)[1:7], nu=13, nv=1, du=1.03, offset_u=0.19)
    [("helical cone beam", image3, helix), ("helical cone beam, arc detector", image3, arc),
     ("parallel beam", image3, parallel), ("ray list", image3, raylist), ("flipped axes", flipped, helix),
     ("2D parallel beam", image2, inplane)]
  end

  # <A x, y> = <x, A' y>: the back projection is the transpose of the forward projection. Both sides
  # sum the same products in different order, so they agree to rounding error. The inner product is
  # a plain reduction: `dot` of device arrays needs the vendor's BLAS (rocBLAS is an optional system
  # library on AMD GPUs).
  inner(a, b) = sum(vec(a) .* vec(b))
  function test_adjointness(op, arrayType, T)
    x = adapt(arrayType, rand(T, imagesize(op)))
    y = adapt(arrayType, rand(T, datasize(op)))
    @test isapprox(inner(op * x, y), inner(x, adjoint(op) * y); rtol=100eps(T))
  end

  # the operator on the device and in precision T against the Float64 operator on the CPU
  function test_parity(make_op, image, geometry, arrayType, T)
    op = make_op(image, geometry; eltype=T, storage=arrayType)
    ref = make_op(image, geometry; eltype=Float64)
    x = rand(Float64, imagesize(ref))
    y = rand(Float64, datasize(ref))
    @test isapprox(Array(op * adapt(arrayType{T}, x)), ref * x; rtol=sqrt(eps(T)))
    @test isapprox(Array(adjoint(op) * adapt(arrayType{T}, y)), adjoint(ref) * y; rtol=sqrt(eps(T)))
  end

  function contract_tests(make_op, arrayTypes, floatTypes)
    for (name, image, geometry) in contract_problems(), arrayType in arrayTypes, T in floatTypes
      @testset "$name, $arrayType{$T}" begin
        op = make_op(image, geometry; eltype=T, storage=arrayType)
        @test eltype(op) == T
        test_adjointness(op, arrayType, T)
        test_parity(make_op, image, geometry, arrayType, T)
      end
    end
  end
end
