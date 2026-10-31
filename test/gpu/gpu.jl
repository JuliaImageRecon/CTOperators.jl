# GPU tests. Included by test/gpu/<backend>.jl, which loads the GPU package and defines
# `arrayTypes` (GPU array types) and `floatTypes` (precisions the device supports).

using CTOperators, Test, Adapt

@testset "GPU $arrayType" for arrayType in arrayTypes
  @testset "$T" for T in floatTypes
    # geometries move to the device in precision T, and the rays computed there by a kernel agree
    # with the rays of the CPU geometry
    @testset "geometry $(nameof(typeof(g)))" for g in (
        circular_cone_beam(; angles=range(0, 2π, 7)[1:6], dso=500, dsd=900, nu=5, nv=4, du=1.5),
        helical_cone_beam(; angles=range(0, 4π, 12), pitch=20, dso=500, dsd=900, nu=5, nv=4, du=1.5),
        RayList(rand(3, 10), rand(3, 10)))
      gd = adapt(arrayType{T}, g)
      @test eltype(gd) == T
      @test CTOperators.storage_array(gd) isa arrayType{T}
      @test datasize(gd) == datasize(g)
      rd, r = RayList(gd), RayList(g)
      @test rd.starts isa arrayType{T}
      @test Array(rd.starts) ≈ r.starts && Array(rd.ends) ≈ r.ends
    end
    @testset "subset on the device" begin
      g = parallel_beam(; angles=range(0, π, 6), nu=4, nv=2, du=1.0)
      gd = subset(adapt(arrayType{T}, g), 2:3)
      @test Array(gd.direction) ≈ g.direction[:, 2:3]
    end
  end
end
