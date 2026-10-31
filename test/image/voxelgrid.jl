@testitem "VoxelGrid 3D" tags = [:image] begin
  using CTOperators: VoxelGrid, bounds, to_index
  using ImageGeoms: axis
  using StaticArrays

  # voxel centres agree with ImageGeoms' axes, and to_index maps them to integer indices
  @testset "offsets $offsets, deltas $deltas" for offsets in [(0, 0, 0), (0.5, -1, 0.25)],
                                                  deltas in [(1.0, 2.0, 0.5), (1.0, -1.0, 2.0)]
    ig = ImageGeom(; dims=(5, 4, 3), deltas, offsets)
    g = VoxelGrid{Float64}(ig)
    @test size(g) == (5, 4, 3)
    for I in CartesianIndices(size(g))
      centre = SVector(ntuple(d -> axis(ig, d)[I[d]], 3))
      @test isapprox(g.first + (SVector(Tuple(I)) .- 1) .* g.deltas, centre)
      @test isapprox(to_index(g, centre), SVector(Tuple(I)))
    end
    # the box is the hull of the voxel faces: centres ± half a voxel
    lo, hi = bounds(g)
    @test all(lo .< hi)
    @test isapprox(hi - lo, abs.(g.deltas) .* SVector(size(g)))
    @test isapprox(lo, min.(g.first, g.first + (SVector(size(g)) .- 1) .* g.deltas) - abs.(g.deltas) / 2)
  end

  @test_throws ArgumentError VoxelGrid{Float64}(ImageGeom(; dims=(2, 2, 2, 2)))
end

@testitem "VoxelGrid 2D" tags = [:image] begin
  using CTOperators: VoxelGrid, bounds
  using StaticArrays

  # a 2D image is one voxel thick, centred at z = 0
  g = VoxelGrid{Float32}(ImageGeom(; dims=(6, 4), deltas=(0.5, 0.5)))
  @test g isa VoxelGrid{Float32}
  @test size(g) == (6, 4, 1)
  @test g.deltas[3] == 1 && g.first[3] == 0
  @test bounds(g) == (SVector(-1.5f0, -1, -0.5), SVector(1.5f0, 1, 0.5))
end
