@testitem "VoxelGrid 3D" tags = [:image] begin
  using CTOperators: VoxelGrid, bounds, to_index
  using ImageGeoms: axis
  using StaticArrays

  # voxel centers agree with ImageGeoms' axes, and to_index maps them to integer indices
  @testset "offsets $offsets, deltas $deltas" for offsets in [(0, 0, 0), (0.5, -1, 0.25)],
                                                  deltas in [(1.0, 2.0, 0.5), (1.0, -1.0, 2.0)]
    ig = ImageGeom(; dims=(5, 4, 3), deltas, offsets)
    g = VoxelGrid{Float64}(ig)
    @test size(g) == (5, 4, 3)
    for I in CartesianIndices(size(g))
      center = SVector(ntuple(d -> axis(ig, d)[I[d]], 3))
      @test isapprox(g.first + (SVector(Tuple(I)) .- 1) .* g.deltas, center)
      @test isapprox(to_index(g, center), SVector(Tuple(I)))
    end
    # the box is the hull of the voxel faces: centers ± half a voxel
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

  # a 2D image is one voxel thick, centered at z = 0
  g = VoxelGrid{Float32}(ImageGeom(; dims=(6, 4), deltas=(0.5, 0.5)))
  @test g isa VoxelGrid{Float32}
  @test size(g) == (6, 4, 1)
  @test g.deltas[3] == 1 && g.first[3] == 0
  @test bounds(g) == (SVector(-1.5f0, -1, -0.5), SVector(1.5f0, 1, 0.5))
end

# The voxel index type follows from the size of the image: Int32 where the index arithmetic fits,
# Int64 beyond. Large sizes are only constructed, nothing is projected.
@testitem "VoxelGrid: index type" tags = [:image] begin
  using CTOperators: VoxelGrid, index_type, voxel_index
  using StaticArrays
  @test index_type((10, 10, 10)) == Int32 && index_type((1024, 1024, 1024)) == Int32
  # 1290^3 is the largest cube whose linear indices fit
  @test index_type((1290, 1290, 1290)) == Int32 && index_type((1291, 1291, 1291)) == Int64
  # products beyond Int64 do not wrap around
  @test index_type((2^21, 2^21, 2^21)) == Int64
  deltas, first = SVector(1.0f0, 1.0f0, 1.0f0), SVector(0.0f0, 0.0f0, 0.0f0)
  small = VoxelGrid{Float32}((64, 64, 64), deltas, first)
  large = VoxelGrid{Float32}((3000, 3000, 3000), deltas, first)
  @test small isa VoxelGrid{Float32,Int32} && eltype(small.dims) == Int32
  @test large isa VoxelGrid{Float32,Int64} && eltype(large.dims) == Int64
  # the size is plain Ints whatever the index type; large images do not overflow
  @test size(small) == (64, 64, 64) && eltype(size(small)) == Int
  @test prod(size(large)) == 27 * 10^9
  # the voxel index keeps the integer type, on the largest coordinates too
  @test voxel_index(3.4f0, Int32(64)) === Int32(3)
  @test voxel_index(2999.6, 3000) === 3000
  @test VoxelGrid{Float32}(ImageGeom(; dims=(8, 8))) isa VoxelGrid{Float32,Int32}
end

# The kernels use the index type of the grid: both types give identical results, and the arithmetic
# stays in the type, also at the edges of the volume, where neighbors lie outside.
@testitem "VoxelGrid: kernels in both index types" tags = [:operators] setup = [Rays] begin
  using CTOperators: VoxelGrid, Gather, Scatter, siddon_segment, joseph_segment, to_index
  image = ImageGeom(; dims=(9, 7, 5), deltas=(1.0, 1.25, 0.8), offsets=(0.3, -0.2, 0.1))
  grid32, grid64 = VoxelGrid{Float64,Int32}(image), VoxelGrid{Float64,Int64}(image)
  g = helical_cone_beam(; angles=range(0, 3pi, 7), pitch=4, dso=30, dsd=60, nu=13, nv=9, du=1.2, dv=1.0)
  x = rand(prod(image.dims))
  for segment in (siddon_segment, joseph_segment), i in 1:nrays(g)
    a, b = ray(g, i)
    @test segment(Gather(x), grid32, a, b, raytype(g)) == segment(Gather(x), grid64, a, b, raytype(g))
    x32, x64 = zeros(length(x)), zeros(length(x))
    segment(Scatter(x32, 1.0), grid32, a, b, raytype(g))
    segment(Scatter(x64, 1.0), grid64, a, b, raytype(g))
    @test x32 == x64
  end
  # projectors take the index type from the image, as a concrete type
  op = JosephProjector(image, g)
  @test op.grid isa VoxelGrid{Float32,Int32}
  @test isconcretetype(typeof(op).parameters[end])
end
