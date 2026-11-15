# Launch configurations (src/operators/launch.jl): the layout (the order in which threads take the rays,
# or the voxels per thread) and the workgroup size change how the kernels run, not their results
# (beyond rounding).

# Every ray exactly once, padding threads none, for detectors that are no multiple of the tile
@testitem "launch: tiles cover every ray once" tags = [:operators] begin
  using CTOperators: TileOrder, DataOrder, nthreads, ray_index
  for (nu, nv, tu, tv) in ((13, 9, 8, 8), (16, 8, 8, 8), (5, 1, 8, 8), (7, 3, 4, 2), (1, 1, 8, 8))
    g = parallel_beam(; angles=range(0, pi, 3), nu, nv, du=1.0)
    o = TileOrder(nu, nv, tu, tv)
    idx = [ray_index(o, j) for j in 1:nthreads(o, g)]
    @test nthreads(o, g) % (tu * tv) == 0
    @test sort(filter(>(0), idx)) == 1:nrays(g)
    # the threads of the first tile take neighboring cells of the first view
    cells = [CartesianIndices(datasize(g))[i] for i in idx[1:tu*tv] if i > 0]
    @test all(c -> c[1] <= tu && c[2] <= tv && c[3] == 1, cells)
  end
  g = parallel_beam(; angles=range(0, pi, 3), nu=4, nv=2, du=1.0)
  @test [ray_index(DataOrder(), j) for j in 1:nthreads(DataOrder(), g)] == 1:nrays(g)
end

@testitem "launch: configuration and defaults" tags = [:operators] begin
  using CTOperators: KA, launch_layout, launch_workgroupsize, ray_order, DataOrder, TileOrder
  # workgroup size and layout of a launch
  settings(op, launch, g, b) = (l = launch_layout(op, launch, g, b); (launch_workgroupsize(op, launch, l, b), l))
  @test LaunchConfig() == LaunchConfig(nothing, nothing)
  @test LaunchConfig(; layout=(4, 8)).layout === (4, 8) && LaunchConfig(; layout=()).layout === ()
  @test_throws ArgumentError LaunchConfig(; workgroupsize=0)
  @test_throws ArgumentError LaunchConfig(; layout=(8, 0))
  image = ImageGeom(; dims=(8, 7, 6))
  g = parallel_beam(; angles=range(0, pi, 3), nu=13, nv=9, du=1.0)
  rays = RayList(rand(3, 5), rand(3, 5))
  cpu = KA.CPU()
  op = JosephProjector(image, g)
  # CPU: data order and the backend's workgroups, also for ray lists
  @test settings(op, nothing, g, cpu) == (nothing, ())
  @test settings(op, LaunchConfig(), rays, cpu) == (nothing, ())
  # tiles: one tile per workgroup, unless the device does not allow workgroups that large; a
  # workgroup size given is kept, and a workgroup size alone keeps the device's layout
  @test settings(op, LaunchConfig(; layout=(4, 8)), g, cpu) == (32, (4, 8))
  @test settings(op, LaunchConfig(; workgroupsize=7, layout=(4, 8)), g, cpu) == (7, (4, 8))
  @test settings(op, LaunchConfig(; workgroupsize=7), g, cpu) == (7, ())
  big = KA.KernelInterface.max_work_group_size(cpu) + 1
  @test settings(op, LaunchConfig(; layout=(big, 1)), g, cpu) == (nothing, (big, 1))
  @test ray_order((4, 8), g) == TileOrder(13, 9, 4, 8)
  @test ray_order((4, 8), rays) == DataOrder()          # ray lists keep their order
  @test ray_order((), g) == DataOrder()
end

# Forward and back projection with tiles and workgroup sizes equal the defaults; the back
# projection only up to rounding (atomic additions in another order)
@testitem "launch: results independent of the configuration" tags = [:operators] setup = [Projections] begin
  using Adapt
  image = ImageGeom(; dims=(8, 7, 6), deltas=(1.0, 1.25, 0.8))
  g = helical_cone_beam(; angles=range(0, 3pi, 5), pitch=4, dso=30, dsd=60, nu=13, nv=9, du=1.17, dv=0.93)
  rays = RayList(g)
  x, y = rand(8, 7, 6), rand(13, 9, 5)
  ray_configs = (LaunchConfig(; layout=(8, 8)), LaunchConfig(; workgroupsize=64, layout=(8, 8)),
                 LaunchConfig(; workgroupsize=32, layout=(4, 2)), LaunchConfig(; workgroupsize=7))
  # runs of voxels: the hits of a run follow from the first by adding steps, rounded differently
  voxel_configs = (LaunchConfig(; layout=(1,)), LaunchConfig(; layout=(3,)), LaunchConfig(; workgroupsize=7, layout=(8,)),
                   LaunchConfig(; workgroupsize=7))
  # distance driven and voxel driven need detector cells: no ray lists
  for (P, geom) in ((SiddonProjector, g), (SiddonProjector, rays), (JosephProjector, g), (JosephProjector, rays),
                    (DistanceDrivenProjector, g), (VoxelDrivenProjector, g))
    ref = P(image, geom; eltype=Float64)
    yr = reshape(y, datasize(ref))
    for launch in (P === VoxelDrivenProjector ? voxel_configs : ray_configs)
      op = P(image, geom; eltype=Float64, launch)
      @test same_projection(P, op * x, ref * x)
      @test isapprox(adjoint(op) * yr, adjoint(ref) * yr; rtol=1e-12)
    end
  end
  # the configuration travels with subset and adapt
  launch = LaunchConfig(; workgroupsize=32, layout=(4, 2))
  op = JosephProjector(image, g; launch)
  @test subset(op, [1, 3]).launch === launch
  @test adapt(Array{Float64}, op).launch === launch
  @test JosephProjector(image, g).launch === nothing
end

# Each projector checks the length of the layout: tiles (tu, tv) for the ray-driven projectors, runs
# (R,) for the voxel-driven one and FDK; () and nothing (the default) for all. The voxel-driven
# defaults on the CPU: runs of 8 voxels, the backend's workgroups (runs ask for no workgroup size).
@testitem "launch: layouts of the projectors" tags = [:operators] begin
  using CTOperators: KA, launch_layout, launch_workgroupsize, voxels_per_thread
  # workgroup size and layout of a launch
  settings(op, launch, g, b) = (l = launch_layout(op, launch, g, b); (launch_workgroupsize(op, launch, l, b), l))
  image = ImageGeom(; dims=(8, 7, 6))
  g = circular_cone_beam(; angles=range(0, 2pi, 5)[1:4], dso=30, dsd=60, nu=13, nv=9, du=1.0)
  tiles, runs = LaunchConfig(; layout=(4, 2)), LaunchConfig(; layout=(3,))
  for P in (SiddonProjector, JosephProjector, DistanceDrivenProjector)
    @test P(image, g; launch=tiles).launch === tiles && P(image, g; launch=LaunchConfig()).launch == LaunchConfig()
    @test_throws ArgumentError P(image, g; launch=runs)
  end
  @test VoxelDrivenProjector(image, g; launch=runs).launch === runs
  @test VoxelDrivenProjector(image, g; launch=LaunchConfig()).launch == LaunchConfig()
  @test_throws ArgumentError VoxelDrivenProjector(image, g; launch=tiles)
  @test_throws ArgumentError VoxelDrivenProjector(image, g; launch=LaunchConfig(; layout=(2, 2, 2)))
  @test_throws ArgumentError FDK(image, g; launch=tiles)
  vop = VoxelDrivenProjector(image, g)
  @test settings(vop, nothing, g, KA.CPU()) == (nothing, (8,))
  @test settings(vop, runs, g, KA.CPU()) == (nothing, (3,))
  @test settings(vop, LaunchConfig(; workgroupsize=16), g, KA.CPU()) == (16, (8,))
  @test voxels_per_thread(()) == 1 && voxels_per_thread((3,)) == 3
end

# A workgroup size larger than the device allows is not changed: KernelAbstractions rejects it when
# the projector runs, forward and back.
@testitem "launch: too large workgroups fail" tags = [:operators] begin
  using CTOperators: KA
  image = ImageGeom(; dims=(8, 7, 6))
  g = circular_cone_beam(; angles=range(0, 2pi, 5)[1:4], dso=30, dsd=60, nu=13, nv=9, du=1.0)
  launch = LaunchConfig(; workgroupsize=KA.KernelInterface.max_work_group_size(KA.CPU()) + 1)
  x, y = rand(8, 7, 6), rand(13, 9, 4)
  for P in (SiddonProjector, JosephProjector, DistanceDrivenProjector, VoxelDrivenProjector)
    op = P(image, g; launch)
    @test_throws ArgumentError op * x
    @test_throws ArgumentError adjoint(op) * y
  end
end

