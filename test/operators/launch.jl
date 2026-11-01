# Launch configurations (src/operators/launch.jl): the order in which threads take the rays and the
# workgroup size change how the kernels run, never their results.

# Every ray exactly once, padding threads none, for detectors that are no multiple of the tile
@testitem "launch: tiles cover every ray once" tags = [:operators] begin
  using CTOperators: TileOrder, DataOrder, nthreads, ray_index
  for (nu, nv, tu, tv) in ((13, 9, 8, 8), (16, 8, 8, 8), (5, 1, 8, 8), (7, 3, 4, 2), (1, 1, 8, 8))
    g = parallel_beam(; angles=range(0, pi, 3), nu, nv, du=1.0)
    o = TileOrder(nu, nv, tu, tv)
    idx = [ray_index(o, j) for j in 1:nthreads(o, g)]
    @test nthreads(o, g) % (tu * tv) == 0
    @test sort(filter(>(0), idx)) == 1:nrays(g)
    # the threads of the first tile take neighbouring cells of the first view
    cells = [CartesianIndices(datasize(g))[i] for i in idx[1:tu*tv] if i > 0]
    @test all(c -> c[1] <= tu && c[2] <= tv && c[3] == 1, cells)
  end
  g = parallel_beam(; angles=range(0, pi, 3), nu=4, nv=2, du=1.0)
  @test [ray_index(DataOrder(), j) for j in 1:nthreads(DataOrder(), g)] == 1:nrays(g)
end

@testitem "launch: configuration and defaults" tags = [:operators] begin
  using CTOperators: KA, launch_config, ray_order, DataOrder, TileOrder
  @test LaunchConfig() == LaunchConfig(nothing, nothing)
  @test_throws ArgumentError LaunchConfig(; workgroupsize=0)
  @test_throws ArgumentError LaunchConfig(; tile=(8, 0))
  g = parallel_beam(; angles=range(0, pi, 3), nu=13, nv=9, du=1.0)
  rays = RayList(rand(3, 5), rand(3, 5))
  # CPU: the backend's workgroups, data order; an explicit configuration is kept
  @test launch_config(nothing, g, KA.CPU()) == LaunchConfig()
  @test launch_config(nothing, rays, KA.CPU()) == LaunchConfig()
  config = LaunchConfig(; workgroupsize=32, tile=(4, 8))
  @test launch_config(config, g, KA.CPU()) === config
  @test ray_order(config, g) == TileOrder(13, 9, 4, 8)
  @test ray_order(config, rays) == DataOrder()          # ray lists keep their order
  @test ray_order(LaunchConfig(), g) == DataOrder()
end

# Forward and back projection with tiles and workgroup sizes equal the defaults; the back
# projection only up to rounding (atomic additions in another order)
@testitem "launch: results independent of the configuration" tags = [:operators] begin
  using Adapt
  image = ImageGeom(; dims=(8, 7, 6), deltas=(1.0, 1.25, 0.8))
  g = helical_cone_beam(; angles=range(0, 3pi, 5), pitch=4, dso=30, dsd=60, nu=13, nv=9, du=1.17, dv=0.93)
  rays = RayList(g)
  x, y = rand(8, 7, 6), rand(13, 9, 5)
  configs = (LaunchConfig(; tile=(8, 8)), LaunchConfig(; workgroupsize=64, tile=(8, 8)),
             LaunchConfig(; workgroupsize=32, tile=(4, 2)), LaunchConfig(; workgroupsize=7))
  for P in (SiddonProjector, JosephProjector), geom in (g, rays)
    ref = P(image, geom; eltype=Float64)
    yr = reshape(y, datasize(ref))
    for launch in configs
      op = P(image, geom; eltype=Float64, launch)
      @test op * x == ref * x
      @test isapprox(adjoint(op) * yr, adjoint(ref) * yr; rtol=1e-12)
    end
  end
  # the configuration travels with subset and adapt
  launch = LaunchConfig(; workgroupsize=32, tile=(4, 2))
  op = JosephProjector(image, g; launch)
  @test subset(op, [1, 3]).launch === launch
  @test adapt(Array{Float64}, op).launch === launch
  @test JosephProjector(image, g).launch === nothing
end
