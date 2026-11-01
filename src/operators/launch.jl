# How the projection kernels are launched: the workgroup size and the order in which the threads
# take the rays. Thread j (1-based) traces ray `ray_index(order, j)`, or none if that is 0
# (padding). Measured on an RX 7900 XT (benchmark/README.md): workgroups of 64 instead of the
# backend's default and tiles of 8 × 8 detector cells make forward projections at 256^3 up to
# 2.2 times faster; on the CPU neither changes much. The best values depend on the device, hence
# the configuration.

"""
    LaunchConfig(; workgroupsize=nothing, tile=nothing)

How a projector launches its kernels (forward and back projection), for tuning on a particular
device. Pass it to a projector with the keyword `launch`.

* `workgroupsize`: the number of threads (work-items) per workgroup, `nothing` for the choice of
  KernelAbstractions.jl.
* `tile`: `(tu, tv)` to let the threads of a workgroup take the rays of tiles of `tu × tv` detector
  cells instead of consecutive cells of a detector row (detector geometries only; ray lists keep
  their order), `nothing` for data order. Tiles at the edges of the detector are padded with idle
  threads; a workgroup size of `tu * tv` gives one tile per workgroup.

Without `launch` (or with `launch=nothing`) a projector takes the defaults of its device, also after
`adapt` to another device: on GPUs workgroups of 64 and tiles of 8 × 8 cells, on the CPU data order
and the backend's workgroups.

The GPU defaults were tuned on one AMD GPU; tune them for yours. See the explanation page
[Workgroups and Ray Order on GPUs](@ref) in the documentation.

# Example

```julia
using CUDA
launch = LaunchConfig(; workgroupsize=128, tile=(16, 8))
op = JosephProjector(image, geometry; storage=CuArray, launch)
```
"""
struct LaunchConfig
  workgroupsize::Union{Nothing,Int}
  tile::Union{Nothing,NTuple{2,Int}}
  function LaunchConfig(workgroupsize, tile)
    isnothing(workgroupsize) || workgroupsize > 0 ||
      throw(ArgumentError("workgroupsize must be positive, got $workgroupsize"))
    isnothing(tile) || all(>(0), tile) || throw(ArgumentError("tile sizes must be positive, got $tile"))
    new(workgroupsize, tile)
  end
end

LaunchConfig(; workgroupsize=nothing, tile=nothing) = LaunchConfig(workgroupsize, tile)

# the configuration of a launch: the projector's own or the defaults of the device
launch_config(launch::LaunchConfig, geometry, backend) = launch
launch_config(::Nothing, geometry, backend) = default_launch(geometry, backend)

# (KA.GPU is no help: since KernelAbstractions 0.10 it is the same as KA.Backend, the CPU included)
function default_launch(geometry, backend)
  backend isa KA.CPU && return LaunchConfig()
  LaunchConfig(; workgroupsize=64, tile=geometry isa AbstractDetectorGeometry ? (8, 8) : nothing)
end

# the rays in data order
struct DataOrder end

# tiles of tu × tv detector cells in data order (u fastest, then v, then the view), within a tile u
# fastest; the tiles at the detector's edges padded
struct TileOrder
  nu::Int
  nv::Int
  tu::Int
  tv::Int
end

# tiles only for detector geometries; ray lists keep their order
ray_order(launch::LaunchConfig, geometry) = DataOrder()
ray_order(launch::LaunchConfig, geometry::AbstractDetectorGeometry) =
  isnothing(launch.tile) ? DataOrder() : TileOrder(detectorsize(geometry)..., launch.tile...)

nthreads(::DataOrder, geometry) = nrays(geometry)
nthreads(o::TileOrder, geometry) = cld(o.nu, o.tu) * o.tu * cld(o.nv, o.tv) * o.tv * nviews(geometry)

@inline ray_index(::DataOrder, j) = j

@inline function ray_index(o::TileOrder, j)
  ntu, ntv = cld(o.nu, o.tu), cld(o.nv, o.tv)
  k, r = divrem(j - 1, ntu * ntv * o.tu * o.tv)   # view, thread within the view
  t, w = divrem(r, o.tu * o.tv)                   # tile, thread within the tile
  tv, tu = divrem(t, ntu)
  wv, wu = divrem(w, o.tu)
  iu, iv = tu * o.tu + wu, tv * o.tv + wv         # 0-based detector cell
  ifelse(iu < o.nu && iv < o.nv, 1 + iu + o.nu * (iv + o.nv * k), 0)
end

# Launches `kernel(out, in, grid, geometry, order)` of the projector `op` over its rays.
function launch_rays!(kernel, op, out, in)
  g = geometry(op)
  nrays(g) == 0 && return out
  backend = KA.get_backend(op)
  config = launch_config(op.launch, g, backend)
  order = ray_order(config, g)
  wg = config.workgroupsize
  k = isnothing(wg) ? kernel(backend) : kernel(backend, wg)
  k(out, in, op.grid, g, order; ndrange=nthreads(order, g))
  out
end
