# How the projection kernels are launched: the workgroup size and the order in which the threads
# take the rays. Thread j (1-based) traces ray `ray_index(order, j)`, or none if that is 0
# (padding). Both affect the speed on GPUs (rays of neighboring detector cells read neighboring
# voxels), and the best values depend on the device, hence the configuration; the defaults are in
# `default_launch`.

"""
    LaunchConfig(; workgroupsize=nothing, layout=())

How a projector launches its kernels (forward and back projection), for tuning on a particular
device. Pass it to a projector (or to [`FDK`](@ref)) with the keyword `launch`.

* `workgroupsize`: the number of threads per workgroup (threads that run together on one part of
  the device), `nothing` for the choice of KernelAbstractions.jl.
* `layout`: how the work is laid out over the threads, a tuple whose meaning depends on the
  projector; `()` for the plain layout. Each projector checks the length it accepts:
  * [`SiddonProjector`](@ref), [`JosephProjector`](@ref), [`DistanceDrivenProjector`](@ref): `()`
    or `(tu, tv)`. Every thread traces one ray. With `(tu, tv)` the threads of a workgroup take the
    rays of tiles of `tu × tv` detector cells instead of consecutive cells of a detector row
    (detector geometries only; ray lists keep their order). Tiles at the edges of the detector are
    padded with idle threads; a workgroup size of `tu * tv` gives one tile per workgroup.
  * [`VoxelDrivenProjector`](@ref) (and [`FDK`](@ref), which back projects with it): `()` or `(R,)`.
    With `(R,)` every thread takes a run of `R` neighboring voxels along `x` instead of one voxel
    (the last run of a row may be shorter); `()` is the same as `(1,)`.

Without `launch` (or with `launch=nothing`) a projector takes the defaults of its device, also after
`adapt` to another device:

* ray-driven projectors: on GPUs workgroups of 64 and tiles of 8 × 8 cells, on the CPU the plain
  layout and the backend's workgroups;
* the voxel-driven projector: on GPUs workgroups of 64 and runs of 2 voxels, on the CPU runs of
  8 voxels and the backend's workgroups.

The defaults were tuned on one AMD GPU and one AMD CPU; tune them for yours. See the explanation
page [Workgroups and Ray Order on GPUs](@ref) in the documentation.

# Example

```julia
using CUDA
launch = LaunchConfig(; workgroupsize=128, layout=(16, 8))
op = JosephProjector(image, geometry; storage=CuArray, launch)
```
"""
struct LaunchConfig{N}
  workgroupsize::Union{Nothing,Int}
  layout::NTuple{N,Int}
  function LaunchConfig{N}(workgroupsize, layout::NTuple{N,Integer}) where {N}
    isnothing(workgroupsize) || workgroupsize > 0 ||
      throw(ArgumentError("workgroupsize must be positive, got $workgroupsize"))
    all(>(0), layout) || throw(ArgumentError("the entries of layout must be positive, got $layout"))
    new{N}(workgroupsize, Int.(layout))
  end
end

LaunchConfig(; workgroupsize=nothing, layout::Tuple{Vararg{Integer}}=()) =
  LaunchConfig{length(layout)}(workgroupsize, layout)

# The layouts a projector accepts: `lengths` the accepted lengths of the layout, `meaning` what the
# layout means for it (for the message).
function check_layout(launch, name, lengths, meaning)
  isnothing(launch) || length(launch.layout) in lengths ||
    throw(ArgumentError("$name: the layout of LaunchConfig must be $meaning, got $(launch.layout)"))
  launch
end

check_ray_layout(launch, name) = check_layout(launch, name, (0, 2), "() or (tu, tv), tiles of detector cells")

# the configuration of a launch: the projector's own or the defaults of the device
launch_config(launch::LaunchConfig, geometry, backend) = launch
launch_config(::Nothing, geometry, backend) = default_launch(geometry, backend)

# (KA.GPU is no help: since KernelAbstractions 0.10 it is the same as KA.Backend, the CPU included)
function default_launch(geometry, backend)
  backend isa KA.CPU && return LaunchConfig()
  LaunchConfig(; workgroupsize=64, layout=geometry isa Union{AbstractDetectorGeometry,MultiRay} ? (8, 8) : ())
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

# tiles only for detector geometries (also with several rays per cell); ray lists keep their order
ray_order(launch::LaunchConfig, geometry) = DataOrder()
ray_order(launch::LaunchConfig{2}, geometry::Union{AbstractDetectorGeometry,MultiRay}) =
  TileOrder(detectorsize(geometry)..., launch.layout...)

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

# Launches `kernel(out, in, grid, geometry, order, raymask, args...)` of the projector `op` over its
# rays; `raymask` is `nothing` or the ray mask on the device (the kernels skip masked rays). Further
# arguments `args...` of a kernel are passed after `raymask`, in order (positional, as kernels of
# KernelAbstractions.jl take no keywords).
function launch_rays!(kernel, op, out, in, args...)
  g = geometry(op)
  nrays(g) == 0 && return out
  backend = KA.get_backend(op)
  config = launch_config(op.launch, g, backend)
  order = ray_order(config, g)
  wg = config.workgroupsize
  k = isnothing(wg) ? kernel(backend) : kernel(backend, wg)
  k(out, in, op.grid, g, order, ray_mask(op), args...; ndrange=nthreads(order, g))
  out
end
