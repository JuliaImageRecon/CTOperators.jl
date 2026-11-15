# How the projection kernels are launched: the workgroup size and the layout of the work over the
# threads (for the ray-driven projectors the order in which the threads take the rays: thread j
# (1-based) traces ray `ray_index(order, j)`, or none if that is 0, padding). Both affect the speed,
# and the best values depend on the device, hence the configuration. What it does not give is
# derived per projector: the layout from the device (`launch_layout`), the workgroup size from the
# layout if the device allows it, else KernelAbstractions.jl's choice (`launch_workgroupsize`). The
# methods per projector are next to the projectors (operators.jl, voxeldriven/kernels.jl).

"""
    LaunchConfig(; workgroupsize=nothing, layout=nothing)

How a projector launches its kernels (forward and back projection), for tuning on a particular
device. Pass it to a projector (or to [`FDK`](@ref)) with the keyword `launch`.

* `layout`: how the work is laid out over the threads, a tuple whose meaning depends on the
  projector; `nothing` for the default of the device. Each projector checks the length it accepts:
  * [`SiddonProjector`](@ref), [`JosephProjector`](@ref), [`DistanceDrivenProjector`](@ref): `()`
    or `(tu, tv)`. Every thread traces one ray. With `()` the threads take the rays in the order of
    the data (consecutive cells of a detector row); with `(tu, tv)` the threads take the rays of
    tiles of `tu × tv` detector cells (detector geometries only; ray lists keep their order). Tiles
    at the edges of the detector are padded with idle threads.
  * [`VoxelDrivenProjector`](@ref) (and [`FDK`](@ref), which back projects with it): `()` or `(R,)`.
    With `(R,)` every thread takes a run of `R` neighboring voxels along `x` (the last run of a row
    may be shorter); `()` is one voxel per thread, the same as `(1,)`.
* `workgroupsize`: the number of threads per workgroup (threads that run together on one part of
  the device). `nothing` derives it: for tiles `tu * tv`, one tile per workgroup, if the device
  allows workgroups that large; otherwise the choice of KernelAbstractions.jl. A number is used as
  given; one the device or the kernel does not allow gives an error when the projector runs.

The defaults (`launch=nothing` or `LaunchConfig()`) of the device the projector is on, also after
`adapt` to another device:

* ray-driven projectors: on GPUs tiles of 8 × 8 cells (in workgroups of 64), on the CPU the order
  of the data;
* the voxel-driven projector: on GPUs one voxel per thread, on the CPU runs of 8 voxels.

The layouts were chosen on one AMD GPU and one AMD CPU; tune them for yours. See the explanation
page [Workgroups and Ray Order on GPUs](@ref) in the documentation.

# Example

```julia
using CUDA
launch = LaunchConfig(; layout=(16, 8))        # workgroups of 128 threads, one tile each
op = JosephProjector(image, geometry; storage=CuArray, launch)
```
"""
struct LaunchConfig{L<:Union{Nothing,Tuple{Vararg{Int}}}}
  workgroupsize::Union{Nothing,Int}
  layout::L
  function LaunchConfig(workgroupsize::Union{Nothing,Integer}, layout::Union{Nothing,Tuple{Vararg{Integer}}})
    isnothing(workgroupsize) || workgroupsize > 0 ||
      throw(ArgumentError("workgroupsize must be positive, got $workgroupsize"))
    isnothing(layout) || all(>(0), layout) ||
      throw(ArgumentError("the entries of layout must be positive, got $layout"))
    l = isnothing(layout) ? nothing : Int.(layout)
    new{typeof(l)}(isnothing(workgroupsize) ? nothing : Int(workgroupsize), l)
  end
end

LaunchConfig(; workgroupsize=nothing, layout=nothing) = LaunchConfig(workgroupsize, layout)

# The layouts a projector accepts: `lengths` the accepted lengths of the layout, `meaning` what the
# layout means for it (for the message).
function check_layout(launch, name, lengths, meaning)
  isnothing(launch) || isnothing(launch.layout) || length(launch.layout) in lengths ||
    throw(ArgumentError("$name: the layout of LaunchConfig must be $meaning, got $(launch.layout)"))
  launch
end

check_ray_layout(launch, name) = check_layout(launch, name, (0, 2), "() or (tu, tv), tiles of detector cells")

# The layout of a launch of `op`: that of the configuration `launch` (`nothing`: LaunchConfig()), or
# else the projector's default on the device, `launch_layout(op, geometry, backend)`.
function launch_layout(op, launch, geometry, backend)
  layout = isnothing(launch) ? nothing : launch.layout
  isnothing(layout) ? launch_layout(op, geometry, backend) : layout
end

# The workgroup size of a launch of `op` with `layout`: that of the configuration `launch`, or else
# the size the projector asks for, `launch_workgroupsize(op, layout)` (an Int, or `nothing` for none),
# if the device allows workgroups that large; otherwise `nothing`, KernelAbstractions' choice. The
# compiled kernel's own limit can be lower; KernelAbstractions checks it at the launch.
function launch_workgroupsize(op, launch, layout, backend)
  given = isnothing(launch) ? nothing : launch.workgroupsize
  isnothing(given) || return given
  wanted = launch_workgroupsize(op, layout)
  isnothing(wanted) || wanted > KA.KernelInterface.max_work_group_size(backend) ? nothing : wanted
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
ray_order(layout, geometry) = DataOrder()
ray_order(layout::NTuple{2,Int}, geometry::Union{AbstractDetectorGeometry,MultiRay}) =
  TileOrder(detectorsize(geometry)..., layout...)

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
  layout = launch_layout(op, op.launch, g, backend)
  wg = launch_workgroupsize(op, op.launch, layout, backend)
  order = ray_order(layout, g)
  k = isnothing(wg) ? kernel(backend) : kernel(backend, wg)
  k(out, in, op.grid, g, order, ray_mask(op), args...; ndrange=nthreads(order, g))
  out
end
