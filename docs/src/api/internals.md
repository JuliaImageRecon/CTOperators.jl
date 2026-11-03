# Internals

For developers of operators (see [Implement a CT Operator](@ref)): the functions each operator
implements, the masks a projector stores, the kernel form of the image grid and the clipping of
rays. None of these is exported.

## Operator implementation

```@autodocs
Modules = [CTOperators]
Pages = ["operators/operators.jl", "operators/launch.jl"]
Public = false
```

## Masks of a projector

```@docs
CTOperators.ProjectorMasks
CTOperators.projector_masks
```

## Geometries

```@autodocs
Modules = [CTOperators]
Pages = ["geometry/geometry.jl", "geometry/raylist.jl", "geometry/conebeam.jl", "geometry/arcconebeam.jl",
         "geometry/parallelbeam.jl", "geometry/trajectories.jl"]
Public = false
```

## Image grid

```@autodocs
Modules = [CTOperators]
Pages = ["image/image.jl", "image/voxelgrid.jl"]
Public = false
```

## Ray tracing

```@autodocs
Modules = [CTOperators]
Pages = ["raytracing/raytracing.jl", "raytracing/clip.jl", "raytracing/actions.jl"]
Public = false
```

## Siddon and Joseph

```@autodocs
Modules = [CTOperators]
Pages = ["operators/siddon/siddon.jl", "operators/siddon/kernels.jl", "operators/joseph/joseph.jl",
         "operators/joseph/kernels.jl"]
Public = false
```
