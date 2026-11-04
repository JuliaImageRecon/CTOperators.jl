# Internals

For developers of operators (see [Write a Projector](@ref)): the functions each operator
implements, the masks a projector stores, the kernel form of the image grid and the clipping of
rays. None of these is exported; those that custom components use are declared `public`.

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
         "geometry/parallelbeam.jl", "geometry/trajectories.jl", "geometry/multiray.jl"]
Public = false
```

## Image grid

```@autodocs
Modules = [CTOperators]
Pages = ["image/image.jl", "image/voxelgrid.jl"]
```

## Ray tracing

```@autodocs
Modules = [CTOperators]
Pages = ["raytracing/raytracing.jl", "raytracing/clip.jl", "raytracing/actions.jl", "raytracing/subrays.jl"]
```

## Siddon and Joseph

```@autodocs
Modules = [CTOperators]
Pages = ["operators/siddon/siddon.jl", "operators/siddon/kernels.jl", "operators/joseph/joseph.jl",
         "operators/joseph/kernels.jl"]
Public = false
```
