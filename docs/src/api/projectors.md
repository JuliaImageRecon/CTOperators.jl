# Projectors

The projectors of the package, all with an exact adjoint. Siddon and Joseph trace the rays of a
geometry through the voxel grid, one per detector cell or several with [`MultiRay`](@ref);
[Siddon's Method](@ref) and [Joseph's Method](@ref) explain how they weight the voxels. Distance
driven integrates over the area of each detector cell and needs a detector geometry.

## Siddon

```@autodocs
Modules = [CTOperators]
Pages = ["operators/siddon/siddon.jl", "operators/siddon/kernels.jl"]
Private = false
```

## Joseph

```@autodocs
Modules = [CTOperators]
Pages = ["operators/joseph/joseph.jl", "operators/joseph/kernels.jl"]
Private = false
```

## Distance Driven

```@autodocs
Modules = [CTOperators]
Pages = ["operators/distancedriven/distancedriven.jl", "operators/distancedriven/kernels.jl"]
Private = false
```
