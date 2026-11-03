# Projectors

The projectors of the package. Both trace one ray per detector cell through the voxel grid and
have an exact adjoint; [Siddon's Method](@ref) and [Joseph's Method](@ref) explain how they
weight the voxels.

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
