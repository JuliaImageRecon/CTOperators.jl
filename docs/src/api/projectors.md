# Projectors

The projectors of the package, all with an exact adjoint. Siddon and Joseph trace the rays of a
geometry through the voxel grid, one per detector cell or several with detector sampling ([`sampled`](@ref));
[Siddon's Method](@ref) and [Joseph's Method](@ref) explain how they weight the voxels. The
distance-driven method integrates over the area of each detector cell, and the voxel-driven method
projects each voxel onto the detector; both need a detector geometry. [System Matrix](@ref)
describes what they have in common.

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

## Distance-Driven

```@autodocs
Modules = [CTOperators]
Pages = ["operators/distancedriven/distancedriven.jl", "operators/distancedriven/kernels.jl"]
Private = false
```

## Voxel-Driven

The projector and its scales: how the bilinear weights of a voxel are scaled per view (see
[Voxel-Driven Method](@ref)).

```@autodocs
Modules = [CTOperators]
Pages = ["operators/voxeldriven/voxeldriven.jl", "operators/voxeldriven/kernels.jl"]
Private = false
```

## FDK

Reconstruction in one pass for circular scans, with the voxel-driven back projection (see
[Reconstruct with FDK](@ref) and [Filtered Back Projection](@ref)), and the redundancy weighting: how
lines measured more than once are weighted.

```@autodocs
Modules = [CTOperators]
Pages = ["fdk/fdk.jl", "fdk/redundancy.jl", "fdk/scan.jl"]
Private = false
```
