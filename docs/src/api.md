# API

The sections follow the parts of the package: the image geometry, the projection geometries and
their rays, the generic operator interface, and one section per projector.

## Image geometry

Images are described by `ImageGeom` from [ImageGeoms.jl](https://github.com/JuliaImageRecon/ImageGeoms.jl),
re-exported by CTOperators.jl. Voxel `i` has its centre at `((i - 1) - (n - 1)/2 - offset) * Δ` per
dimension; a 2D image is treated as a volume of thickness one voxel.

## Projection geometries

### Interface

```@autodocs
Modules = [CTOperators]
Pages = ["geometry/geometry.jl"]
Private = false
```

### Geometry types

```@autodocs
Modules = [CTOperators]
Pages = ["geometry/raylist.jl", "geometry/conebeam.jl", "geometry/arcconebeam.jl", "geometry/parallelbeam.jl"]
Private = false
```

### Trajectories

```@autodocs
Modules = [CTOperators]
Pages = ["geometry/trajectories.jl"]
Private = false
```

## Operators

### Generic interface

```@docs
AbstractCTOperator
```

```@autodocs
Modules = [CTOperators]
Pages = ["operators/operators.jl"]
Private = false
```

### Siddon

```@autodocs
Modules = [CTOperators]
Pages = ["operators/siddon/siddon.jl", "operators/siddon/kernels.jl"]
Private = false
```

### Joseph

```@autodocs
Modules = [CTOperators]
Pages = ["operators/joseph/joseph.jl", "operators/joseph/kernels.jl"]
Private = false
```

## Internals

For developers of operators: the kernel form of the image grid, clipping of rays and the
functions each operator implements.

```@autodocs
Modules = [CTOperators]
Public = false
```
