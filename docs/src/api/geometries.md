# API

The sections follow the parts of the package: the image geometry, the projection geometries and
their rays, the generic operator interface, and one section per projector.

## Image geometry

Images are described by `ImageGeom` from [ImageGeoms.jl](https://github.com/JuliaImageRecon/ImageGeoms.jl),
re-exported by CTOperators.jl. Voxel `i` has its center at `((i - 1) - (n - 1)/2 - offset) * Δ` per
dimension; a 2D image is treated as a slice at `z = 0` that is 1 length unit thick. Lengths have
no fixed unit: image and projection geometry must use the same one.

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
Pages = ["operators/operators.jl", "operators/launch.jl"]
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
