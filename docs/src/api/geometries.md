# Projection Geometries

A projection geometry describes where the rays run and the shape of the projection data. Detector
geometries (cone beam with a flat or an arc detector, parallel beam) also describe the detector
cells; ray lists hold arbitrary rays. See [Set Up a Geometry](@ref) for examples.

## Interface

```@autodocs
Modules = [CTOperators]
Pages = ["geometry/geometry.jl"]
Private = false
```

## Geometry types

```@autodocs
Modules = [CTOperators]
Pages = ["geometry/raylist.jl", "geometry/conebeam.jl", "geometry/arcconebeam.jl", "geometry/parallelbeam.jl"]
Private = false
```

## Point and weighting strategies

Point strategies (where the rays of a sample end in a detector cell or start in a focal spot) and
weighting strategies, together with the functions that custom strategies implement (see
[Write a Sampling Strategy](@ref)).

```@autodocs
Modules = [CTOperators]
Pages = ["geometry/cellsampled.jl"]
Private = false
```

## Sampling of the beam

The samples of a focal spot and of the detector cells, for [`sampled`](@ref) and
[`stacked`](@ref), and the geometry of a sample.

```@autodocs
Modules = [CTOperators]
Pages = ["geometry/sampling.jl", "geometry/shifted.jl"]
Private = false
```

## Trajectories

Constructors for the common scans.

```@autodocs
Modules = [CTOperators]
Pages = ["geometry/trajectories.jl"]
Private = false
```
