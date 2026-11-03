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

## Trajectories

Constructors for the common scans.

```@autodocs
Modules = [CTOperators]
Pages = ["geometry/trajectories.jl"]
Private = false
```
