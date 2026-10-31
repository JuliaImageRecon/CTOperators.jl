# CTOperators.jl

*Matched forward and back projectors for X-ray CT on CPUs and GPUs.*

CTOperators.jl provides projection operators for tomographic reconstruction, written once with
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) and run multi-threaded
on the CPU and on GPUs (CUDA, AMD, Apple). An operator holds its geometry and workspace and is
reused by every application, the setting of iterative solvers such as those of
[RegularizedLeastSquares.jl](https://github.com/JuliaImageRecon/RegularizedLeastSquares.jl).

!!! warning
    CTOperators.jl is in an early stage of development and not registered yet; its interface may
    still change.

## Features

- **Projectors for iterative reconstruction**: geometry and workspace are set up once; the back
  projection is the exact transpose of the forward projection.
- **Siddon's method** (in the incremental form of Jacobs et al.): exact intersection lengths of rays
  and voxels, see [Siddon's method](@ref). Footprint-based projectors are planned.
- **Generic geometries**: arbitrary rays given by start and end points, and per-view descriptions of
  cone-beam and parallel-beam scanners with any trajectory: circular, helical, or a moving object.
- **One code base for CPU and GPU.**
- **Differentiable** with respect to image and projection data (ChainRulesCore rules).

[Getting Started](@ref) simulates a helical cone-beam scan and reconstructs it with
RegularizedLeastSquares.jl.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/nHackel/CTOperators.jl")
```
