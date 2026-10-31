# CTOperators.jl

*Matched forward and back projectors for X-ray CT on CPUs and GPUs.*

CTOperators.jl provides projection operators for tomographic reconstruction, written once with
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) and run multi-threaded
on the CPU and on GPUs (CUDA, AMD, Apple). An operator holds its geometry and workspace and is
reused by every application, the setting of iterative solvers such as those of
[RegularizedLeastSquares.jl](https://github.com/JuliaImageRecon/RegularizedLeastSquares.jl).

!!! warning
    CTOperators.jl is in an early stage of development; its interface may still change.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/nHackel/CTOperators.jl")
```
