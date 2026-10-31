# CTOperators.jl

This package provides matched forward and back projectors for X-ray computed tomography that run
multi-threaded on the CPU and on GPUs (CUDA, AMD, Apple) from the same code base.

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://nHackel.github.io/CTOperators.jl/dev)
[![Build Status](https://github.com/nHackel/CTOperators.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/nHackel/CTOperators.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/nHackel/CTOperators.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/nHackel/CTOperators.jl)

> [!IMPORTANT]
> CTOperators.jl is in an early stage of development and not registered yet. Its interface may
> still change.

## Features (planned)

- Projection operators for iterative reconstruction: geometry and workspace are set up once and
  reused by every forward and back projection.
- Exact adjoints: the back projection is the transpose of the forward projection.
- Generic geometries: arbitrary rays given by start and end points, and compact per-view
  descriptions of cone-beam and parallel-beam scanners with any trajectory.
- One code base for CPU and GPU, written with
  [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl).
- Part of the [JuliaImageRecon](https://github.com/JuliaImageRecon) family: use the operators with
  [RegularizedLeastSquares.jl](https://github.com/JuliaImageRecon/RegularizedLeastSquares.jl) and
  [LinearOperatorCollection.jl](https://github.com/JuliaImageRecon/LinearOperatorCollection.jl).

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/nHackel/CTOperators.jl")
```

## Community Standards

This project is part of the Julia community and follows the
[Julia community standards](https://julialang.org/community/standards/).
