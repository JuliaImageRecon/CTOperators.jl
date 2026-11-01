# CTOperators.jl

This package provides matched forward and back projectors for X-ray computed tomography that run
multi-threaded on the CPU and on GPUs (CUDA, AMD, Apple) from the same code base.

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaImageRecon.github.io/CTOperators.jl/dev)
[![Build Status](https://github.com/JuliaImageRecon/CTOperators.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaImageRecon/CTOperators.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/JuliaImageRecon/CTOperators.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaImageRecon/CTOperators.jl)

> [!IMPORTANT]
> CTOperators.jl is in an early stage of development and not registered yet. Its interface may
> still change.

## Features

- **Projectors for iterative reconstruction**: a projector is created once and applied many times;
  the back projection is the exact adjoint of the forward projection.
- **Two ray-driven projectors**: Siddon's method, which weights each voxel by the exact length of
  the ray inside it (in the incremental form of Jacobs et al.), and Joseph's method, which
  interpolates linearly between voxel centers.
- **Generic geometries**: arbitrary rays given by start and end points, and cone-beam (flat or
  curved detector) and parallel-beam scanners described view by view, with any source trajectory
  (circular, helical, ...), including objects that move during the scan. Geometries of
  [Sinograms.jl](https://github.com/JuliaImageRecon/Sinograms.jl) work directly.
- **One code base for CPU and GPU**, written with
  [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl): data as `Array`,
  `CuArray` (CUDA.jl), `ROCArray` (AMDGPU.jl) or `MtlArray` (Metal.jl).
- **Differentiable** with respect to image and projection data (ChainRulesCore rules).
- Part of the [JuliaImageRecon](https://github.com/JuliaImageRecon) family: images are described by
  [ImageGeoms.jl](https://github.com/JuliaImageRecon/ImageGeoms.jl), reconstructions run with
  [RegularizedLeastSquares.jl](https://github.com/JuliaImageRecon/RegularizedLeastSquares.jl).

## Example

```julia
using CTOperators

image = ImageGeom(; dims=(128, 128, 128), deltas=(1.0, 1.0, 1.0))
geometry = helical_cone_beam(; angles=range(0, 4pi, 360), pitch=64, z0=-64, dso=256,
                             dsd=512, nu=256, nv=128, du=1.0, dv=1.0)
op = SiddonProjector(image, geometry)        # Float32 on the CPU
x = rand(Float32, 128, 128, 128)
y = op * x                                   # projections, size (256, 128, 360)
z = adjoint(op) * y                          # back projection
```

On a GPU, create the projector with the GPU's array type, e.g. `SiddonProjector(image, geometry;
storage=CuArray)`, and pass data that lives on the GPU.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/JuliaImageRecon/CTOperators.jl")
```

## Benchmarks

`benchmark/` compares CTOperators.jl with [parallelproj](https://github.com/gschramm/parallelproj)
and [RadonKA.jl](https://github.com/roflmaostc/RadonKA.jl) on the CPU and the GPU; its
[README](benchmark/README.md) describes how to run the benchmarks and lists the results.

## Community Standards

This project is part of the Julia community and follows the
[Julia community standards](https://julialang.org/community/standards/).
