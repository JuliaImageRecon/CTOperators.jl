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
- **Two ray-driven projectors**: Siddon's method (in the incremental form of Jacobs et al.), exact
  intersection lengths of rays and voxels, and Joseph's method, linear interpolation between the
  voxel centres, see [Siddon's method](@ref) and [Joseph's method](@ref).
  Footprint-based projectors are planned.
- **Generic geometries**: arbitrary rays given by start and end points, and per-view descriptions of
  cone-beam (flat or curved detector) and parallel-beam scanners with any trajectory: circular,
  helical, or a moving object. Geometries of
  [Sinograms.jl](https://github.com/JuliaImageRecon/Sinograms.jl) work directly.
- **One code base for CPU and GPU.**
- **Differentiable** with respect to image and projection data (ChainRulesCore rules).

## Where to go next

- [Getting Started](@ref) simulates a helical cone-beam scan and reconstructs it with
  RegularizedLeastSquares.jl.
- The how-to guides show how to [set up a geometry](@ref "Set up a geometry"), including an object
  that moves during the scan, how to [run on a GPU](@ref "Run on a GPU"), how to build
  [ordered-subset methods](@ref "Ordered subsets and normalisation"), how to
  [differentiate through a projector](@ref "Differentiate through a projector"), how to
  [use geometries of Sinograms.jl](@ref "Use geometries of Sinograms.jl") and how to
  [implement a CT operator](@ref "Implement a CT operator").
- The explanation pages describe the [coordinates and conventions](@ref "Coordinates and conventions")
  and the projection methods, [Siddon's method](@ref) and [Joseph's method](@ref).

## Related packages

- [ImageGeoms.jl](https://github.com/JuliaImageRecon/ImageGeoms.jl) describes the voxel grid of the
  image; CTOperators.jl takes its `ImageGeom` and re-exports it.
- [ImagePhantoms.jl](https://github.com/JuliaImageRecon/ImagePhantoms.jl) provides phantoms and
  their analytic projections; the examples and tests use them.
- [Sinograms.jl](https://github.com/JuliaImageRecon/Sinograms.jl) describes CT geometries and
  provides analytic reconstruction (FBP, FDK); the projectors accept its geometries directly.
- [RadonKA.jl](https://github.com/roflmaostc/RadonKA.jl) computes Radon transforms and back
  projections, parallel-beam (also attenuated) and of arbitrary in-plane rays, slice by slice with
  KernelAbstractions.jl, also on GPUs.
- [SPECTrecon.jl](https://github.com/JuliaImageRecon/SPECTrecon.jl) provides forward and back
  projectors and ML-EM reconstruction for SPECT.
- [LinearOperatorCollection.jl](https://github.com/JuliaImageRecon/LinearOperatorCollection.jl)
  collects linear operators (FFT, NFFT, wavelets, ...) for composing system matrices.
- [RegularizedLeastSquares.jl](https://github.com/JuliaImageRecon/RegularizedLeastSquares.jl)
  solves regularised least-squares problems (CGNR, FISTA, ADMM, Kaczmarz, ...); its solvers work
  with the projectors of CTOperators.jl.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/JuliaImageRecon/CTOperators.jl")
```
