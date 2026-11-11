# CTOperators.jl

*Forward and back projectors for X-ray CT on CPUs and GPUs.*

CTOperators.jl provides projection operators for tomographic reconstruction that run
multi-threaded on the CPU and on GPUs (CUDA, AMD, Apple) from the same code. You create an operator
once and apply it many times, which is what iterative solvers such as those of
[RegularizedLeastSquares.jl](https://github.com/JuliaImageRecon/RegularizedLeastSquares.jl) do.
The forward projection `op * x` computes line integrals of an image `x` along the rays of a scan;
the back projection `adjoint(op) * y` is its adjoint.

```julia
using CTOperators
image = ImageGeom(; dims=(128, 128, 128), deltas=(1.0, 1.0, 1.0))
geometry = helical_cone_beam(; angles=range(0, 4pi, 360), pitch=64, z0=-64, dso=256, dsd=512,
                             nu=256, nv=128, du=1.0, dv=1.0)
op = JosephProjector(image, geometry)        # Float32 on the CPU
y = op * rand(Float32, 128, 128, 128)        # projections, size (256, 128, 360)
```

!!! warning
    CTOperators.jl is in an early stage of development; its interface may still change.

## Features

- **Projectors for iterative reconstruction**: a projector is created once and applied many times;
  the back projection is the exact adjoint of the forward projection.
- **Two ray-driven projectors**: Siddon's method, which weights each voxel by the exact length of
  the ray inside it (in the incremental form of Jacobs et al.), and Joseph's method, which
  interpolates linearly between voxel centers; see [Siddon's Method](@ref) and
  [Joseph's Method](@ref).
- **A distance-driven projector** (De Man & Basu), which averages the image over the area each
  detector cell sees, for detector geometries; see [Distance-Driven Method](@ref).
- **A voxel-driven projector**, which projects each voxel center onto the detector and interpolates
  between the cells, for detector geometries: its back projection writes each voxel once, its
  forward projection shows a pattern when the cells are smaller than the voxels; see
  [Voxel-Driven Method](@ref).
- **Generic geometries**: arbitrary rays given by start and end points, and cone-beam (flat or
  curved detector) and parallel-beam scanners described view by view, with any source trajectory
  (circular, helical, ...), including objects that move during the scan. Geometries of
  [Sinograms.jl](https://github.com/JuliaImageRecon/Sinograms.jl) work directly.
- **One code base for CPU and GPU**, written with
  [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl): data as `Array`,
  `CuArray` (CUDA.jl), `ROCArray` (AMDGPU.jl) or `MtlArray` (Metal.jl).
- **Differentiable** with respect to image and projection data (ChainRulesCore rules).

## Where to go next

- [Getting Started](@ref) simulates a cone-beam scan of a moving object and reconstructs it with
  RegularizedLeastSquares.jl.
- The tutorials go through the common scans, each from its geometry to the reconstruction:
  [parallel beam](@ref "Parallel Beam in 2D"), [fan beam](@ref "Fan Beam with a Curved Detector"),
  [circular](@ref "Circular Cone-Beam CT") and [helical](@ref "Helical Cone-Beam CT") cone beam,
  [sparse views and limited angles](@ref "Sparse Views and Limited Angles") and
  [tomosynthesis](@ref "Tomosynthesis").
- The how-to guides show how to [set up a geometry](@ref "Set Up a Geometry"), including an object
  that moves during the scan, how to [run on a GPU](@ref "Run on a GPU") and
  [choose the precision](@ref "Choose the Precision"), how to build
  [ordered-subset methods](@ref "Ordered Subsets and Normalization"), how to
  [use masks](@ref "Use Masks") for the image and the rays, how to
  [use several rays per detector cell](@ref "Use Several Rays per Cell"), how to
  [differentiate through a projector](@ref "Differentiate through a Projector"), how to
  [use geometries of Sinograms.jl](@ref "Use Geometries of Sinograms.jl") and how to
  [write your own projector](@ref "Write a Projector") or [sampling strategy](@ref "Write a Sampling Strategy").
- The explanation pages introduce [the parameters of a scan](@ref "Parameters of a Scan") step by step,
  describe the [coordinates and conventions](@ref "Coordinates and Conventions"),
  what every projector computes (the [system matrix](@ref "System Matrix")), the projection methods
  ([Siddon's Method](@ref), [Joseph's Method](@ref), [several rays per cell](@ref "Several Rays per Cell"),
  the [Distance-Driven Method](@ref) and the [Voxel-Driven Method](@ref)), how they
  [compare](@ref "Comparing Projectors"), and
  [workgroups and ray order on GPUs](@ref "Workgroups and Ray Order on GPUs").
- The reference lists all types and functions by domain: [Images](api/images.md), [Projection Geometries](api/geometries.md), [Operators](api/operators.md), [Projectors](api/projectors.md) and, for implementers, [Internals](api/internals.md).

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
  solves regularized least-squares problems (CGNR, FISTA, ADMM, Kaczmarz, ...); its solvers work
  with the projectors of CTOperators.jl.

## Installation

```julia
using Pkg
Pkg.add("CTOperators")
```
