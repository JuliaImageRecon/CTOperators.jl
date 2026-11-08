# # Run on a GPU
#
# The projectors are written once with
# [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) and run on the CPU
# (multi-threaded) and on GPUs from the same code. Where a projector runs is decided when it is
# created, by the array type in which it stores its geometry.
#
# ## Create the projector on the device
#
# Load the GPU package and pass its array type as `storage`. The geometry is created on the CPU as
# usual and copied to the device by the constructor, in the element type `eltype` (default
# `Float32`):
#
# ```julia
# using CTOperators, CUDA
# image = ImageGeom(; dims=(256, 256, 256), deltas=(1.0, 1.0, 1.0))
# geometry = circular_cone_beam(; angles=range(0, 2pi, 721)[1:720], dso=500, dsd=1000,
#                               nu=512, nv=512, du=0.8)
# op = JosephProjector(image, geometry; storage=CuArray)
# ```
#
# For AMD GPUs use `ROCArray` (AMDGPU.jl), for Apple GPUs `MtlArray` (Metal.jl); Metal has no
# `Float64`. The test suite runs on CPUs and on NVIDIA and AMD GPUs.
#
# ## Choose the GPU
#
# On a machine with several GPUs, select one with `device!` of KernelAbstractions.jl (counted from 1)
# *before* creating the projector: the projector stores its geometry on the GPU that is active when
# it is created, and all data passed to it must live there.
#
# ```julia
# using KernelAbstractions: device!, ndevices
# ndevices(CUDABackend())              # number of GPUs
# device!(CUDABackend(), 2)            # the second GPU, for this task
# op = JosephProjector(image, geometry; storage=CuArray)
# ```

# ## Apply it to data on the device
#
# Image and projection data must live on the same device as the projector:
#
# ```julia
# x = CuArray(rand(Float32, 256, 256, 256))
# y = op * x                 # a CuArray of size (512, 512, 720)
# z = adjoint(op) * y        # back projection, a CuArray
# z_cpu = Array(z)           # back to the CPU
# ```
#
# Data on another device than the projector's raises an error from the GPU package (for example
# when a kernel is given a CPU array).
#
# `mul!(y, op, x)` and `mul!(z, adjoint(op), y)` reuse preallocated arrays. Iterative solvers work
# unchanged when their data lives on the device; the solvers of RegularizedLeastSquares.jl allocate
# their work arrays like the measured data. The normal operator of LinearOperatorCollection.jl
# allocates its buffer on the projector's device (see [Getting Started](@ref)):
#
# ```julia
# using RegularizedLeastSquares, LinearOperatorCollection
# solver = createLinearSolver(CGNR, op; AHA=normalOperator(op), iterations=20)
# x = solve!(solver, vec(y))      # y a CuArray: the iterations run on the GPU
# ```
#
# ## Move an existing projector
#
# `adapt` from [Adapt.jl](https://github.com/JuliaGPU/Adapt.jl) creates the same projector with its
# geometry on another device or in another precision:
#
# ```julia
# using Adapt
# op_gpu = adapt(CuArray, op_cpu)
# op_f64 = adapt(Array{Float64}, op_cpu)
# ```
#
# ## Precision
#
# `Float32` is the default; `Float64` (`eltype=Float64`) on the CPU or on GPUs with fast double
# precision gives reference results. The element type of a projector is that of its ray geometry and
# of its arithmetic, and must be at least `Float32`.
#
# Images and data can be stored in a lower precision, such as `Float16`, which halves their memory.
# The projector converts the values when it reads them and sums in its own element type:
#
# ```julia
# x16 = CuArray{Float16}(x)    # an image stored in Float16
# y = op * x16                 # Float32 data
# y16 = similar(y, Float16)
# mul!(y16, op, x16)           # Float16 data: every value rounded once
# x = adjoint(op) * y16        # back projection of Float16 data into a Float32 image
# ```
#
# The image of a back projection collects many contributions per voxel, so it must be at least
# `Float32`; `mul!` into a `Float16` image throws an `ArgumentError`. Timings with `Float16` images
# are in the
# [benchmarks](https://github.com/JuliaImageRecon/CTOperators.jl/blob/main/benchmark/README.md).
#
# ## Performance notes
#
# - Forward projections run one thread per ray and need no atomic operations. Back projections add
#   into the image from many rays at once with atomic additions.
# - The workgroup size and the order in which threads take the rays can be tuned for a device with
#   the keyword `launch`, a [`LaunchConfig`](@ref); the defaults (workgroups of 64, tiles of 8 × 8
#   detector cells) were chosen on an AMD Radeon RX 7900 XT. See
#   [Workgroups and Ray Order on GPUs](@ref).
# - The projector holds no workspace: its memory is the geometry (a few numbers per view for
#   detector geometries, two points per ray for ray lists) and the masks.
# - Creating a projector is cheap; for ordered subsets, create the projectors of all subsets once
#   (see [Ordered Subsets and Normalization](@ref)).
