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
# usual and copied to the device by the constructor, in the element type `eltype` (default `Float32`):
#
# ```julia
# using CTOperators, CUDA
# image = ImageGeom(; dims=(256, 256, 256), deltas=(1.0, 1.0, 1.0))
# geometry = circular_cone_beam(; angles=range(0, 2pi, 721)[1:720], dso=500, dsd=1000,
#                               nu=512, nv=512, du=0.8)
# op = JosephProjector(image, geometry; storage=CuArray)
# ```
#
# For AMD GPUs use `ROCArray` (AMDGPU.jl), for Apple GPUs `MtlArray` (Metal.jl). Metal has no
# `Float64`; production use is tested on CPUs and NVIDIA GPUs.
#
# ## Apply it to data on the device
#
# Image and projection data must live on the same device as the projector:
#
# ```julia
# x = CuArray(rand(Float32, 256, 256, 256))
# y = op * x                 # a CuArray of size (512, 512, 720)
# z = adjoint(op) * y        # back projection, a CuArray
# ```
#
# `mul!(y, op, x)` and `mul!(z, adjoint(op), y)` reuse preallocated arrays. Iterative solvers work
# unchanged when their data lives on the device; the solvers of RegularizedLeastSquares.jl allocate
# their work arrays like the measured data `b` (CGNR also needs the normal operator `adjoint(op) * op`,
# see [Getting Started](@ref)):
#
# ```julia
# solver = createLinearSolver(CGNR, op; iterations=20)
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
# `Float32` is the default and the right choice on GPUs; `Float64` (`eltype=Float64`) on the CPU or
# on GPUs with fast double precision gives reference results. The element types of the data must
# match the projector's.
#
# ## Performance notes
#
# - Forward projections run one thread per ray and need no atomic operations. Back projections add
#   into the image from many rays at once with atomic additions.
# - The projector holds no large workspace: its memory is the geometry (a few numbers per view for
#   detector geometries, two points per ray for ray lists).
# - Creating a projector is cheap; for ordered subsets, create the projectors of all subsets once
#   (see [Ordered subsets and normalisation](@ref)).
