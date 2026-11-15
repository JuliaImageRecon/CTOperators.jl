# CTOperators.jl: FDK on an NVIDIA GPU (CUDA), Float32, plan and data on the device.

using CUDA

# the GPU to run on, before any array is created on it
CTOperators.KA.device!(CUDABackend(), DEVICE)
const STORAGE = CuArray
include("ctoperators-fdk-plan.jl")

device_name() = CUDA.name(CUDA.device())
device_array(::FDK, a) = CuArray(a)
synchronize(::FDK) = CUDA.synchronize()
