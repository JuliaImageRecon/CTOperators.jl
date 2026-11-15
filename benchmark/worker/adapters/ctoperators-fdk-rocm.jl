# CTOperators.jl: FDK on an AMD GPU (ROCm), Float32, plan and data on the device.

using AMDGPU

# the GPU to run on, before any array is created on it
CTOperators.KA.device!(ROCBackend(), DEVICE)
const STORAGE = ROCArray
include("ctoperators-fdk-plan.jl")

device_name() = AMDGPU.HIP.name(AMDGPU.device())
device_array(::FDK, a) = ROCArray(a)
synchronize(::FDK) = AMDGPU.synchronize()
