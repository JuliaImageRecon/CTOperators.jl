# Setups shared by the test items.

# Ray helpers: the bridge to ImagePhantoms.jl's exact line integrals.
@testsnippet Rays begin
  using LinearAlgebra, StaticArrays

  # Coordinates (u, v, ϕ, θ) of ImagePhantoms.jl's 3D `radon` for the line through a and b. Its
  # lines are u e₁ + v e₃ + t e₂ with e₁ = (cos ϕ, sin ϕ, 0), e₂ = (−sin ϕ cos θ, cos ϕ cos θ, sin θ)
  # (the direction) and e₃ = (sin ϕ sin θ, −cos ϕ sin θ, cos θ).
  function line_coordinates(a, b)
    d = normalize(b - a)
    theta = asin(clamp(d[3], -1, 1))
    phi = atan(-d[1], d[2])
    sphi, cphi = sincos(phi)
    stheta, ctheta = sincos(theta)
    e1 = SVector(cphi, sphi, 0)
    e3 = SVector(sphi * stheta, -cphi * stheta, ctheta)
    dot(a, e1), dot(a, e3), phi, theta
  end

  # distance of the origin from the line through a and b
  line_distance(a, b) = norm(cross(a, normalize(b - a)))

  # all rays of a geometry on the CPU
  allrays(g) = [ray(g, i) for i in 1:nrays(g)]
end

# Array and element types under test: Array (KernelAbstractions' CPU backend) and, with
# test/gpu/run.jl, the backend chosen by CTOPERATORS_TEST_DEVICE: a GPU, or JLArrays (the GPU code
# path run on the CPU).
@testsnippet Backends begin
  using Adapt

  arrayTypes = Any[Array]
  floatTypes = [Float32, Float64]
  let device = get(ENV, "CTOPERATORS_TEST_DEVICE", "")
    if !isempty(device)
      pkg, name = Dict("jlarrays" => ("JLArrays", :JLArray), "cuda" => ("CUDA", :CuArray),
                       "rocm" => ("AMDGPU", :ROCArray), "metal" => ("Metal", :MtlArray))[device]
      push!(arrayTypes, getproperty(Base.require(Base.identify_package(pkg)), name))
      device == "metal" && (floatTypes = [Float32])   # Metal has no Float64
    end
  end
end

# How two forward projections of the same input compare, per projector. The ray-driven projectors
# gather: one thread sums each data value in a fixed order, so repeated projections are bitwise
# equal, and data stored in a lower precision are the result rounded once. The voxel-driven
# projector adds into the data with atomic additions in no fixed order, in the element type of the
# data: equal up to rounding.
@testsnippet Projections begin
  atomic_forward(P) = P <: VoxelDrivenProjector
  same_projection(P, a, b; rtol=sqrt(eps(Float64))) = atomic_forward(P) ? isapprox(a, b; rtol) : a == b
end
