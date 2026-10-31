module CTOperators

using LinearAlgebra
using StaticArrays
using Adapt
using KernelAbstractions
using Atomix
const KA = KernelAbstractions

export AbstractCTOperator

"""
    AbstractCTOperator{T}

Supertype of the CT projection operators of CTOperators.jl. `T` is the real element type of image
and projection data. An operator holds its geometry on the device and the workspace of its
kernels, created once and reused by every application.
"""
abstract type AbstractCTOperator{T} end

end
