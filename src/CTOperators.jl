module CTOperators

using LinearAlgebra
using StaticArrays
using Adapt
using KernelAbstractions
using Atomix
using ImageGeoms: ImageGeom, axis
const KA = KernelAbstractions

export AbstractCTOperator
export ImageGeom
export AbstractProjectionGeometry, AbstractDetectorGeometry, RaySegment, RayLine
export RayList, ConeBeam, ParallelBeam
export circular_cone_beam, helical_cone_beam, parallel_beam
export datasize, nrays, ray, raytype, nviews, detectorsize, subset

"""
    AbstractCTOperator{T}

Supertype of the CT projection operators of CTOperators.jl. `T` is the real element type of image
and projection data. An operator holds its geometry on the device and the workspace of its
kernels, created once and reused by every application.
"""
abstract type AbstractCTOperator{T} end

include("image/voxelgrid.jl")
include("geometry/geometry.jl")
include("geometry/raylist.jl")
include("geometry/conebeam.jl")
include("geometry/parallelbeam.jl")
include("geometry/trajectories.jl")

end
