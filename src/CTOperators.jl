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
export RayList, ConeBeam, ArcConeBeam, ParallelBeam
export circular_cone_beam, helical_cone_beam, parallel_beam
export datasize, nrays, ray, raytype, nviews, detectorsize, subset, projection_geometry
export AbstractProjector, AdjointCTOperator, SiddonProjector, JosephProjector, LaunchConfig
export geometry, image_geometry, imagesize, rowsums, colsums

"""
    AbstractCTOperator{T}

Supertype of the CT projection operators of CTOperators.jl. `T` is the real element type of image
and projection data. An operator is created once and applied many times: `op * x` is the forward
projection, `adjoint(op) * y` the back projection (its adjoint; `transpose(op)` is the same).
`op * x` returns an array on the operator's device, a vector for a vector `x` and an array of the
data shape otherwise; `mul!` accepts both shapes.

With LinearOperators.jl loaded, `LinearOperators.storage_type(op)` is the vector type of the
operator's device, so that operators of LinearOperatorCollection.jl take CT operators, e.g. the
normal operator `normalOperator(op)` for the solvers of RegularizedLeastSquares.jl.
"""
abstract type AbstractCTOperator{T} end

include("image/image.jl")
include("geometry/geometry.jl")
include("raytracing/raytracing.jl")
include("operators/operators.jl")

end
