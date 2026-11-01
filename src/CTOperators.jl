module CTOperators

using LinearAlgebra
using StaticArrays
using Adapt
using KernelAbstractions
using Atomix
using ImageGeoms: ImageGeom, axis, MaskCircle, MaskEllipse, MaskAllButEdge
const KA = KernelAbstractions

export AbstractCTOperator
export ImageGeom, MaskCircle, MaskEllipse, MaskAllButEdge
export AbstractProjectionGeometry, AbstractDetectorGeometry, RaySegment, RayLine
export RayList, ConeBeam, ArcConeBeam, ParallelBeam
export MultiRay, AbstractPoints, RegularPoints, JitteredPoints, RandomPoints, GaussianPoints, Points
export AbstractWeights, UniformWeights, GaussianWeights
export circular_cone_beam, helical_cone_beam, parallel_beam
export datasize, nrays, ray, raytype, nviews, detectorsize, subset, projection_geometry
export AbstractProjector, AdjointCTOperator, SiddonProjector, JosephProjector, DistanceDrivenProjector, LaunchConfig
export geometry, image_geometry, imagesize, rowsums, colsums

# not exported, but part of the interface: for projectors, geometries and strategies of other packages
public forward!, backward!, similar_projector, storage_array, device_geometry
public point_offset, random_pair, points_eltype, subray_weight, weights_eltype
public trace_element, in_plane, one_layer, cell_ray, is_masked, visit, Gather, Scatter, to_index, clip, VoxelGrid, index_type
public image_mask, ray_mask, handles_raymask, projector_masks, ProjectorMasks

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
