# # Normal Operator
#
# Iterative solvers such as CGNR work with the normal operator ``A^T W A`` of a projector ``A``,
# with ``W`` a diagonal matrix of data weights, one per data element (the identity without data
# weights); the weights of the [System Matrix](@ref) are the entries of ``A``. This page explains
# how the normal operator of the projectors is applied in one pass over the data elements, and what
# that changes. [Ordered Subsets and Normalization](@ref) and the tutorials show it in solvers.

using CTOperators, LinearOperatorCollection, LinearAlgebra

# ## Forward, weight, back
#
# The generic normal operator of LinearOperatorCollection.jl applies its three parts one after the
# other: the forward projection into a buffer of the size of the projection data, the data weights, and
# the back projection of the buffer. The buffer stays allocated as long as the operator exists.
#
# ## One pass over the data elements
#
# The normal operator is a sum over the data elements: with ``a_i`` row ``i`` of ``A`` (the weights
# of data element ``i`` in the voxels) and ``w_i`` its data weight,
#
# ```math
# A^T W A \, x = \sum_i a_i \, w_i \, (a_i^T x).
# ```
#
# Each term needs only its own data element. So one thread per data element can trace its rays and
# sum the image along them (``a_i^T x``), multiply by the data weight, and trace the same rays again
# to add the result to the voxels. No thread waits for another, since the value of a data element is
# complete before its back
# projection starts; the additions into the image are atomic, as in the back projection. The
# projection data are never stored, and one kernel launch replaces two.
#
# With LinearOperatorCollection.jl loaded, `normalOperator` of a [`SiddonProjector`](@ref), a
# [`JosephProjector`](@ref) or a [`DistanceDrivenProjector`](@ref) returns such a fused operator.
# It covers several rays per cell ([`MultiRay`](@ref): all rays of a cell are summed, then traced
# back) and the masks, and takes the weights as a vector or as a `WeightingOp`:

image = ImageGeom(; dims=(32, 32, 16))
geometry = circular_cone_beam(; angles=range(0, 2pi, 61)[1:60], dso=80, dsd=160, nu=48, nv=24, du=1.5)
op = JosephProjector(image, geometry)
w = rand(Float32, nrays(geometry))
N = normalOperator(op, w)
x = rand(Float32, prod(image.dims))
norm(N * x - vec(adjoint(op) * (w .* vec(op * x)))) / norm(N * x)

# The result equals forward, weight and back up to rounding: the atomic additions on a GPU run in no
# fixed order. A weighted projector `ProdOp(WeightingOp(w), op)` is fused as well: its normal
# operator ``A^T W^2 A`` is that of `op` with the squared weights. Other weighting operators, and
# operators other than these projectors, use the generic normal operator. Each solver call stays
# the same, e.g. `AHA=normalOperator(op)` for CGNR of RegularizedLeastSquares.jl.
#
# ## What it changes
#
# * **Memory:** the buffer of the projection data is not needed. For 360 views of 256 × 256 cells it
#   is 90 MiB in Float32, for 1000 views of 1000 × 1000 cells 3.7 GiB.
# * **Time:** each ray is still traced twice, and the back projection with its atomic additions
#   remains. In our
#   [benchmarks](https://github.com/JuliaImageRecon/CTOperators.jl/blob/main/benchmark/README.md#fused-normal-operator)
#   the fused operator took the same time as the separate parts for Siddon's method and about 5 %
#   less for Joseph's on a GPU.
# * **Data weights** cost no extra pass over the data: each data element multiplies its value by its
#   data weight.
