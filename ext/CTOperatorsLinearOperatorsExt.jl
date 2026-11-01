# LinearOperators.jl: the storage type of CT operators, so that the operators of
# LinearOperatorCollection.jl can take them as parents, e.g. the normal operator
# `normalOperator(op)` (forward and back projection through one projection buffer on the operator's
# device) for the keyword `AHA` of RegularizedLeastSquares.jl's CGNR.
module CTOperatorsLinearOperatorsExt

using CTOperators, LinearOperators
using CTOperators: KA

# the vector type of the operator's device and element type, e.g. Vector{Float32} or
# CuVector{Float32}
LinearOperators.storage_type(op::AbstractCTOperator) =
  typeof(vec(KA.allocate(KA.get_backend(op), eltype(op), 0)))

end
