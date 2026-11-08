# LinearOperatorCollection.jl: `normalOperator(op)` and `NormalOp` of the projectors of this
# package build a normal operator that applies Aᴴ W A in one pass over the rays
# (`CTOperators.fused_normal!`), without the buffer for the projection data that the generic
# normal operator holds. Weights given as a vector or a `WeightingOp` are applied per ray; other
# weighting operators use the generic normal operator.
module CTOperatorsLinearOperatorCollectionExt

using CTOperators, LinearOperatorCollection, LinearOperators, Adapt
using CTOperators: fused_normal!
using LinearOperatorCollection: WeightingOp
import LinearOperatorCollection: NormalOp

const FusedProjector = Union{SiddonProjector,JosephProjector,DistanceDrivenProjector}

# The fields of an operator of LinearOperators.jl (those of LinearOperatorCollection's NormalOpImpl),
# plus the projector and the weights. With real weights Aᴴ W A is Hermitian, and symmetric as the
# projectors are real; LinearOperators.jl then takes the operator as its own adjoint and transpose.
mutable struct FusedNormalOp{T,vecT,P,W} <: NormalOp{T,P}
  const nrow::Int
  const ncol::Int
  const symmetric::Bool
  const hermitian::Bool
  const prod!::Function
  const tprod!::Nothing
  const ctprod!::Nothing
  nprod::Int
  ntprod::Int
  nctprod::Int
  Mv::vecT
  Mtu::vecT
  const parent::P
  const weights::W
end

function FusedNormalOp(parent::FusedProjector, weights::Union{Nothing,WeightingOp})
  S = LinearOperators.storage_type(parent)
  w = isnothing(weights) ? nothing : weights.weights
  n = size(parent, 2)
  real_weights = isnothing(w) || eltype(w) <: Real
  FusedNormalOp{eltype(parent),S,typeof(parent),typeof(weights)}(n, n, real_weights, real_weights,
    (z, x) -> fused_normal!(z, parent, x, w), nothing, nothing, 0, 0, 0, S(undef, 0), S(undef, 0),
    parent, weights)
end

NormalOp(::Union{Type{T},Type{Complex{T}}}, parent::FusedProjector, weights::Union{Nothing,WeightingOp};
         kwargs...) where {T} = FusedNormalOp(parent, weights)
NormalOp(::Union{Type{T},Type{Complex{T}}}, parent::FusedProjector, weights::AbstractVector{T}) where {T} =
  FusedNormalOp(parent, WeightingOp(weights))

LinearOperators.storage_type(::FusedNormalOp{T,vecT}) where {T,vecT} = vecT
Base.copy(op::FusedNormalOp) = FusedNormalOp(op.parent, op.weights)
Adapt.adapt_structure(to, op::FusedNormalOp) = FusedNormalOp(adapt(to, op.parent), adapt(to, op.weights))

end
