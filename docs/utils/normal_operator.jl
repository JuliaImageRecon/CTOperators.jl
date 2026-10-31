# The normal operator adjoint(op) * op for RegularizedLeastSquares.jl (CGNR builds it by default).
# Included invisibly by the documentation examples. It will be provided by LinearOperatorCollection.jl
# (its NormalOp) once that package wraps CTOperators.jl; until then the examples define it here.

using CTOperators, LinearAlgebra

# Applies the forward and the back projection through a projection buffer it holds (on the
# operator's device), so repeated applications allocate nothing.
struct NormalCTOperator{T,O<:AbstractCTOperator{T},B} <: AbstractCTOperator{T}
  parent::O
  buffer::B
end

normal_operator(op::AbstractCTOperator) =
  NormalCTOperator(op, CTOperators.KA.allocate(CTOperators.KA.get_backend(op), eltype(op), datasize(op)))

function Base.:*(a::CTOperators.AdjointCTOperator, op::AbstractCTOperator)
  a.parent === op || throw(ArgumentError("only adjoint(op) * op (the normal operator) is supported"))
  normal_operator(op)
end

CTOperators.imagesize(n::NormalCTOperator) = imagesize(n.parent)
CTOperators.datasize(n::NormalCTOperator) = imagesize(n.parent)
CTOperators.KA.get_backend(n::NormalCTOperator) = CTOperators.KA.get_backend(n.parent)
CTOperators.forward!(z, n::NormalCTOperator, x) =
  CTOperators.backward!(z, n.parent, CTOperators.forward!(n.buffer, n.parent, x))
CTOperators.backward!(z, n::NormalCTOperator, x) = CTOperators.forward!(z, n, x)   # self-adjoint
LinearAlgebra.adjoint(n::NormalCTOperator) = n
