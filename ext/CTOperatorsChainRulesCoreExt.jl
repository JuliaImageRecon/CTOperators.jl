# Reverse-mode rules for applying an operator: the operators are linear, so the pullback of
# y = op * x is adjoint(op) * ybar. The same rule covers adjoint(op), an AbstractCTOperator too.
# No derivatives with respect to the operator (its geometry).
module CTOperatorsChainRulesCoreExt

using CTOperators, ChainRulesCore

function ChainRulesCore.rrule(::typeof(*), op::AbstractCTOperator, x::AbstractArray)
  function times_pullback(ybar)
    xbar = @thunk(reshape(adjoint(op) * unthunk(ybar), size(x)))
    NoTangent(), NoTangent(), xbar
  end
  op * x, times_pullback
end

end
