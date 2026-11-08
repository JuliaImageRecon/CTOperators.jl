# Operators

The generic interface of all CT operators: application with `*` and `mul!`, adjoints, sizes,
row and column sums, subsets of views, moving to a device with `adapt`, and the launch
configuration of the kernels.

## Generic interface

```@docs
AbstractCTOperator
```

```@autodocs
Modules = [CTOperators]
Pages = ["operators/operators.jl"]
Private = false
```

## Image and ray masks

The image mask comes from the `ImageGeom`, the ray mask from the keyword `raymask` of a projector
(see [Use Masks](@ref) and [Masks](@ref)). These functions are not exported.

```@docs
CTOperators.image_mask
CTOperators.ray_mask
CTOperators.handles_raymask
```

## Launch configuration

```@autodocs
Modules = [CTOperators]
Pages = ["operators/launch.jl"]
Private = false
```

## Extensions

* With LinearOperators.jl loaded, `normalOperator(op)` of LinearOperatorCollection.jl gives the
  normal operator `AᵀA` of a CT operator, e.g. for `AHA` of the solvers of
  RegularizedLeastSquares.jl. With LinearOperatorCollection.jl loaded, the normal operator of the
  projectors of this package runs in one pass over the rays, also with weights (see
  [Normal Operator](@ref)).
* With ChainRulesCore.jl loaded, the operators are differentiable (see
  [Differentiate through a Projector](@ref)).
* With Sinograms.jl loaded, its geometries are converted by `projection_geometry` (see
  [Use Geometries of Sinograms.jl](@ref)).
