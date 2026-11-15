# The nonlinear model of a measurement that averages intensities over its rays (the log is taken
# after the average): p(x) = -log sum_k w_k exp(-(A_k x)), with the projections A_k x of the samples
# from a SampleStack. Evaluated with a stable log-sum-exp per data element: L_k = log w_k - (A_k x),
# M = max_k L_k, p = -(M + log sum_k exp(L_k - M)); the weights of the Jacobian are
# pi_k = exp(L_k - M) / sum_j exp(L_j - M).

"""
    IntensityModel(op::AbstractProjector; source=nothing, detector=nothing)

The model of a measurement that averages the intensities `exp(-p_k)` over the rays of its samples
before the log: `p(x) = -log(sum_k w_k exp(-(A_k x)))` per data element, with the projections
`A_k x` and weights `w_k` of [`stacked`](@ref)`(op; source, detector)`. The mean of the line
integrals ([`sampled`](@ref)) is never smaller (Jensen's inequality); at `x = 0` both models have the
same derivative.

`m(x)` evaluates the model (data of the size of `op`). [`CTOperators.jacobian`](@ref) is its
derivative at `x`, [`CTOperators.gradient!`](@ref) the gradient of `|p(x) - y|^2 / 2`, and
[`IntensityGradient`](@ref) that gradient as an operator for first-order solvers.

The same model as the simulator of De Man et al. (1999) for the focal spot and the detector cells
(the logarithm of the summed photon flux).

The samples run sequentially, so a detector sampling needs points that do not vary per cell
(`RegularPoints`, `DiskPoints`, `Points`; see [`CTOperators.varies_per_cell`](@ref)). Holds buffers,
so it must not be evaluated by several tasks at once.

References:
* B. De Man, J. Nuyts, P. Dupont, G. Marchal, P. Suetens, "Metal streak artifacts in X-ray computed
  tomography: a simulation study", IEEE Transactions on Nuclear Science 46(3), 691–696, 1999.
"""
struct IntensityModel{T,B<:SampleStack{T},L,Q}
  stack::B
  logweights::L     # (1, 1, nviews, K): log w_k per view, on the device
  projections::Q    # buffer of size datasize(stack)
end

function IntensityModel(op::AbstractProjector; source=nothing, detector=nothing)
  B = stacked(op; source, detector)
  T = eltype(B)
  K = length(B.ops)
  host = zeros(T, 1, 1, nviews(geometry(op)), K)
  for (k, w) in enumerate(B.weights)
    host[1, 1, :, k] .= log.(Array(w))
  end
  logweights = copyto!(KA.allocate(KA.get_backend(B), T, size(host)), host)
  IntensityModel(B, logweights, KA.allocate(KA.get_backend(B), T, datasize(B)))
end

Base.eltype(::IntensityModel{T}) where {T} = T
datasize(m::IntensityModel) = datasize(first(m.stack.ops))
imagesize(m::IntensityModel) = imagesize(m.stack)
KA.get_backend(m::IntensityModel) = KA.get_backend(m.stack)

# p(x) and the weights pi_k of the Jacobian, from the projections of the samples
function intensity_terms(m::IntensityModel, x)
  mul!(m.projections, m.stack, x)
  L = m.logweights .- m.projections
  M = maximum(L; dims=4)
  E = exp.(L .- M)
  S = sum(E; dims=4)
  dropdims(-(M .+ log.(S)); dims=4), E ./ S
end

function (m::IntensityModel)(x::AbstractArray)
  p, _ = intensity_terms(m, x)
  x isa AbstractVector ? vec(p) : p
end

"""
    jacobian(m::IntensityModel, x)

The derivative of the model `m` at the image `x`: `J = sum_k diag(pi_k) A_k`, a [`SampleSum`](@ref)
with the weights `pi_k = w_k exp(-(A_k x)) / sum_j w_j exp(-(A_j x))` per data element (they sum to 1).
At `x = 0` it is the linear model [`sampled`](@ref).
"""
function jacobian(m::IntensityModel, x)
  _, weights = intensity_terms(m, x)
  B = m.stack
  K = size(weights, 4)
  SampleSum(B.ops, [weights[:, :, :, k] for k in 1:K], similar(B.data), similar(B.image))
end

"""
    gradient!(g, m::IntensityModel, x, y)

The gradient of `|m(x) - y|^2 / 2` at `x`, `J(x)ᵀ (m(x) - y)`, written into `g` (of the size of the
image, or a vector). One projection and one back projection of every sample.
"""
function gradient!(g, m::IntensityModel, x, y)
  p, weights = intensity_terms(m, x)
  r = p .- reshape(y, size(p))
  mul!(g, adjoint(m.stack), weights .* r)
end

"""
    IntensityGradient(m::IntensityModel, y)

The gradient of `|m(x) - y|^2 / 2` as an operator of the image: `mul!(g, G, x)` is
[`CTOperators.gradient!`](@ref)`(g, m, x, y)`, of size `n × n` (`n` the number of voxels). For the
solvers of RegularizedLeastSquares.jl that take only a normal operator, e.g. `FISTA(; AHA=G, rho)`
with the right-hand side `b = 0`; the iteration `x -= rho * (G x - b)` is then a gradient step. In
`FISTA` of RegularizedLeastSquares.jl:

* `rho` has to be given: its default comes from power iterations, which assume a linear operator.
* The convergence test divides the residual by the norm of `b`; with `b = 0` it never stops early,
  and the solver runs the given number of iterations.

The objective is in general not convex: the result depends on the start (`x0`).
"""
struct IntensityGradient{T,M<:IntensityModel{T},Y}
  model::M
  y::Y
end

Base.eltype(::IntensityGradient{T}) where {T} = T
Base.size(G::IntensityGradient) = (n = prod(imagesize(G.model)); (n, n))
Base.size(G::IntensityGradient, d::Integer) = d <= 2 ? size(G)[d] : 1
LinearAlgebra.mul!(g::AbstractArray, G::IntensityGradient, x::AbstractArray) = gradient!(g, G.model, x, G.y)

function Base.:*(G::IntensityGradient, x::AbstractArray)
  g = similar(x, eltype(G))
  mul!(g, G, x)
end

function Base.show(io::IO, ::MIME"text/plain", m::IntensityModel)
  print(io, "IntensityModel of ", length(m.stack.ops), " samples of ", nameof(typeof(first(m.stack.ops))), ", data ",
        datasize(m), ", image ", imagesize(m))
end
