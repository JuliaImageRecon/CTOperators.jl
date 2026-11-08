# The normal operator Aᴴ W A of the projectors of this package in one pass over the rays, for the
# normal operators of LinearOperatorCollection.jl (package extension).

"""
    fused_normal!(z, op, x, weights)

`z = Aᴴ W A x` for the projector `op`, with `W` the diagonal matrix of `weights` (one per data
element, or `nothing` for `W = I`), in one pass over the rays: every ray is traced and summed,
multiplied by its weight, and traced again to add its value to the voxels, without an array for the
projection data. `normalOperator(op)` of LinearOperatorCollection.jl uses it.
"""
function fused_normal!(z, op::Union{SiddonProjector,JosephProjector}, x, weights)
  check_length(x, imagesize(op), "input")
  check_length(z, imagesize(op), "output")
  isnothing(weights) || check_length(weights, datasize(op), "weights")
  check_accumulation(z)
  masks = projector_masks(op)
  zi, xi = reshape(z, imagesize(op)), reshape(x, imagesize(op))
  fill!(zi, zero(eltype(zi)))
  launch_rays!(normal_kernel!, op, vec(zi), vec(isnothing(masks.image) ? xi : xi .* masks.image),
               isnothing(weights) ? nothing : vec(weights), ray_segment(op))
  isnothing(masks.image) || (zi .*= masks.image)
  z
end
ray_segment(::SiddonProjector) = siddon_segment
ray_segment(::JosephProjector) = joseph_segment

# The normal operator splits into rays, Aᴴ W A = sum_i a_i w_i a_iᴴ, so one thread per ray needs no
# synchronization with the others; the second trace adds into the image with atomics.
@kernel inbounds = true function normal_kernel!(z, @Const(x), grid, geometry, order, raymask, weights, segment)
  i = ray_index(order, @index(Global, Linear))
  if i > 0 && !is_masked(raymask, i)
    v = trace_element(segment, Gather(x), grid, geometry, i)
    v = v * ray_weight(weights, v, i)
    if !iszero(v)
      trace_element(segment, Scatter(z, v), grid, geometry, i)
    end
  end
end

@inline ray_weight(::Nothing, v, i) = one(v)
@inline ray_weight(weights, v, i) = @inbounds oftype(v, weights[i])
