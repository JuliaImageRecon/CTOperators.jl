"""
    Gather(image)

What a ray does in the forward projection: it sums the weighted values of the voxels it visits.
`image` is indexed linearly. Its values are converted to the element type of the weights when read,
so an image stored in lower precision (e.g. `Float16`) is summed in the precision of the projector.
"""
struct Gather{A}
  image::A
end

"""
    Scatter(image, value)

What a ray does in the back projection: it adds its weighted `value` to the voxels it visits.
`image` is indexed linearly.
"""
struct Scatter{A,T}
  image::A
  value::T
end

"""
    visit(action, acc, lin, w) -> acc

A ray meets the voxel with linear index `lin`, with weight `w`: [`CTOperators.Gather`](@ref) adds to
the accumulated value `acc`, [`CTOperators.Scatter`](@ref) writes to the image. Called in kernels.
"""
@inline visit(g::Gather, acc, lin, w) = @inbounds muladd(w, oftype(w, g.image[lin]), acc)

@inline function visit(s::Scatter, acc, lin, w)
  @inbounds Atomix.@atomic :monotonic s.image[lin] += w * s.value
  acc
end
