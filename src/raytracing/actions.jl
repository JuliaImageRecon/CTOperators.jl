# What a ray does with the voxels it visits, shared by the ray-driven methods: sum the weighted
# image values (forward projection) or add its weighted value to them (back projection).
struct Gather{A}
  image::A
end

struct Scatter{A,T}
  image::A
  value::T
end

@inline visit(g::Gather, acc, lin, w) = @inbounds muladd(w, g.image[lin], acc)

@inline function visit(s::Scatter, acc, lin, w)
  @inbounds Atomix.@atomic :monotonic s.image[lin] += w * s.value
  acc
end
