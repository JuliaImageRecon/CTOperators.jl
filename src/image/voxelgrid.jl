"""
    VoxelGrid{T}(ig::ImageGeom)

The voxel grid of an `ImageGeom` in the form the kernels use: three dimensions (a 2D `ImageGeom`
becomes a volume with `nz = 1` and voxel size 1 along `z`, centered at `z = 0`), element type `T`,
no units. `first` is the world coordinate of the center of voxel `(1, 1, 1)`; voxel
`(i, j, k)` has its center at `first + ((i, j, k) .- 1) .* deltas`. Negative `deltas` (flipped
axes, allowed by ImageGeoms) are kept.

`lo` and `hi` are the corners of the bounding box of the mask of the `ImageGeom` (the smallest box
containing its voxels), in voxel coordinates (see [`to_index`](@ref CTOperators.to_index)):
`[1/2, n + 1/2]` per dimension without a mask. Projectors clip their rays to this box (Joseph widens
it by half a voxel across the ray, for the interpolation), so that they do not traverse voxels that
the mask sets to zero.

Units of the `ImageGeom` are stripped; they must be the units of the projection geometry.
"""
struct VoxelGrid{T}
  dims::NTuple{3,Int}
  deltas::SVector{3,T}
  first::SVector{3,T}
  lo::SVector{3,T}
  hi::SVector{3,T}
end

# the whole volume as the box
function VoxelGrid{T}(dims::NTuple{3,Int}, deltas::SVector{3,T}, first::SVector{3,T}) where {T}
  half = T(1 // 2)
  VoxelGrid{T}(dims, deltas, first, SVector(half, half, half), SVector(map(n -> T(n) + half, dims)))
end

function VoxelGrid{T}(ig::ImageGeom{D}) where {T<:AbstractFloat,D}
  D in (2, 3) || throw(ArgumentError("only 2D and 3D image geometries are supported, got $(D)D"))
  dims = ntuple(d -> d <= D ? ig.dims[d] : 1, 3)
  deltas = SVector(ntuple(d -> d <= D ? T(strip_unit(ig.deltas[d])) : one(T), 3))
  first = SVector(ntuple(d -> d <= D ? T(strip_unit(axis(ig, d)[1])) : zero(T), 3))
  lo, hi = mask_box(ig.mask, T)
  VoxelGrid{T}(dims, deltas, first, lo, hi)
end

# The box around the true voxels of a mask, in voxel coordinates (3D; a 2D mask covers z in
# [1/2, 3/2]). An empty mask gives lo > hi: rays miss it.
function mask_box(mask::AbstractArray{Bool,D}, ::Type{T}) where {D,T}
  half = T(1 // 2)
  lo, hi = ones(T, 3) .- half, ones(T, 3) .+ half
  for d in 1:D
    hits = findall(vec(any(mask; dims=Tuple(e for e in 1:D if e != d))))
    if isempty(hits)
      return SVector{3,T}(1, 1, 1), SVector{3,T}(0, 0, 0)
    end
    lo[d], hi[d] = first(hits) - half, last(hits) + half
  end
  SVector{3,T}(lo), SVector{3,T}(hi)
end

# a length without its unit (Unitful quantities), plain numbers unchanged
strip_unit(x) = x / oneunit(x)

Base.size(g::VoxelGrid) = g.dims
Base.eltype(::VoxelGrid{T}) where {T} = T

"""
    bounds(g::VoxelGrid) -> (lo, hi)

Corners of the box covered by the voxels (outer faces, not centers), `lo .<= hi`.
"""
function bounds(g::VoxelGrid)
  a = g.first .- g.deltas ./ 2
  b = g.first .+ (SVector(g.dims) .- one(eltype(g)) ./ 2) .* g.deltas
  min.(a, b), max.(a, b)
end

"""
    to_index(g::VoxelGrid, p)

Continuous voxel coordinates of the world point `p`: voxel `(i, j, k)` covers
`[i − 1/2, i + 1/2) × …`, its center has the integer coordinates `(i, j, k)`. In these coordinates
the voxels are unit cubes with positive orientation, whatever the sign of `deltas`.
"""
@inline to_index(g::VoxelGrid, p) = (p .- g.first) ./ g.deltas .+ one(eltype(g))

# the voxel containing the voxel coordinate q, clamped to 1:n
@inline voxel_index(q, n) = clamp(unsafe_trunc(Int, floor(q + oftype(q, 0.5))), 1, n)
