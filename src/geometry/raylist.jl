"""
    RayList(starts, ends; datasize=(size(starts, 2),))

Arbitrary rays given by their start and end points (columns of the `3 × nrays` matrices `starts`
and `ends`), as in parallelproj. Ray `i` is the segment from `starts[:, i]` to `ends[:, i]`.
`datasize` gives the projection data a shape (column-major order of the rays), e.g. a sinogram.

    RayList(g::AbstractProjectionGeometry)

The rays of a geometry with [`RaySegment`](@ref) rays, computed on the device of `g`.
"""
struct RayList{T,M<:AbstractMatrix{T},N} <: AbstractProjectionGeometry{T}
  starts::M
  ends::M
  datasize::NTuple{N,Int}
  function RayList{T,M,N}(starts::M, ends::M, datasize::NTuple{N,Int}) where {T,M<:AbstractMatrix{T},N}
    size(starts, 1) == 3 || throw(DimensionMismatch("starts must be 3 × nrays, got $(size(starts))"))
    size(starts) == size(ends) || throw(DimensionMismatch("starts $(size(starts)) and ends $(size(ends)) differ"))
    prod(datasize) == size(starts, 2) ||
      throw(DimensionMismatch("datasize $datasize does not match $(size(starts, 2)) rays"))
    new{T,M,N}(starts, ends, datasize)
  end
end

function RayList(starts::AbstractMatrix, ends::AbstractMatrix; datasize::Dims=(size(starts, 2),))
  T = promote_type(eltype(starts), eltype(ends))
  s, e = convert(AbstractMatrix{T}, starts), convert(AbstractMatrix{T}, ends)
  s, e = promote(s, e)
  RayList{T,typeof(s),length(datasize)}(s, e, datasize)
end

datasize(g::RayList) = g.datasize
storage_array(g::RayList) = g.starts
raytype(::RayList) = RaySegment()
@inline ray(g::RayList, i) = column(g.starts, i), column(g.ends, i)

function Base.show(io::IO, ::MIME"text/plain", g::RayList{T}) where {T}
  print(io, "RayList{", T, "}: ", nrays(g), " rays, data size ", datasize(g), ", stored in ",
        nameof(typeof(storage_array(g))))
end

Adapt.adapt_structure(to, g::RayList) = RayList(adapt(to, g.starts), adapt(to, g.ends); datasize=g.datasize)

RayList(g::AbstractProjectionGeometry) = RayList(raytype(g), g)
RayList(::RayLine, g) = throw(ArgumentError("$(nameof(typeof(g))) has lines, not segments; a RayList needs end points"))
function RayList(::RaySegment, g)
  starts, ends = geometry_array(g, 3, nrays(g)), geometry_array(g, 3, nrays(g))
  backend = KA.get_backend(starts)
  rays_kernel!(backend)(starts, ends, g; ndrange=nrays(g))
  RayList(starts, ends; datasize=datasize(g))
end

@kernel function rays_kernel!(starts, ends, g)
  i = @index(Global)
  a, b = ray(g, i)
  @inbounds for d in 1:3
    starts[d, i] = a[d]
    ends[d, i] = b[d]
  end
end
