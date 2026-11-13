# FDK (Feldkamp, Davis & Kress 1984) for circular cone beams, and filtered back projection for
# parallel beams: weight the data, filter the detector rows with the ramp, back project with the
# voxel-driven projector. In the quantities of this package (fan-beam forms of Kak & Slaney 1988,
# chapter 3, carried over to the cone as Feldkamp et al. do):
#
# 1. Weight: every cell by the cosine of the angle between its ray and the central ray of the view
#    (the ray from the source to the rotation axis, perpendicular to it). For a flat detector this
#    is Feldkamp's d / sqrt(d^2 + Y^2 + Z^2), for a detector on an arc cos(gamma) cos(kappa) (fan
#    and cone angle); for parallel beams 1. Times the weight of the redundancy weighting
#    (redundancy.jl; 1 for a full scan).
# 2. Filter: every detector row with the ramp, in the length unit of the detector (filter.jl).
# 3. Back project: the voxel-driven back projection with the weight c_k (R/r)^2 of voxel and view,
#    R/r the magnification of the voxel, c_k = w_k dso / dsd with w_k the weight of the view from the
#    redundancy weighting (a full scan: half the angle step of the view; parallel beams over half the
#    circle: the angle step; dso / dsd = 1 for parallel beams).
#
# For a flat detector (R/r)^2 dso / dsd = dso dsd / L^2 with L the distance of the voxel from the
# source along the central ray: Feldkamp's weight d^2 / (d + r.x)^2 with the filter in the
# coordinates of the detector instead of a detector through the axis. For a detector on an arc,
# R/r = dsd / L with L the distance from the source within the plane of the orbit.

# The scale of the back projection of FDK: weights[k] * magnification^2 in view k.
struct FDKScale{V<:AbstractVector} <: AbstractVoxelScale
  weights::V
end

Adapt.@adapt_structure FDKScale

subset(s::FDKScale, views) = FDKScale(s.weights[views])

@inline voxel_scale(s::FDKScale, geometry, k, grid::VoxelGrid{T}, hit) where {T} =
  T(s.weights[k]) * hit.magnification * hit.magnification

"""
    FDK(image::ImageGeom, geometry; redundancy=FullScan(), window=:hann, cutoff=1, eltype=Float32, storage=Array,
        launch=nothing)

Reconstruction with the method of Feldkamp, Davis & Kress (1984) for a circular cone-beam scan
([`ConeBeam`](@ref) or [`ArcConeBeam`](@ref) with the source on a circle about the rotation axis),
and filtered back projection for a [`ParallelBeam`](@ref): the data are weighted, the detector rows
filtered with the ramp filter and the result back projected with the voxel-driven method (see
[`VoxelDrivenProjector`](@ref); [Filtered Back Projection](@ref) explains the steps). The plan holds the filter, the FFT plans and the buffers;
`plan * y` and `mul!(x, plan, y)` reconstruct the image from the line integrals `y` (of the data
size of `geometry`). [`CTOperators.fdk`](@ref)`(image, geometry, y)` does it in one call.

The geometry must be a circular scan: the rotation axis runs along the detector columns (`v`) in all
views, the rows (`u`) are perpendicular to it, the sources lie on a circle about the axis in a plane
perpendicular to it, a flat detector is perpendicular to the central ray (the ray from the source to
the axis), the cells have the same width in all views, and the views are in order along the
circle; geometries that break these give an `ArgumentError` (helical scans, for example). How much of
the circle the views must cover depends on the redundancy weighting: with [`FullScan`](@ref), the
default, the whole circle (parallel beams: half or the whole circle), with [`ParkerWeights`](@ref)
half the circle plus the fan angle; the angle steps may differ between views. Whether the detector
covers the object in every view is not checked: data of an object that reaches beyond the rays of
the detector give a wrong image. Sinograms.jl geometries are converted with
[`projection_geometry`](@ref).

Keywords:

* `redundancy`: how lines measured more than once are weighted, an [`AbstractRedundancy`](@ref):
  [`FullScan`](@ref) (the default), [`ParkerWeights`](@ref) (short scans), [`ArrayRedundancy`](@ref)
  (weights as arrays) or a weighting of your own.
* `window`: the window of the ramp filter, `:hann` (the default), `:ramp` (none), `:hamming`,
  `:cosine`, `:shepplogan`, or a function `w(f)` of the frequency `f` as a fraction of the cutoff.
  A window lowers the ramp at high frequencies: the image is smoother, with less resolution; on
  data free of noise the plain ramp is closest to the object (see [Reconstruct with FDK](@ref)).
* `cutoff`: the highest frequency passed, a fraction of the Nyquist frequency of the cells (`0 <
  cutoff ≤ 1`).
* `eltype`, `storage`, `launch`: as for the projectors (the element type of the computation, the
  array type that selects the device, the launch of the back projection).

The image mask of `image` selects the voxels that are reconstructed. An image with one voxel along
`z` (2D, or 3D with `nz = 1`) needs a detector with one row. A plan owns its buffers: it must not
run on several tasks at the same time. `adapt(storage, plan)` (Adapt.jl) moves a plan to another
device, with new buffers and FFT plans.

# Example

```julia
image = ImageGeom(; dims=(128, 128, 64), deltas=(1, 1, 1))
geometry = circular_cone_beam(; angles=range(0, 2pi, 361)[1:360], dso=500, dsd=1000,
                              nu=256, nv=128, du=1, dv=1)
plan = FDK(image, geometry)
x = plan * y               # y: line integrals of size (256, 128, 360)
```

References:
* L. A. Feldkamp, L. C. Davis, J. W. Kress, "Practical cone-beam algorithm", J. Opt. Soc. Am. A
  1(6), 612–619, 1984.
* A. C. Kak, M. Slaney, "Principles of Computerized Tomographic Imaging", IEEE Press, 1988,
  chapter 3.
"""
struct FDK{T,P<:VoxelDrivenProjector{T},C,W,H,A,B,S,F,R}
  backprojector::P
  central::C        # 3 x nviews: the central ray of each view (nothing for parallel beams)
  redundancy::W     # the kernel side of the redundancy weighting, on the device
  response::H       # the frequency response of the filter, length div(npad, 2) + 1
  padded::A         # (npad, nv, nviews): the weighted rows, padded with zeros, and the filtered rows
  filtered::B       # (nu, nv, nviews): the filtered rows, input of the back projection
  spectrum::S       # (div(npad, 2) + 1, nv, nviews), complex
  rfft::F
  brfft::R
end

FDK(image::ImageGeom, geometry; kwargs...) = FDK(image, projection_geometry(geometry); kwargs...)

FDK(image::ImageGeom, geometry::AbstractProjectionGeometry; kwargs...) =
  throw(ArgumentError("FDK needs a ConeBeam, ArcConeBeam or ParallelBeam, not a $(nameof(typeof(geometry)))"))

function FDK(image::ImageGeom, geometry::AbstractDetectorGeometry; redundancy::AbstractRedundancy=FullScan(),
             window=:hann, cutoff::Real=1,
             eltype::Type{T}=Float32, storage=Array, launch::Union{Nothing,LaunchConfig}=nothing) where {T<:AbstractFloat}
  scan = circular_scan(geometry)
  nu, nv = detectorsize(geometry)
  one_layer(image) && nv != 1 &&
    throw(ArgumentError("FDK of an image with one voxel along z (2D, or nz = 1) needs a detector with one row, " *
                        "got $nv rows"))
  # the filter: the cell spacing along the rows, and the radius of an arc detector
  h = adapt(Array{Float64}, geometry)
  du = norm(h.u[:, 1])
  npad = padded_length(nu)
  tap = geometry isa ArcConeBeam ? (n -> ramp_tap(n, du, sum(scan.dsd) / length(scan.dsd))) : (n -> ramp_tap(n, du))
  response = filter_response(tap, nu, du, npad, window, float(cutoff))
  # the geometry of the plan: element type T, on the device
  g = device_geometry(geometry, T, storage)
  weight, views = redundancy_weights(redundancy, scan, g)
  length(views) == nviews(geometry) ||
    throw(DimensionMismatch("redundancy_weights gave $(length(views)) view weights for $(nviews(geometry)) views"))
  weights = views .* scan.dso ./ scan.dsd
  # on the device
  backend = KA.get_backend(storage_array(g))
  device(a) = copyto!(KA.allocate(backend, T, size(a)), T.(a))
  backprojector = voxel_driven_projector(image, g, launch, nothing, FDKScale(device(weights)))
  central = geometry isa ParallelBeam ? nothing : device(scan.central)
  fdk_plan(backprojector, central, adapt(storage, weight), device(response), npad)
end

# the plan with new buffers and FFT plans on the device of the back projector
function fdk_plan(backprojector, central, redundancy, response, npad)
  T = eltype(backprojector)
  backend = KA.get_backend(backprojector)
  nu, nv = detectorsize(geometry(backprojector))
  n = nviews(geometry(backprojector))
  padded = KA.zeros(backend, T, npad, nv, n)
  spectrum = KA.allocate(backend, Complex{T}, div(npad, 2) + 1, nv, n)
  FDK(backprojector, central, redundancy, response, padded, KA.allocate(backend, T, nu, nv, n), spectrum,
      FFTL.plan_rfft(padded, 1), FFTL.plan_brfft(spectrum, npad, 1))
end

Adapt.adapt_structure(to, plan::FDK) =
  fdk_plan(adapt(to, plan.backprojector), adapt(to, plan.central), adapt(to, plan.redundancy), adapt(to, plan.response),
           size(plan.padded, 1))

"""
    CTOperators.fdk(image::ImageGeom, geometry, y; kwargs...)

The [`FDK`](@ref) reconstruction of the line integrals `y` in one call, `FDK(image, geometry;
kwargs...) * y`. Public but not exported: Sinograms.jl exports a function `fdk` of its own; with
Sinograms.jl loaded, its `fdk` also takes an `FDK` plan, `fdk(plan, y)`, or the image and a
geometry of this package, `fdk(image, geometry, y)`.
"""
fdk(image::ImageGeom, geometry, y; kwargs...) = FDK(image, geometry; kwargs...) * y

Base.eltype(::FDK{T}) where {T} = T
geometry(plan::FDK) = geometry(plan.backprojector)
image_geometry(plan::FDK) = image_geometry(plan.backprojector)
imagesize(plan::FDK) = imagesize(plan.backprojector)
datasize(plan::FDK) = datasize(plan.backprojector)
KA.get_backend(plan::FDK) = KA.get_backend(plan.backprojector)

function Base.:*(plan::FDK, y::AbstractArray)
  x = KA.allocate(KA.get_backend(plan), eltype(plan), imagesize(plan))
  mul!(x, plan, y)
  y isa AbstractVector ? vec(x) : x
end

function LinearAlgebra.mul!(x::AbstractArray, plan::FDK, y::AbstractArray)
  check_length(y, datasize(plan), "input")
  check_length(x, imagesize(plan), "output")
  g = geometry(plan)
  nu, nv = detectorsize(g)
  backend = KA.get_backend(plan)
  # 1. weighted rows into the padded buffer (the padding stays zero)
  fill!(plan.padded, zero(eltype(plan)))
  kernel = fdk_weight_kernel!(backend)
  kernel(plan.padded, vec(y), g, plan.central, plan.redundancy; ndrange=length(y))
  # 2. the ramp filter along the rows
  mul!(plan.spectrum, plan.rfft, plan.padded)
  plan.spectrum .*= plan.response
  mul!(plan.padded, plan.brfft, plan.spectrum)
  plan.filtered .= view(plan.padded, 1:nu, :, :)
  # 3. back projection
  mul!(x, adjoint(plan.backprojector), plan.filtered)
end

@kernel inbounds = true function fdk_weight_kernel!(padded, @Const(y), geometry, central, redundancy)
  i = @index(Global, Linear)
  nu, nv = detectorsize(geometry)
  iu, iv, k = pixel_index(i, nu, nv)
  T = eltype(padded)
  w = cosine_weight(geometry, central, iu, iv, k) * redundancy_weight(redundancy, geometry, iu, iv, k)
  padded[iu, iv, k] = T(w) * T(y[i])
end

# the cosine of the angle between the ray of cell (iu, iv) and the central ray of view k
@inline cosine_weight(::ParallelBeam{T}, central, iu, iv, k) where {T} = one(T)

@inline function cosine_weight(g::AbstractDetectorGeometry{T}, central, iu, iv, k) where {T}
  s, b = cell_ray(g, T(iu), T(iv), k)
  d = b - s
  dot(d, column(central, k)) / norm(d)
end
