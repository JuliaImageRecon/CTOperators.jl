# # Choose the Precision
#
# Three element types decide the precision of a projection, and each is chosen through what you pass
# to the operator:
#
# 1. **The operator** (keyword `eltype`, default `Float32`): the ray geometry and the arithmetic.
# 2. **The input array**: how the image (forward projection) or the data (back projection) are
#    stored. The projector converts every value to its own element type when it reads it.
# 3. **The output array**: what the result is stored in. In the forward projection every data value
#    is rounded once. In the back projection every voxel collects the contributions of many data
#    elements, and they are added up in this type.
#
# Any floating-point type is accepted in each place. Lower precision halves the memory of the arrays
# it is used for, but costs accuracy; this page shows how to set up the usual combinations and what
# the others cost on a small example.

using CTOperators, BFloat16s, ImagePhantoms, LinearAlgebra
using ImageGeoms: axis

n = 32
image = ImageGeom(; dims=(n, n, n), deltas=(1.0, 1.0, 1.0))
geometry = circular_cone_beam(; angles=range(0, 2pi, 91)[1:90], dso=2n, dsd=4n, nu=48, nv=n, du=2.0)
x = 0.02 .* Float32.(phantom(axis(image, 1), axis(image, 2), axis(image, 3),
                             [ellipsoid((0, 0, 0), (0.4n, 0.35n, 0.3n), (0, 0, 0), 1.0)]));

# ## Float32 arithmetic with Float16 storage
#
# The operator computes in `Float32`, images and data are stored in `Float16`, and the back
# projection collects into a `Float32` image. The forward projection reads the `Float16` image and
# writes `Float16` data when the output array is `Float16`:

op = JosephProjector(image, geometry)        # Float32 geometry and arithmetic
x16 = Float16.(x)                            # an image stored in Float16
y16 = similar(x16, datasize(op))             # data stored in Float16
mul!(y16, op, x16)                           # every data value rounded once to Float16
b = adjoint(op) * y16;                       # a Float32 image: the sums in Float32

# On a GPU the same holds with the array types of the device, e.g. `CuArray{Float16}` for the image
# and the data and `storage=CuArray` for the operator (see [Run on a GPU](@ref)).
#
# ## BFloat16
#
# `BFloat16` (from BFloat16s.jl) is used the same way. It has the range of `Float32` but 8 bits of
# mantissa instead of the 11 of `Float16`, so each value is rounded more coarsely. `Float16` cannot
# represent values above 65504: data that are not log-transformed, or images in Hounsfield units
# along long rays (1000 HU over 500 mm give 5 · 10⁵), exceed it, and there `BFloat16` keeps the
# range:

xb = BFloat16.(x)
yb = similar(xb, datasize(op))
mul!(yb, op, xb);

# ## What each choice costs
#
# The relative error ``\|y - y_\text{ref}\| / \|y_\text{ref}\|``, in percent, of the forward
# projection of the image above and of the back projection of its projection, against `Float64`, for
# Joseph's method:

reldiff(a, b) = round(100 * norm(Float64.(a) .- b) / norm(b); sigdigits=2)
ref = JosephProjector(image, geometry; eltype=Float64)
y = ref * Float64.(x)
b64 = adjoint(ref) * y
rows = Any[]
for S in (Float16, BFloat16)
  ys = similar(y, S)
  mul!(ys, op, S.(x))
  push!(rows, (setting="Float32 operator, $S image and data", forward=reldiff(ys, y),
               back=reldiff(adjoint(op) * S.(y), b64)))
end
for S in (Float16, BFloat16)
  bs = zeros(S, image.dims)
  mul!(bs, adjoint(op), Float32.(y))
  push!(rows, (setting="Float32 operator, back projection into a $S image", forward=missing,
               back=reldiff(bs, b64)))
end
for T in (Float16, BFloat16)
  opT = JosephProjector(image, geometry; eltype=T)
  push!(rows, (setting="$T operator, image and data", forward=reldiff(opT * T.(x), y),
               back=reldiff(adjoint(opT) * T.(y), b64)))
end
rows

# * Storing image and data in a lower precision rounds every value once; the projection inherits
#   this rounding, which is smaller for `Float16` than for `BFloat16`.
# * A back projection into a low-precision image adds many contributions per voxel in that
#   precision, so its rounding errors can add up with the number of data elements that reach a
#   voxel.
# * An operator in `Float16` or `BFloat16` also holds the ray geometry in that precision. Positions
#   are then rounded to a spacing of 2⁻¹⁰ (`Float16`) or 2⁻⁷ (`BFloat16`) times their magnitude, so
#   rays far from the origin can land in neighboring voxels.
#
# Timings with `Float16` and `BFloat16` images on a GPU are in the
# [benchmarks](https://github.com/JuliaImageRecon/CTOperators.jl/blob/main/benchmark/README.md#float16-and-bfloat16-images-gpu).
