# # Reconstruct with FDK
#
# FDK (Feldkamp, Davis & Kress 1984) reconstructs a circular cone-beam scan in one pass, without
# iterations: weight the data, filter each detector row with a ramp filter, back project. For
# parallel beams the same steps are filtered back projection (FBP). This guide reconstructs a 3D
# Shepp-Logan phantom with [`FDK`](@ref), compares the windows of the filter, and uses the result as
# the start image of an iterative method.

using CTOperators, ImagePhantoms, RegularizedLeastSquares, LinearOperatorCollection, CairoMakie, LinearAlgebra
using ImageGeoms: axis
using ImagePhantoms: ellipsoid_parameters

# ## Phantom and data
#
# The 3D Shepp-Logan phantom in a volume of 64 × 64 × 48 voxels of 1 mm, and a circular scan with a
# flat detector: 180 views over one turn, the source 200 mm from the axis, the detector 400 mm from
# the source, cells of 2 mm (1 mm at the axis). The data are the forward projection of the phantom
# with Joseph's method.

image = ImageGeom(; dims=(64, 64, 48), deltas=(1.0, 1.0, 1.0))
phantom_ellipsoids = ellipsoid(ellipsoid_parameters(; fovs=(64, 64, 48)))
x_true = Float32.(phantom(axis(image, 1), axis(image, 2), axis(image, 3), phantom_ellipsoids, 2))
geometry = circular_cone_beam(; angles=range(0, 2pi, 181)[1:180], dso=200, dsd=400, nu=112, nv=64, du=2.0)
op = JosephProjector(image, geometry)
y = op * x_true;

# ## Reconstruction
#
# `FDK(image, geometry)` prepares the filter, the FFT plans and the buffers; applied to the data it
# returns the image. The plan can be reused for further data of the same scan.
# `CTOperators.fdk(image, geometry, y)` (not exported) does both in one call.

plan = FDK(image, geometry)
x_fdk = plan * y;

# The axial slice through the middle of the volume and the sagittal slice through the axis (`z`
# up), with a narrow gray scale for the small contrasts inside the skull:

slices(x) = (x[:, :, 25], x[33, :, :]) #hide
fig = Figure(size=(800, 560)) #hide
for (i, (name, x)) in enumerate((("phantom", x_true), ("FDK, Hann window", x_fdk))) #hide
  for (j, s) in enumerate(slices(x)) #hide
    ax = Axis(fig[j, i]; aspect=DataAspect(), title=j == 1 ? name : "", xticksvisible=false, yticksvisible=false, #hide
              xticklabelsvisible=false, yticklabelsvisible=false) #hide
    heatmap!(ax, s; colormap=:grays, colorrange=(0.95, 1.05)) #hide
  end #hide
end #hide
save("fdk_slices.png", fig) #hide
nothing #hide
# ![Axial and sagittal slices of the phantom and the FDK reconstruction](fdk_slices.png)

# ## Windows
#
# The ramp filter amplifies high frequencies, and with them noise and the discretization error of
# the data. A window lowers the ramp towards high frequencies: the keyword `window` takes `:ramp`
# (no window), `:hann` (the default), `:hamming`, `:cosine`, `:shepplogan` or a function of the
# frequency, and `cutoff` the highest frequency passed, as a fraction of the Nyquist frequency of the
# cells. A profile through the axial slice, along `x`:

x_ramp = CTOperators.fdk(image, geometry, y; window=:ramp)
x_half = CTOperators.fdk(image, geometry, y; window=:hann, cutoff=0.5)
fig = Figure(size=(800, 320)) #hide
ax = Axis(fig[1, 1]; xlabel="x (mm)", ylabel="value") #hide
for (name, x) in (("phantom", x_true), ("ramp", x_ramp), ("Hann", x_fdk), ("Hann, cutoff 0.5", x_half)) #hide
  lines!(ax, collect(axis(image, 1)), x[:, 33, 25]; label=name) #hide
end #hide
axislegend(ax; position=:cb) #hide
save("fdk_windows.png", fig) #hide
nothing #hide
# ![A profile through the reconstructions with different windows](fdk_windows.png)

# The relative errors of the whole volume:

[(window=name, error=round(norm(x - x_true) / norm(x_true); sigdigits=3))
 for (name, x) in (("ramp", x_ramp), ("Hann", x_fdk), ("Hann, cutoff 0.5", x_half))]

# ## Start image for an iterative method
#
# An iterative method starts from an image; the FDK image is a better start than zero. With CGNR
# from RegularizedLeastSquares.jl, starting from `x_fdk` is the same as solving for the correction
# `x - x_fdk` from the residual of the data, starting from zero:

function cgnr(x0; iterations=10)
  errors = Float64[]
  solver = createLinearSolver(CGNR, op; AHA=normalOperator(op), iterations)
  dx = solve!(solver, vec(y - op * x0);
              callbacks=(s, _) -> push!(errors, norm(solversolution(s) + vec(x0) - vec(x_true)) / norm(x_true)))
  x0 + reshape(dx, size(x0)), errors
end
x_cg, errors_fdk = cgnr(x_fdk)
x_zero, errors_zero = cgnr(zero(x_true));

# The error after each iteration (0: the start image), from zero and from the FDK image:

fig = Figure(size=(500, 320)) #hide
ax = Axis(fig[1, 1]; xlabel="iteration", ylabel="relative error") #hide
lines!(ax, errors_zero; label="start at zero") #hide
lines!(ax, errors_fdk; label="start at FDK") #hide
axislegend(ax) #hide
save("fdk_start.png", fig) #hide
nothing #hide
# ![Error of CGNR from zero and from the FDK image](fdk_start.png)

# ## Requirements
#
# FDK needs a full circular scan:
#
# - the source on a circle about the rotation axis, in a plane perpendicular to it (no helix);
# - the rotation axis along the detector columns (`v`), the detector rows (`u`) perpendicular to it,
#   a flat detector perpendicular to the ray from the source to the axis;
# - views around the whole circle (parallel beams: half or the whole circle), in order along it;
#   the angle steps may differ.
#
# Other geometries give an `ArgumentError`. How much of the circle the views must cover is set by the
# keyword `redundancy`, an [`AbstractRedundancy`](@ref): how lines measured more than once are
# weighted. The default, [`FullScan`](@ref), weights every view with half its angle step, since a full
# scan measures every line twice. Short scans need Parker weights, which are not implemented. Weights
# computed elsewhere can be passed as arrays with [`ArrayRedundancy`](@ref): a weight per data value
# and a weight per view. [`circular_cone_beam`](@ref) (flat or arc detector, with offsets) and
# [`parallel_beam`](@ref) build such scans, and geometries of Sinograms.jl work through
# [`projection_geometry`](@ref). With Sinograms.jl loaded, its function `fdk` also takes an `FDK`
# plan, `fdk(plan, y)`, or the image and a geometry of this package, `fdk(image, geometry, y)`.
#
# The back projection is that of the [`VoxelDrivenProjector`](@ref), with the weights of FDK; on a
# GPU, pass `storage` as for the projectors (see [Run on a GPU](@ref)).
