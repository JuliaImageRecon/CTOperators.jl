# # Parallel beam in 2D
#
# The simplest CT scan: parallel rays through a slice of the object, from many directions. This
# tutorial simulates such a scan of the Shepp-Logan phantom, looks at its data, the sinogram, and
# reconstructs the slice, first by plain back projection and then iteratively.

using CTOperators, ImagePhantoms, RegularizedLeastSquares, CairoMakie
using ImageGeoms: axis
include(joinpath(pkgdir(CTOperators), "docs", "utils", "normal_operator.jl")) #hide
include(joinpath(pkgdir(CTOperators), "docs", "utils", "tutorial_figures.jl")) #hide

# ## The object
#
# A slice of 128 × 128 pixels of 1 mm. A 2D `ImageGeom` is a slice at `z = 0`; the rays run in its
# plane. The phantom, sampled with 2 × 2 points per pixel, is the ground truth.

n = 128
image = ImageGeom(; dims=(n, n), deltas=(1.0, 1.0))
x_true = Float32.(phantom(axis(image, 1), axis(image, 2), shepp_logan(SheppLoganToft(); fovs=(n, n)), 2));

# ## The scan
#
# A detector row of 192 cells of 1 mm, wide enough for the diagonal of the slice, measures the line
# integrals of parallel rays. Detector and rays rotate about the centre; since a ray and the ray in
# the opposite direction measure the same line integral, half a turn suffices: 180 views, one per
# degree. `nv = 1` makes it a single detector row.

angles = range(0, pi, 181)[1:180]
geometry = parallel_beam(; angles, nu=192, nv=1, du=1.0)

# Seen from above, for the first view and for a view at 60 degrees: the rays (orange, every 24th
# cell) all have the same direction and cross the slice (grey) to the detector (blue). Parallel rays
# are infinite lines: the detector marks where they are, not where they end.

fig = topview_figure([("view 1 (0 degrees)", geometry, 1), ("view 61 (60 degrees)", geometry, 61)], image; #hide
                     limits=(-160, 160, -160, 160)) #hide
save("parallel_scan.png", fig) #hide
nothing #hide
# ![The scan from above](parallel_scan.png)

# ## The data: a sinogram
#
# Joseph's method computes the line integrals: `op * x` projects the image `x` into the data of all
# views, a `192 × 1 × 180` array.

op = JosephProjector(image, geometry)
y = op * x_true
size(y)

# Stacked view by view, the detector rows form the sinogram. Every point of the object traces a sine
# curve in it: the position on the detector at which the point appears moves back and forth as the
# scanner turns, the further the point is from the centre the wider. The bright band is the skull of
# the phantom, which every ray through the head crosses twice.

fig = sinogram_figure(y; angles) #hide
save("parallel_sinogram.png", fig) #hide
nothing #hide
# ![Sinogram](parallel_sinogram.png)

# ## Reconstruction
#
# The back projection `adjoint(op) * y` smears every measured value back along its ray. It is not
# the inverse: the centre of the slice is crossed by rays of every view and the low spatial
# frequencies pile up, so the result is a blurred version of the object.

x_bp = adjoint(op) * y;

# An iterative method finds the image whose projections match the data, in the least-squares sense
# `min ‖A x − y‖²`. CGNR from
# [RegularizedLeastSquares.jl](https://github.com/JuliaImageRecon/RegularizedLeastSquares.jl) needs
# nothing but forward and back projections:

solver = createLinearSolver(CGNR, op; iterations=30)
x_cg = reshape(solve!(solver, vec(y)), n, n);

# The back projection (in its own grey scale) and the CGNR reconstruction, with its relative error
# to the phantom:

fig = images_figure([("phantom", x_true), ("back projection", x_bp, extrema(x_bp)), #hide
                    ("CGNR, 30 iterations", x_cg)]; #hide
                    colorrange=(0, 0.4), truth=x_true) #hide
save("parallel_reconstruction.png", fig) #hide
nothing #hide
# ![Phantom, back projection and CGNR](parallel_reconstruction.png)

# The data here is ideal: no noise and enough views.
# [Sparse views and limited angles](@ref) takes away views, and
# [Fan beam with a curved detector](@ref) replaces the parallel rays by those of a clinical scanner.
