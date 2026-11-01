# # Getting Started
#
# This example simulates a cone-beam CT scan of the 3D Shepp-Logan phantom in which the object moves
# along the rotation axis during the scan, and reconstructs it iteratively with
# [RegularizedLeastSquares.jl](https://github.com/JuliaImageRecon/RegularizedLeastSquares.jl).

using CTOperators, ImagePhantoms, RegularizedLeastSquares, LinearOperatorCollection
using ImageGeoms: axis
using CairoMakie, Random   # for the figures and the noise of this example
using WGLMakie, Bonito #hide
WGLMakie.activate!(px_per_unit=2, scalefactor=1) #hide
CairoMakie.activate!() #hide
include(joinpath(pkgdir(CTOperators), "docs", "utils", "bonito_page.jl")) #hide
bonito_page("getting_started") #hide

# ## The image
#
# The image is the voxel grid we reconstruct into, described by an `ImageGeom` from
# [ImageGeoms.jl](https://github.com/JuliaImageRecon/ImageGeoms.jl) (re-exported by CTOperators.jl;
# its function `axis` gives the voxel centers along one dimension): 64³ voxels of 1 mm, centered at
# the origin.

n = 64
image = ImageGeom(; dims=(n, n, n), deltas=(1.0, 1.0, 1.0));

# The object is the 3D Shepp-Logan phantom of
# [ImagePhantoms.jl](https://github.com/JuliaImageRecon/ImagePhantoms.jl), a set of ellipsoids.
# Sampled on the image grid (each voxel the average of 2³ samples) it is the ground truth.

objects = ellipsoid(ellipsoid_parameters(; fovs=(n, n, n)))
x_true = Float32.(phantom(axis(image, 1), axis(image, 2), axis(image, 3), objects, 2));

# ## The scan
#
# In a cone-beam CT scanner an X-ray source and a flat detector sit on opposite sides of the object.
# The source sends a cone of rays through the object to the detector. Each detector cell measures
# how much the ray reaching it was attenuated; the logarithm of this measurement is the line
# integral of the attenuation coefficient along the ray. These line integrals are the data. Source
# and detector rotate about the object, and in our scan the object also moves along the rotation
# axis.
#
# The geometry describes this: per view the position of the source and of the detector. The source
# is 128 mm from the rotation axis and 256 mm from the detector, so the object is magnified by 2 on
# the detector. The detector has 192 × 96 cells of 1 mm; at the rotation axis a cell covers 0.5 mm,
# so two rays pass through each voxel per direction. (With one ray per voxel, the rays of a view
# cover the voxels unevenly; see [Siddon's Method](@ref).) The object moves by 32 mm, half the
# volume height, per turn. In the object's frame, source and detector therefore move on a helix: two
# turns, 240 views.

geometry = helical_cone_beam(; angles=range(0, 4pi, 240), pitch=n / 2, z0=-n / 2, dso=2n, dsd=4n,
                             nu=192, nv=96, du=1.0, dv=1.0);

# The projector holds the geometry and is created once; it is then applied many times. `op * x`
# computes the line integrals of the image `x` along all rays, `adjoint(op) * y` the back
# projection. Siddon's method computes a line integral by walking along the ray from voxel to voxel
# and adding each voxel's value times the length of the ray inside it.

op = SiddonProjector(image, geometry);

# The simulated measurement: the projection of the phantom with Gaussian noise of 0.1 % of its
# maximum, one image of 192 × 96 values per view.

Random.seed!(1)
y = op * x_true
y .+= 0.001f0 * maximum(y) .* randn(Float32, size(y))
size(y)

# ### One view
#
# The phantom's ellipsoids in the volume, the source (orange dot), some of its rays and the
# detector, colored by the measured line integrals: bright where the rays crossed much of the
# object. Rotate and zoom with the mouse.

include(joinpath(pkgdir(CTOperators), "docs", "utils", "scan_figures.jl")) #hide
params = ellipsoid_parameters(; fovs=(n, n, n)) #hide
single_view_app(geometry, params, y, n) #hide

# ### The motion
#
# Over the scan the source runs on its helix around the object (orange line). The slider moves
# through the views: source, cone and detector with the projection of that view. The cone moves up
# along the object: its lower parts leave the cone while the upper parts enter it.

motion_app(geometry, params, y, n) #hide

# ## Reconstruction
#
# CGNR solves the least-squares problem `min ‖A x − y‖²` with the projector `op` as `A`, using only
# forward and back projections. It also needs the normal operator `adjoint(op) * op`, which
# `normalOperator` of
# [LinearOperatorCollection.jl](https://github.com/JuliaImageRecon/LinearOperatorCollection.jl)
# provides; it computes a forward and a back projection without allocating new arrays. The solvers
# work on vectors, hence `vec(y)` and the `reshape` of the result.

solver = createLinearSolver(CGNR, op; AHA=normalOperator(op), iterations=40);
x = reshape(solve!(solver, vec(y)), size(x_true));

# Slices through the small structures of the phantom: axial (x–y plane) at z = −8.5 mm and sagittal
# (y–z plane) at x = −0.5 mm. The axes are voxel indices. The window from 0.98 to 1.06 shows the
# contrasts of 1–2 % against the background of 1. Without regularization, the noise of the data is
# amplified in the reconstruction; the tutorial [Sparse Views and Limited Angles](@ref) adds a
# regularization term.

fig = Figure(size=(700, 560))
window = (0.98, 1.06)
for (i, (name, img)) in enumerate((("phantom", x_true), ("reconstruction", x)))
  heatmap(fig[i, 1], img[:, :, 24]; colormap=:grays, colorrange=window,
          axis=(; title="$name, axial", aspect=DataAspect()))
  heatmap(fig[i, 2], img[div(n, 2), :, :]; colormap=:grays, colorrange=window,
          axis=(; title="$name, sagittal", aspect=DataAspect()))
end
save("slices.png", fig) #hide
nothing #hide
# ![Slices of phantom and reconstruction](slices.png)

# ## On the GPU
#
# The same code runs on a GPU when the GPU package is loaded, the projector is created with the
# GPU's array type and the data lives on the GPU; the solver then also works on the GPU. With
# CUDA.jl:
#
# ```julia
# using CUDA
# op = SiddonProjector(image, geometry; storage=CuArray)
# y = op * CuArray(x_true)
# ```
#
# See [Run on a GPU](@ref) for details.
