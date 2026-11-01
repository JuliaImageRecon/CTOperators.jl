# # Ordered Subsets and Normalization
#
# Algebraic methods such as SIRT and SART, and the ordered-subset variants of statistical methods,
# normalize each update by the sums of the rows and columns of the system matrix, and many of them
# cycle through subsets of the views. This guide shows the building blocks, `rowsums`/`colsums` and
# `subset_blocks`, and an OS-SART reconstruction of a 2D phantom with the solver `OSSART` of
# [RegularizedLeastSquares.jl](https://github.com/JuliaImageRecon/RegularizedLeastSquares.jl).

using CTOperators, ImagePhantoms, RegularizedLeastSquares, LinearOperatorCollection, CairoMakie
using ImageGeoms: axis
using LinearAlgebra: norm

# A 2D Shepp-Logan phantom of 128² pixels and 180 parallel-beam views:

n = 128
image = ImageGeom(; dims=(n, n), deltas=(1.0, 1.0))
x_true = Float32.(phantom(axis(image, 1), axis(image, 2), shepp_logan(SheppLoganToft(); fovs=(n, n)), 2))
geometry = parallel_beam(; angles=range(0, pi, 181)[1:180], nu=2n, nv=1, du=0.5)
op = JosephProjector(image, geometry)
y = op * x_true;

# ## Row and column sums
#
# `rowsums(op)` is `op * ones`: for a projector the length of every ray inside the image (exactly
# for Siddon's method; for Joseph's method approximately, since its image fades to zero just outside
# the volume). `colsums(op)` is `adjoint(op) * ones`: for every voxel the summed weights of all
# rays, its sensitivity. `r` has the shape of the projection data, `c` that of the image; both live
# on the device of the projector.

r = rowsums(op)
c = colsums(op)
size(r), size(c)

# ## Subsets
#
# `subset(op, views)` is the projector of some of the views, sharing the image geometry.
# Ordered-subset methods interleave the views so that consecutive subsets see the object from
# different directions. `subset_blocks(op, n)` returns the projectors of `n` interleaved subsets and,
# for each, the indices of its data in `vec(y)`. A `RowBlockOp` of
# [LinearOperatorCollection.jl](https://github.com/JuliaImageRecon/LinearOperatorCollection.jl) stacks
# them into one operator that equals `op`, but knows its blocks; the solver below takes it as the
# keyword `blocks`:

blocks = RowBlockOp(subset_blocks(op, 10)...)
length(rowblocks(blocks)), size(blocks) == size(op)

# ## OS-SART
#
# `OSSART` makes one update per block: it back projects the residual of the block, normalized by the
# row sums (the length of each ray in the image), and divides by the column sums (the sensitivity of
# each pixel to the block). Rays that miss the image and pixels that no ray hits have sums of zero
# and get no update. `PositiveRegularization` clips negative values after each block. Without
# `blocks`, the projector is a single block, and this is SIRT. A callback records the error of the start image
# and after each pass:

function reconstruct(; iterations=10, kwargs...)
  error = CompareSolutionCallback(vec(x_true), (ref, x) -> norm(x - ref) / norm(ref))
  solver = createLinearSolver(OSSART, op; reg=PositiveRegularization(), iterations, kwargs...)
  x = reshape(solve!(solver, vec(y); callbacks=error), n, n)
  x, error.results
end
x_os, err_os = reconstruct(; blocks)
x_sirt, err_sirt = reconstruct();

# Ten subsets reach in a few passes through the data what SIRT reaches only after many more:

fig = Figure(size=(900, 300))
ax = Axis(fig[1, 1]; xlabel="pass through all views", ylabel="relative error", title="convergence")
lines!(ax, 0:length(err_sirt)-1, err_sirt; label="SIRT")
lines!(ax, 0:length(err_os)-1, err_os; label="OS-SART, 10 subsets")
axislegend(ax)
for (j, (name, img)) in enumerate((("SIRT", x_sirt), ("OS-SART", x_os)))
  heatmap(fig[1, j+1], img; colormap=:grays, colorrange=(0, 0.4),
          axis=(; title="$name, 10 passes", aspect=DataAspect()))
end
save("os_sart.png", fig) #hide
nothing #hide
# ![OS-SART and SIRT](os_sart.png)

# On a GPU, the blocks of interleaved views gather their data with index vectors on the device;
# `subset_blocks(op, n; mode=:contiguous)` takes consecutive views instead, whose data needs no
# copies. The same blocks serve other ordered-subset methods, for example OS-EM for emission data,
# which divides by the column sums of the block as well.
