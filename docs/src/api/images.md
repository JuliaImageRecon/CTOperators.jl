# Images

Images are described by `ImageGeom` from [ImageGeoms.jl](https://github.com/JuliaImageRecon/ImageGeoms.jl),
re-exported by CTOperators.jl together with the mask types `MaskCircle`, `MaskEllipse` and
`MaskAllButEdge`. Voxel `i` has its center at `((i - 1) - (n - 1)/2 - offset) * Δ` per dimension; a
2D image is treated as a slice at `z = 0` that is 1 length unit thick. Lengths have no fixed unit:
image and projection geometry must use the same one.

The mask of the `ImageGeom` is the image mask of a projector: the voxels that are reconstructed
(see [Use Masks](@ref)).

How the projectors use the image geometry internally (`VoxelGrid`) is described under
[Internals](@ref).
