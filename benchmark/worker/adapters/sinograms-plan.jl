# Sinograms.jl (MIT): `fdk` with `plan_fbp`, on the CPU (threaded over voxels), Float32, with the
# keyword `decon1=DECON1` of its filter (default true: the filter divided by sinc²). Included by the
# adapters sinograms-fdk and sinograms-fdk-nodecon1.

using Sinograms

library_version() = string(pkgversion(Sinograms))
method_name() = DECON1 ? "fdk" : "fdk, decon1=false"

# the Sinograms.jl geometry of each problem; converted, it must be the problem's geometry
const GEOMETRIES = Dict("fdk128" => () -> CtFanFlat(; ns=256, nt=128, ds=2.0f0, dt=2.0f0, na=360, dsd=1000.0f0, dod=500.0f0))
supports(p) = haskey(GEOMETRIES, p.name)

function setup(p)
  rg = GEOMETRIES[p.name]()
  g = projection_geometry(rg)
  same = typeof(g).name === typeof(p.geometry).name && datasize(g) == datasize(p.geometry) &&
         all(f -> isapprox(getfield(g, f), getfield(p.geometry, f); rtol=1e-6), (:source, :center, :u, :v))
  same || error("the Sinograms.jl geometry of $(p.name) is not the problem's geometry")
  plan_fbp(rg, p.image; decon1=DECON1)
end
reconstruct!(plan, x, y) = (x .= Sinograms.fdk(plan, y))
