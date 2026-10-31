# The geometries of Sinograms.jl as geometries of CTOperators.jl. Both follow the conventions of
# Fessler's codes (source at angle beta counter-clockwise from the y axis, detector rows along +x at
# beta = 0, offsets in cells), so the conversion is a direct translation of the parameters:
#
# - CtFanFlat, CtFanArc (3D cone beam, circular or helical source) -> ConeBeam, ArcConeBeam
# - CtPar (3D parallel beam) -> ParallelBeam
# - SinoFanFlat, SinoFanArc, SinoPar (2D) -> the same with one detector row of height 1 at z = 0
#
# Lengths lose their units (Unitful): they must be in the units of the image geometry. Angles are
# converted to radians by Sinograms' to_radians (degrees unless they carry units).
module CTOperatorsSinogramsExt

using CTOperators, Sinograms
using Sinograms: RayGeom, CtFanFlat, CtFanArc, CtPar, SinoFanFlat, SinoFanArc, SinoPar
using Sinograms: CtSourceCircle, CtSourceHelix

CTOperators.projection_geometry(rg::RayGeom) =
  throw(ArgumentError("no CTOperators geometry for $(nameof(typeof(rg))) (supported: CtFanFlat, CtFanArc, " *
                      "CtPar, SinoFanFlat, SinoFanArc, SinoPar)"))

# a length without its unit
strip_unit(x) = Float64(x / oneunit(x))

view_angles(rg) = Float64.(Sinograms.to_radians(collect(Sinograms.angles(rg))))

# Heights of the source per view, as Sinograms defines them: a helix advances by pitch times the
# axial field of view (at the rotation axis) per turn.
source_heights(rg, ::CtSourceCircle) = zeros(rg.na)
function source_heights(rg, src::CtSourceHelix)
  zfov = rg isa CtPar ? rg.nt * rg.dt : (rg.dsd - rg.dod) / rg.dsd * rg.nt * rg.dt
  per_turn = rg.na * 360 / strip_unit(rg.orbit)
  dz = rg.na == 1 ? 0.0 : src.pitch * strip_unit(zfov) / per_turn
  strip_unit(src.source_z0) .+ (0:rg.na-1) .* dz
end

# Cone beam: Sinograms' source_offset shifts source and detector sideways (along the detector
# rows), usually 0.
function fan_beam(rg, angles, nv, dv, offset_v, zs, detector)
  g = circular_cone_beam(; angles, dso=strip_unit(rg.dsd - rg.dod), dsd=strip_unit(rg.dsd), nu=nb(rg), nv,
                         du=strip_unit(cell(rg)), dv, offset_u=Float64(offset(rg)), offset_v, source_z=zs, detector)
  so = strip_unit(rg.source_offset)
  iszero(so) && return g
  shift = so .* vcat(cos.(angles)', sin.(angles)', zeros(1, length(angles)))
  shifted(g, g.source .+ shift, g.centre .+ shift)
end

shifted(g::ConeBeam, source, centre) = ConeBeam(source, centre, g.u, g.v; nu=g.nu, nv=g.nv)
shifted(g::ArcConeBeam, source, centre) = ArcConeBeam(source, centre, g.u, g.v; nu=g.nu, nv=g.nv)

nb(rg::Union{CtFanFlat,CtFanArc,CtPar}) = rg.ns
nb(rg::Union{SinoFanFlat,SinoFanArc,SinoPar}) = rg.nb
cell(rg::Union{CtFanFlat,CtFanArc,CtPar}) = rg.ds
cell(rg::Union{SinoFanFlat,SinoFanArc,SinoPar}) = rg.d
offset(rg::Union{CtFanFlat,CtFanArc,CtPar}) = rg.offset_s
offset(rg::Union{SinoFanFlat,SinoFanArc,SinoPar}) = rg.offset

detector_kind(::Union{CtFanFlat,SinoFanFlat}) = :flat
detector_kind(::Union{CtFanArc,SinoFanArc}) = :arc

CTOperators.projection_geometry(rg::Union{CtFanFlat,CtFanArc}) =
  fan_beam(rg, view_angles(rg), rg.nt, strip_unit(rg.dt), Float64(rg.offset_t), source_heights(rg, rg.src),
           detector_kind(rg))

CTOperators.projection_geometry(rg::Union{SinoFanFlat,SinoFanArc}) =
  fan_beam(rg, view_angles(rg), 1, 1.0, 0.0, 0.0, detector_kind(rg))

function CTOperators.projection_geometry(rg::CtPar)
  rg.src isa CtSourceCircle || throw(ArgumentError("CtPar with a helical source is not supported"))
  parallel_beam(; angles=view_angles(rg), nu=rg.ns, nv=rg.nt, du=strip_unit(rg.ds), dv=strip_unit(rg.dt),
                offset_u=Float64(rg.offset_s), offset_v=Float64(rg.offset_t))
end

CTOperators.projection_geometry(rg::SinoPar) =
  parallel_beam(; angles=view_angles(rg), nu=rg.nb, nv=1, du=strip_unit(rg.d), offset_u=Float64(rg.offset), dv=1.0)

CTOperators.ConeBeam(rg::Union{CtFanFlat,SinoFanFlat}) = projection_geometry(rg)
CTOperators.ArcConeBeam(rg::Union{CtFanArc,SinoFanArc}) = projection_geometry(rg)
CTOperators.ParallelBeam(rg::Union{CtPar,SinoPar}) = projection_geometry(rg)

end
