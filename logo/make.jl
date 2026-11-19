# The CTOperators.jl icon and logo, drawn with CTOperators.jl.
#
# Motif: the focal spot (red) on its trajectory (a dashed arc with a gap around the spot), the beam
# as a solid purple fan, and a flat detector whose stems are the line integrals through a mini
# Shepp-Logan phantom (not drawn), ending in green, blue and red dots. The line integrals are a
# forward projection of the rasterized phantom with a SiddonProjector and a RayList. The phantom
# lies across the beam, so the outer rays miss it and their dots sit on the detector line.
#
# The animation turns the motif about the phantom, which is drawn in grays above the beam; the
# trajectory arcs reach up to the detector, and the stems follow the views.
#
# Light and dark mode: the logo has both in one SVG (a prefers-color-scheme media query switches the
# ink); Documenter takes a separate icon per theme (assets/logo.svg, assets/logo-dark.svg).
#
#   julia --project=logo logo/make.jl
#
# writes docs/src/assets/: logo.svg, logo-dark.svg (Documenter's sidebar icon), ctoperators-logo.svg
# (README and home page) and ctoperators-rotation.svg (the animation, an SVG with SMIL animations,
# at the end of the home page).
#
# The wordmark is Inter Bold (SIL Open Font License 1.1) as outlines; the font is found with
# fontconfig, or set LOGO_FONT to the path of Inter-Bold.otf.

using CTOperators
using ImageGeoms: ImageGeom, axis
using FreeType
using Printf

const INK, INK_DARK = "#1A1A1A", "#F2F2EE"
const GREEN, PURPLE, RED, BLUE = "#389826", "#9558B2", "#CB3C33", "#4063D8"
const DOTS = (GREEN, BLUE, RED, GREEN, BLUE)   # top to bottom
const PHANTOM_GRAYS = ("#A8A8A8", "#6E6E6E")   # skull, and where the inner ellipses add up

fmt(x) = @sprintf("%.2f", x)

style() = "<style>.ink{stroke:$INK}.inkfill{fill:$INK}.inkarea{fill:$INK}" *
          "@media (prefers-color-scheme: dark){.ink{stroke:$INK_DARK}.inkfill{fill:$INK_DARK}" *
          ".inkarea{fill:$INK_DARK}}</style>"
style(ink) = "<style>.ink{stroke:$ink}.inkfill{fill:$ink}.inkarea{fill:$ink}</style>"

# ---------- layout of the motif, in a 32 x 32 design box (SVG coordinates, y down) ----------

const CENTER, RADIUS = (16.0, 16.0), 11.0       # the trajectory arc
const SOURCE, SOURCE_R = (5.0, 16.0), 2.6        # the focal spot
const DETECTOR_X = 20.5
const CELLS = (6.0, 11.0, 16.0, 21.0, 26.0)      # detector positions of the five rays
const DOT_R, LINE_W, DET_W = 1.75, 1.4, 1.8
const STEM_MIN = DETECTOR_X + DET_W / 2 + DOT_R  # a dot on the detector line: no stem
const STEM_MAX = 9.0                             # the longest stem, beyond DETECTOR_X + 0.5

# ---------- phantom and line integrals ----------

# a point rotated by `angle` degrees about `c`, as SVG's rotate(angle cx cy)
function rotate((x, y), angle, (cx, cy))
  c, s = cosd(angle), sind(angle)
  (cx + c * (x - cx) - s * (y - cy), cy + s * (x - cx) + c * (y - cy))
end

# mini Shepp-Logan around `center`, turned by `turn` degrees (as SVG's rotate): skull ellipse and
# two tilted inner ellipses (densities add up)
function shepp((cx, cy); turn=0.0)
  ellipses = [(; dx=0.0, dy=0.0, rx=3.6, ry=4.8, angle=0.0, density=1.0),
              (; dx=-1.3, dy=-0.3, rx=0.9, ry=2.4, angle=18.0, density=0.9),
              (; dx=1.3, dy=-0.3, rx=0.75, ry=2.0, angle=-18.0, density=0.9)]
  map(ellipses) do e
    x, y = rotate((cx + e.dx, cy + e.dy), turn, (cx, cy))
    (; cx=x, cy=y, e.rx, e.ry, angle=e.angle + turn, e.density)
  end
end

# between the spot and the detector, its long axis across the beam; the animation turns the motif
# about its center
const PHANTOM = shepp((12.5, 16.0); turn=-90.0)

phantom_center(phantom) = (phantom[1].cx, phantom[1].cy)

function inside(e, x, y)
  c, s = cosd(e.angle), sind(e.angle)
  u, v = (x - e.cx) * c + (y - e.cy) * s, -(x - e.cx) * s + (y - e.cy) * c
  (u / e.rx)^2 + (v / e.ry)^2 <= 1
end

# the phantom on a fine grid (x along the first axis, y along the second)
function rasterize(phantom; n=1000, width=10.0)
  h = width / n
  image = ImageGeom(; dims=(n, n), deltas=(h, h), offsets=.-phantom_center(phantom) ./ h)
  xs, ys = axis(image, 1), axis(image, 2)
  image, [sum(e.density for e in phantom if inside(e, x, y); init=0.0) for x in xs, y in ys]
end

"""
The line integrals through `phantom` of the five rays from the focal spot to the detector cells,
with the motif rotated by each of `angles` about `center`: a matrix of 5 rays x views.
"""
function line_integrals(angles; phantom=PHANTOM, center=phantom_center(phantom))
  image, x = rasterize(phantom)
  starts = zeros(3, length(CELLS), length(angles))
  ends = zeros(3, length(CELLS), length(angles))
  for (k, angle) in enumerate(angles), (i, y) in enumerate(CELLS)
    starts[1:2, i, k] .= rotate(SOURCE, angle, center)
    ends[1:2, i, k] .= rotate((DETECTOR_X, y), angle, center)
  end
  rays = RayList(reshape(starts, 3, :), reshape(ends, 3, :); datasize=(length(CELLS), length(angles)))
  op = SiddonProjector(image, rays; eltype=Float64)
  op * x
end

# the end of a stem for a line integral `v`, scaled by the largest `vmax`
stem(v, vmax) = max(STEM_MIN, DETECTOR_X + 0.5 + STEM_MAX * v / vmax)

# ---------- motif ----------

"""
The motif, placed at `(x, y)` and scaled by `scale`; `stems` are the x positions of the dots.

For an animation, `stems` is a matrix (rays x frames) and `rotation` a period in seconds: the motif
turns about the phantom center and the stems follow the frames. `phantom`, if given, is drawn in a
fixed layer above the beam, in opaque grays (denser is darker), the skull outlined in ink with width
`outline`; `tint` is the opacity of the beam over the phantom. `angle` turns the motif (not the
phantom) by a fixed angle in degrees, for single frames.
"""
function motif(stems; x=0.0, y=0.0, scale=1.0, angle=0.0, rotation=nothing, phantom=nothing, tint=0.0, outline=0.6,
               span=52.0, dash=2.2, gap=1.8, spot_gap=1.7)
  (cx, cy), r, (sx, sy) = CENTER, RADIUS, SOURCE
  (px, py) = phantom_center(PHANTOM)
  out = ["<g transform=\"translate($(fmt(x)),$(fmt(y))) scale($(@sprintf("%.4f", scale)))\">"]
  turning = isnothing(rotation) ? (iszero(angle) ? "<g>" : "<g transform=\"rotate($angle $px $py)\">") :
    "<g><animateTransform attributeName=\"transform\" type=\"rotate\" from=\"0 $px $py\" to=\"360 $px $py\" " *
    "dur=\"$(rotation)s\" repeatCount=\"indefinite\"/>"
  # beam: solid purple fan
  push!(out, turning)
  fan = "<path d=\"M$sx $sy L$DETECTOR_X $(fmt(CELLS[1] - 1.6)) L$DETECTOR_X $(fmt(CELLS[end] + 1.6)) Z\" " *
        "fill=\"$PURPLE\""
  push!(out, fan * "/>")
  push!(out, "</g>")
  # the phantom, fixed, above the beam
  if !isnothing(phantom)
    for (k, e) in enumerate(phantom)
      push!(out, "<ellipse cx=\"$(fmt(e.cx))\" cy=\"$(fmt(e.cy))\" rx=\"$(fmt(e.rx))\" ry=\"$(fmt(e.ry))\" " *
                 "transform=\"rotate($(e.angle) $(fmt(e.cx)) $(fmt(e.cy)))\" fill=\"$(PHANTOM_GRAYS[min(k, 2)])\"" *
                 (k == 1 ? " class=\"ink\" stroke-width=\"$outline\"/>" : "/>"))
    end
    tint > 0 && push!(out, turning * fan * " fill-opacity=\"$tint\"/></g>")
  end
  push!(out, turning)
  # trajectory: two halves from the spot outward (SVG y down: increasing angle = sweep flag 1), the
  # first dash starting `spot_gap` outside the spot
  p(a) = (cx + r * cosd(a), cy + r * sind(a))
  (ax, ay), (bx, by) = p(180 - span), p(180 + span)
  period = dash + gap
  offset = period - mod(SOURCE_R + spot_gap, period)
  for (sweep, (ex, ey)) in ((0, (ax, ay)), (1, (bx, by)))
    push!(out, "<path class=\"ink\" d=\"M$(fmt(sx)) $(fmt(sy)) A$r $r 0 0 $sweep $(fmt(ex)) $(fmt(ey))\" " *
               "fill=\"none\" stroke-width=\"$LINE_W\" stroke-dasharray=\"$dash $gap\" " *
               "stroke-dashoffset=\"$(@sprintf("%.3f", offset))\"/>")
  end
  # detector and projection: stems from the detector line to the dots
  push!(out, "<line class=\"ink\" x1=\"$DETECTOR_X\" y1=\"$(fmt(CELLS[1] - 2.6))\" x2=\"$DETECTOR_X\" " *
             "y2=\"$(fmt(CELLS[end] + 2.6))\" stroke-width=\"$DET_W\" stroke-linecap=\"round\"/>")
  for (k, yy) in enumerate(CELLS)
    tx = stems[k, 1]
    anim(attr) = isnothing(rotation) ? "" :
      "<animate attributeName=\"$attr\" values=\"$(join(fmt.([stems[k, :]; stems[k, 1]]), ';'))\" " *
      "dur=\"$(rotation)s\" repeatCount=\"indefinite\"/>"
    if !isnothing(rotation) || tx > STEM_MIN
      push!(out, "<line class=\"ink\" x1=\"$DETECTOR_X\" y1=\"$(fmt(yy))\" x2=\"$(fmt(tx))\" y2=\"$(fmt(yy))\" " *
                 "stroke-width=\"$LINE_W\" stroke-linecap=\"round\">$(anim("x2"))</line>")
    end
    push!(out, "<circle cx=\"$(fmt(tx))\" cy=\"$(fmt(yy))\" r=\"$DOT_R\" fill=\"$(DOTS[k])\">$(anim("cx"))</circle>")
  end
  # focal spot
  push!(out, "<circle cx=\"$sx\" cy=\"$sy\" r=\"$SOURCE_R\" fill=\"$RED\"/>")
  push!(out, "</g></g>")
  join(out)
end

svg(width, height, body; style=style(), viewbox="0 0 $(round(Int, width)) $(round(Int, height))") =
  "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"$viewbox\" width=\"$(round(Int, width))\" " *
  "height=\"$(round(Int, height))\"><title>CTOperators.jl</title>$style$body</svg>\n"

# ---------- wordmark (Inter Bold, outlined) ----------

const SUBSTITUTE = Dict('I' => "I.1")   # serifed capital I, so it does not read as a lowercase l

function font_path()
  haskey(ENV, "LOGO_FONT") && return ENV["LOGO_FONT"]
  path = readchomp(`fc-match -f "%{file}" "Inter:bold"`)
  occursin("Inter", basename(path)) || error("Inter Bold not found (fc-match gives $path); set LOGO_FONT")
  path
end

struct Font
  face::FT_Face
  upm::Int
end

function Font(path)
  library, face = Ref{FT_Library}(), Ref{FT_Face}()
  FT_Init_FreeType(library) == 0 || error("FreeType failed to start")
  FT_New_Face(library[], path, 0, face) == 0 || error("FreeType cannot read $path")
  Font(face[], unsafe_load(face[]).units_per_EM)
end

glyph_index(font::Font, name::String) = Int(FT_Get_Name_Index(font.face, name))
glyph_index(font::Font, c::Char) = Int(FT_Get_Char_Index(font.face, UInt(c)))

# the glyph slot after loading glyph `index` in font units
function load_glyph(font::Font, index)
  FT_Load_Glyph(font.face, index, FT_LOAD_NO_SCALE) == 0 || error("FreeType cannot load glyph $index")
  unsafe_load(unsafe_load(font.face).glyph)
end

# outline decomposition into SVG path commands; `transform` maps font units to the page
mutable struct PathState
  io::IOBuffer
  transform::Function
  open::Bool
end

function point(state, v::Ptr{FT_Vector})
  p = unsafe_load(v)
  x, y = state.transform(p.x, p.y)
  "$(fmt(x)) $(fmt(y))"
end

function move_to(to, user)
  state = unsafe_pointer_to_objref(user)::PathState
  state.open && print(state.io, "Z")
  print(state.io, "M", point(state, to))
  state.open = true
  Cint(0)
end
line_to(to, user) = (state = unsafe_pointer_to_objref(user)::PathState; print(state.io, "L", point(state, to)); Cint(0))
function conic_to(c, to, user)
  state = unsafe_pointer_to_objref(user)::PathState
  print(state.io, "Q", point(state, c), " ", point(state, to))
  Cint(0)
end
function cubic_to(c1, c2, to, user)
  state = unsafe_pointer_to_objref(user)::PathState
  print(state.io, "C", point(state, c1), " ", point(state, c2), " ", point(state, to))
  Cint(0)
end

function outline_path(slot, transform)
  funcs = Ref(FT_Outline_Funcs(@cfunction(move_to, Cint, (Ptr{FT_Vector}, Ptr{Cvoid})),
                               @cfunction(line_to, Cint, (Ptr{FT_Vector}, Ptr{Cvoid})),
                               @cfunction(conic_to, Cint, (Ptr{FT_Vector}, Ptr{FT_Vector}, Ptr{Cvoid})),
                               @cfunction(cubic_to, Cint, (Ptr{FT_Vector}, Ptr{FT_Vector}, Ptr{FT_Vector}, Ptr{Cvoid})),
                               0, 0))
  state = PathState(IOBuffer(), transform, false)
  outline = Ref(slot.outline)
  GC.@preserve state FT_Outline_Decompose(outline, funcs, pointer_from_objref(state))
  state.open && print(state.io, "Z")
  String(take!(state.io))
end

# glyph `index` at pen position (x, y) and font size `size`: the SVG path and the next pen position
function glyph(font::Font, index, size, x, y)
  slot = load_glyph(font, index)
  s = size / font.upm
  d = outline_path(slot, (px, py) -> (x + s * px, y - s * py))
  (isempty(d) ? "" : "<path d=\"$d\" class=\"inkfill\"/>"), x + slot.advance.x * s
end

"""
`text` plus '.jl', with the period and the dot of the j as Julia-colored dots; returns the SVG,
the end of the text and the cap height.
"""
function wordmark(font::Font, text, size, x, y)
  s = size / font.upm
  cap = load_glyph(font, glyph_index(font, 'H')).metrics.horiBearingY * s
  xheight = load_glyph(font, glyph_index(font, 'x')).metrics.horiBearingY * s
  out = String[]
  for c in text
    index = haskey(SUBSTITUTE, c) ? glyph_index(font, SUBSTITUTE[c]) : 0
    index == 0 && (index = glyph_index(font, c))
    path, x = glyph(font, index, size, x, y)
    push!(out, path)
  end
  r = 0.085 * size
  period_w = load_glyph(font, glyph_index(font, '.')).advance.x * s
  push!(out, "<circle cx=\"$(fmt(x + period_w / 2))\" cy=\"$(fmt(y - r))\" r=\"$(fmt(r))\" fill=\"$GREEN\"/>")
  x += period_w
  j = first(i for i in (glyph_index(font, n) for n in ("jdotless", "uni0237", "dotlessj")) if i != 0)
  m = load_glyph(font, j).metrics
  jx = x + (m.horiBearingX + m.width) * s - 0.065 * size
  path, x_after_j = glyph(font, j, size, x, y)
  push!(out, path)
  push!(out, "<circle cx=\"$(fmt(jx))\" cy=\"$(fmt(y - xheight - 1.6 * r))\" r=\"$(fmt(r))\" fill=\"$RED\"/>")
  path, x = glyph(font, glyph_index(font, 'l'), size, x_after_j, y)
  push!(out, path)
  join(out), x, cap
end

# ---------- output ----------

function logo(font, stems)
  size, height, base = 120, 200, 136
  wm, x_end, cap = wordmark(font, "CTOperators", size, 4 + 176 + 30, base)
  mid = base - cap / 2
  svg(x_end + 20, height, motif(stems; x=4, y=mid - 176 / 2, scale=176 / 32) * wm)
end

# the span of the trajectory arcs when they reach up to the detector, `gap` short of its line
detector_span(gap=1.0) = 180 - acosd((DETECTOR_X - DET_W / 2 - gap - CENTER[1]) / RADIUS)

function animation(; frames=180, period=12)
  angles = range(0, 360, frames + 1)[1:frames]
  v = line_integrals(angles)
  stems = stem.(v, maximum(v))
  # the motif turns about the phantom center; the farthest point is a dot at full length
  (px, py) = phantom_center(PHANTOM)
  reach = ceil(DETECTOR_X + 0.5 + STEM_MAX + DOT_R - px + 1)
  box = "$(px - reach) $(py - reach) $(2reach) $(2reach)"
  body = motif(stems; rotation=period, phantom=PHANTOM, span=detector_span())
  svg(400, 400, body; viewbox=box)
end

function main()
  assets = joinpath(@__DIR__, "..", "docs", "src", "assets")
  mkpath(assets)
  v = line_integrals([0.0])
  stems = stem.(v, maximum(v))
  # the icon at 256 px: Documenter shows it at its own size, up to 6rem
  icon(ink) = svg(256, 256, motif(stems); style=style(ink), viewbox="0 0 32 32")
  files = [joinpath(assets, "logo.svg") => icon(INK),
           joinpath(assets, "logo-dark.svg") => icon(INK_DARK),
           joinpath(assets, "ctoperators-logo.svg") => logo(Font(font_path()), stems),
           joinpath(assets, "ctoperators-rotation.svg") => animation()]
  for (path, content) in files
    write(path, content)
    println(normpath(path))
  end
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()
