# Helpers shared by the tests.

using LinearAlgebra, StaticArrays

# Coordinates (u, v, ϕ, θ) of ImagePhantoms.jl's 3D `radon` for the line through a and b. Its lines
# are u e₁ + v e₃ + t e₂ with e₁ = (cos ϕ, sin ϕ, 0), e₂ = (−sin ϕ cos θ, cos ϕ cos θ, sin θ) (the
# direction) and e₃ = (sin ϕ sin θ, −cos ϕ sin θ, cos θ).
function line_coordinates(a, b)
  d = normalize(b - a)
  θ = asin(clamp(d[3], -1, 1))
  ϕ = atan(-d[1], d[2])
  sϕ, cϕ = sincos(ϕ)
  sθ, cθ = sincos(θ)
  e1 = SVector(cϕ, sϕ, 0)
  e3 = SVector(sϕ * sθ, -cϕ * sθ, cθ)
  dot(a, e1), dot(a, e3), ϕ, θ
end

# distance of the origin from the line through a and b
line_distance(a, b) = norm(cross(a, normalize(b - a)))

# all rays of a geometry on the CPU
allrays(g) = [ray(g, i) for i in 1:nrays(g)]
