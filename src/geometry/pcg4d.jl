# The hash pcg4d of M. Jarzynski and M. Olano, "Hash Functions for GPU Rendering", Journal of
# Computer Graphics Techniques 9(3), 21-38, 2020, ported from their HLSL
# (https://github.com/markjarzynski/PCG3D, pcg4d.hlsl), which carries this notice:
#
# Copyright 2020 Mark Jarzynski & Marc Olano
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
# OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# Four UInt32 inputs to four UInt32 outputs that look independent even for neighboring inputs;
# random numbers computed from indices, the same on every device. UInt32 arithmetic wraps.
@inline function pcg4d(x::UInt32, y::UInt32, z::UInt32, w::UInt32)
  x = x * 0x0019660d + 0x3c6ef35f
  y = y * 0x0019660d + 0x3c6ef35f
  z = z * 0x0019660d + 0x3c6ef35f
  w = w * 0x0019660d + 0x3c6ef35f

  x += y * w
  y += z * x
  z += x * y
  w += y * z

  x = xor(x, x >> 16)
  y = xor(y, y >> 16)
  z = xor(z, z >> 16)
  w = xor(w, w >> 16)

  x += y * w
  y += z * x
  z += x * y
  w += y * z

  x, y, z, w
end

# a number in [0, 1) from the upper 24 bits of a hash value (exact in Float32)
@inline unit_interval(::Type{T}, h::UInt32) where {T} = T(h >> 8) * T(0x1p-24)
