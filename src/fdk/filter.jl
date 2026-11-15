# The ramp filter of FDK along the detector rows, as the frequency response of a discrete convolution
# applied with FFTs (FFTL.jl). The convolution kernel is the band-limited ramp sampled at the cell
# spacing du, in the length unit of the detector (see also Kak & Slaney 1988, chapter 3):
#
#   h(0) = 1 / (4 du^2),   h(n) = 0 for even n,   h(n) = -1 / (n pi du)^2 for odd n,
#
# and for a detector on an arc of radius dsd (cells at angles n alpha, alpha = du / dsd) the
# equiangular kernel in arc length, h(n) (n alpha / sin(n alpha))^2. The
# filtered row is du * sum_m h(n - m) q(m). The rows are padded with zeros to at least 2 nu - 1
# values, so the circular convolution of the FFT equals the linear one (no wraparound).

ramp_tap(n, du) = n == 0 ? 1 / (4 * du^2) : iseven(n) ? 0.0 : -1 / (n * pi * du)^2

function ramp_tap(n, du, dsd)
  n == 0 && return 1 / (4 * du^2)
  iseven(n) && return 0.0
  -1 / (pi * dsd * sin(n * du / dsd))^2
end

# The weight of the window `window` at frequency `f`, a fraction of the cutoff frequency (0 <= f <= 1):
# :ramp (no window, weight 1), :hann, :hamming, :cosine, :shepplogan, or a function (any callable) of f.
window_weight(window, f) = window(f)

function window_weight(window::Symbol, f)
  window === :ramp && return 1.0
  window === :hann && return (1 + cospi(f)) / 2
  window === :hamming && return 0.54 + 0.46 * cospi(f)
  window === :cosine && return cospi(f / 2)
  window === :shepplogan && return sinc(f / 2)
  throw(ArgumentError("unknown window :$window; use :ramp, :hann, :hamming, :cosine, :shepplogan or a function"))
end

# the length of the padded rows: at least 2 nu - 1, with small prime factors for fast FFTs
padded_length(nu) = nextprod((2, 3, 5), 2 * nu - 1)

# The frequency response (length div(npad, 2) + 1, real) of the filter with the window, divided by npad
# for the unnormalized inverse transform (brfft). Frequencies above `cutoff` (a fraction of the
# Nyquist frequency of the cells) get 0. `tap(n)` is the kernel without the factor du.
function filter_response(tap, nu, du, npad, window, cutoff)
  0 < cutoff <= 1 || throw(ArgumentError("cutoff must be in (0, 1] (a fraction of the Nyquist frequency), got $cutoff"))
  window_weight(window, 0.0)        # unknown windows fail here, at construction
  kernel = zeros(npad)
  for n in -(nu - 1):(nu - 1)
    kernel[mod(n, npad)+1] = du * tap(n)
  end
  response = real.(FFTL.rfft(kernel))
  for j in eachindex(response)
    f = 2 * (j - 1) / npad / cutoff
    response[j] *= f <= 1 ? window_weight(window, f) : 0.0
  end
  response ./ npad
end
