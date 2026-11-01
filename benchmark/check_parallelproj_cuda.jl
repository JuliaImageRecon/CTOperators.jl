# Check of parallelproj's CUDA build before the benchmark, in its environment:
#   julia --project=benchmark/envs/parallelproj-cuda benchmark/check_parallelproj_cuda.jl
# Projects a random image along random rays with CuPy arrays (the data stays on the GPU) and with
# NumPy arrays (copied to the GPU per call), compares both and times them. If the CuPy call fails or
# is not faster than the NumPy one, parallelproj_core does not take device arrays directly and the
# adapter parallelproj-cuda would measure transfers.

using CondaPkg, Libdl
Libdl.dlopen(joinpath(CondaPkg.envdir(), "lib", "libparallelproj.so.2"), Libdl.RTLD_GLOBAL)
using PythonCall

np, cp, ppc = pyimport("numpy"), pyimport("cupy"), pyimport("parallelproj_core")
println("parallelproj_core ", ppc.__version__, ", CuPy ", cp.__version__, ", GPU ",
        cp.cuda.runtime.getDeviceProperties(0)["name"].decode())
println("libparallelproj: ", CondaPkg.envdir())
CondaPkg.status()   # shows the build of libparallelproj (cuda129 or cuda130)

n, nrays = 128, 1_000_000
rng = np.random.default_rng(1)
img = rng.random((n, n, n); dtype=np.float32)
starts = rng.uniform(-2n, 2n; size=(nrays, 3)).astype(np.float32)
ends = rng.uniform(-2n, 2n; size=(nrays, 3)).astype(np.float32)
origin = np.asarray(Float32[-(n - 1) / 2, -(n - 1) / 2, -(n - 1) / 2])
voxsize = np.asarray(Float32[1, 1, 1])

y_np = np.zeros(nrays; dtype=np.float32)
ppc.joseph3d_fwd(starts, ends, img, origin, voxsize, y_np)

starts_d, ends_d, img_d = cp.asarray(starts), cp.asarray(ends), cp.asarray(img)
y_d = cp.zeros(nrays; dtype=cp.float32)
ppc.joseph3d_fwd(starts_d, ends_d, img_d, origin, voxsize, y_d)
cp.cuda.Device().synchronize()
diff = pyconvert(Float64, np.abs(cp.asnumpy(y_d) - y_np).max() / np.abs(y_np).max())
println("largest difference CuPy against NumPy call (relative): ", diff)

timeit(f; reps=5) = minimum(begin
  t = time(); f(); cp.cuda.Device().synchronize(); time() - t
end for _ in 1:reps)
t_np = timeit(() -> ppc.joseph3d_fwd(starts, ends, img, origin, voxsize, y_np))
t_d = timeit(() -> ppc.joseph3d_fwd(starts_d, ends_d, img_d, origin, voxsize, y_d))
println("forward projection, 1e6 rays through 128³: NumPy arrays ", round(1e3t_np; digits=1), " ms, ",
        "CuPy arrays ", round(1e3t_d; digits=1), " ms")
println(diff < 1e-4 && t_d < t_np ? "OK: device arrays are used directly" :
        "CHECK: device arrays seem not to be used directly (see above)")
