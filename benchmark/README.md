# Benchmarks of CTOperators.jl

Compares the projectors of CTOperators.jl (Siddon, Joseph) with parallelproj (Joseph's method, C/OpenMP, Python) and
RadonKA.jl (parallel beam as a stack of 2D slices). Run from the repository root with any Julia; the
orchestrator uses the standard library only.

```
julia benchmark/bench.jl list                       # suites, problems, libraries
julia benchmark/bench.jl run quick                  # all libraries, 128³ problems, ≈ 3 min
julia benchmark/bench.jl run standard --threads 1,24
julia benchmark/bench.jl run quick --libs ctoperators-joseph,parallelproj --budget 1
julia benchmark/bench.jl run standard --libs ctoperators-siddon-rocm,ctoperators-joseph-rocm,radonka-rocm --threads 1
julia benchmark/bench.jl run standard --libs ctoperators-siddon-cuda,ctoperators-joseph-cuda,parallelproj-cuda --threads 1
```

GPU libraries (`*-rocm`: AMD GPUs through AMDGPU.jl; `*-cuda`: NVIDIA GPUs through CUDA.jl, and
`parallelproj-cuda`, the CUDA build of parallelproj with CuPy arrays in its own environment
`envs/parallelproj-cuda`) run only when named with `--libs`. Before a first CUDA run,
`julia --project=benchmark/envs/parallelproj-cuda benchmark/check_parallelproj_cuda.jl` checks that
parallelproj computes on CuPy arrays directly (otherwise its timings would include transfers). To choose
a `LaunchConfig` for a GPU, `julia --project=benchmark/envs/ctoperators benchmark/tune_launch.jl
<cuda|rocm> [device] [problem] [projector]` times workgroup sizes and tiles of detector cells on one
problem.

Options: `--libs` (comma separated), `--threads` (comma separated, default all cores), `--device`
(the GPU of GPU libraries, counted from 1, default 1; e.g. a free one on a shared machine), `--budget`
(seconds of repeated calls per measurement, default 2). Measure on a quiet machine.

## How it works

Each library runs in a worker process with its own environment (`envs/<lib>/Project.toml`,
instantiated on first use; the methods of CTOperators.jl share `envs/ctoperators`), so competitors never become dependencies of CTOperators.jl, and each
library gets exactly the requested number of threads (`--threads` for Julia, `OMP_NUM_THREADS` for
OpenMP).

- `src/problems.jl`: the problems (image geometry, projection geometry), a phantom of three
  ellipsoids (ImagePhantoms.jl) and the rays on which the error is measured, generated
  deterministically so that every worker sees the same problem. `src/suites.jl`: the suites.
- `worker/worker.jl`: per problem the setup time (second call), forward and back projection
  (minimum and median of repeated calls within the budget; the back projection of random data), and
  the relative error of the forward projection of the voxelized phantom against its analytic line
  integrals on up to 20 000 rays (where the library takes our rays).
- `worker/adapters/<lib>.jl`: maps our problem onto the library:
  - `ctoperators-siddon`, `ctoperators-joseph`: `SiddonProjector`, `JosephProjector`, Float32,
    KernelAbstractions' CPU backend.
  - `parallelproj`: `parallelproj_core.joseph3d_fwd/back` with the rays of our geometry as start
    and end points (parallel-beam lines become segments through the volume). Installed from
    conda-forge by CondaPkg.jl into `envs/parallelproj/.CondaPkg` (its compiled backend is not on
    PyPI); arrays are passed without copies (a Julia `(nx, ny, nz)` array is a C-ordered
    `(nz, ny, nx)` array).
  - `radonka`: `radon`/`backproject` on the same image size, number of angles and detector cells
    per slice. Its geometry is not expressed in our rays, so only the time is compared.
  - `ctoperators-siddon-rocm`, `ctoperators-joseph-rocm`, `radonka-rocm`: the same on an AMD GPU,
    with image, data (and RadonKA's angles) as `ROCArray`s. Adapters for devices define
    `device_array` (moves the worker's arrays to the device), `synchronize` (the timings wait for
    the device) and `device_name`; the defaults are those of the CPU. RadonKA requires
    KernelAbstractions 0.9, so its environment resolves an older AMDGPU (2.8) than ours (2.10).

Results go to `out/<time>-<suite>.csv` (not versioned), one row per library, problem and thread
count, with the device.

The methods differ: Siddon's weights are exact intersection lengths, Joseph's interpolate linearly
within planes, RadonKA's interpolate in a circular field of view. The error column shows what each
costs in accuracy on the same phantom.

## Results

CPU only, Float32, AMD Ryzen 9 9900X (12 cores, 24 threads), Julia 1.13.3, CTOperators at e427496,
parallelproj 2.0.7, RadonKA 0.6.3; quick suite, minimum time, 2026-10-31. CTOperators (Joseph) measured
later the same day; Siddon and parallelproj, measured again in that run, agree with these rows to
10 %.

| problem | library | forward, 24 threads | back, 24 threads | forward, 1 thread | back, 1 thread | phantom error |
|---|---|---|---|---|---|---|
| cone128 | CTOperators (Siddon) | 82 ms | 167 ms | 1.26 s | 2.17 s | 0.0091 |
| cone128 | CTOperators (Joseph) | 136 ms | 333 ms | 1.74 s | 5.05 s | 0.0086 |
| cone128 | parallelproj (Joseph) | 139 ms | 375 ms | 1.62 s | 5.40 s | 0.0086 |
| parallel128 | CTOperators (Siddon) | 62 ms | 133 ms | 0.79 s | 2.01 s | 0.0124 |
| parallel128 | CTOperators (Joseph) | 104 ms | 302 ms | 1.32 s | 5.08 s | 0.0085 |
| parallel128 | parallelproj (Joseph) | 146 ms | 363 ms | 1.61 s | 5.50 s | 0.0085 |
| parallel128 | RadonKA | 177 ms | 185 ms | 3.45 s | 3.39 s | — |

cone128: 128³ volume, cone beam with a 128² detector and 180 views (2.95 M rays). parallel128: 128³
volume, parallel beam with a 128² detector and 180 views. Phantom error: relative error of the
projected voxelized phantom against its analytic line integrals.

CTOperators' Joseph gives the same phantom error as parallelproj's (the same method). It runs at the
same speed for cone128 and 1.2 to 1.4× faster for parallel128. On the CPU, Siddon is 1.4 to 1.7×
faster than Joseph forward and 2.0 to 2.5× back.

### GPU

AMD Radeon RX 7900 XT (ROCm 7.2, AMDGPU.jl 2.10; RadonKA with AMDGPU 2.8), Float32, standard suite,
minimum time, 2026-11-01. CTOperators with the default launch configuration of GPUs (workgroups of
64, tiles of 8 × 8 detector cells, see `LaunchConfig`); the column "forward, before" is the earlier
run at 4d3b472 with the backend's workgroups and data order, as are RadonKA's rows. The CPU columns
(24 threads) were measured with the earlier run; the CPU code path is unchanged. The phantom
errors on the GPU equal those on the CPU.

| problem | library | forward, GPU | forward, before | back, GPU | forward, CPU | back, CPU | speedup forward / back |
|---|---|---|---|---|---|---|---|
| cone128 | CTOperators (Siddon) | 5.6 ms | 5.6 ms | 13.5 ms | 77 ms | 160 ms | 14× / 12× |
| cone128 | CTOperators (Joseph) | 6.3 ms | 7.2 ms | 30.7 ms | 138 ms | 336 ms | 22× / 11× |
| parallel128 | CTOperators (Siddon) | 4.9 ms | 5.6 ms | 10.9 ms | 57 ms | 134 ms | 12× / 12× |
| parallel128 | CTOperators (Joseph) | 5.9 ms | 7.2 ms | 28.8 ms | 105 ms | 310 ms | 18× / 11× |
| parallel128 | RadonKA | 20.3 ms | – | 30.4 ms | 180 ms | 186 ms | 9× / 6× |
| cone256 | CTOperators (Siddon) | 109 ms | 147 ms | 260 ms | 1.21 s | 2.48 s | 11× / 10× |
| cone256 | CTOperators (Joseph) | 108 ms | 231 ms | 537 ms | 2.26 s | 5.15 s | 21× / 10× |
| helix256 | CTOperators (Siddon) | 53 ms | 61 ms | 118 ms | 0.65 s | 1.29 s | 12× / 11× |
| helix256 | CTOperators (Joseph) | 59 ms | 104 ms | 272 ms | 1.30 s | 2.82 s | 22× / 10× |
| parallel256 | CTOperators (Siddon) | 102 ms | 166 ms | 236 ms | 0.89 s | 2.05 s | 9× / 9× |
| parallel256 | CTOperators (Joseph) | 111 ms | 224 ms | 512 ms | 1.74 s | 4.85 s | 16× / 9× |
| parallel256 | RadonKA | 91 ms | – | 252 ms | 2.96 s | 3.02 s | 33× / 12× |

cone256: 256³ volume, cone beam with a 256² detector and 360 views (23.6 M rays); helix256: 256³
volume, helical cone beam with 540 views over three turns; parallel256: 256³ volume, parallel beam
with a 256² detector and 360 views.

Observations:

- Launch configuration (`LaunchConfig`): workgroups of 64 instead of the backend's default (chosen
  for maximal occupancy) make the forward projection 1.2–1.9× faster even in data order; tiles of
  8 × 8 detector cells per workgroup instead of consecutive cells of a row give Joseph another
  1.4–1.8×, Siddon little. Together at 256³: Joseph about 2×, Siddon 1.2–1.6× faster, at 128³ up
  to 1.25×. The back projection gains nothing (1–6 % slower, close to the noise); it runs with the same configuration to keep one code path. On the CPU tiles
  change the forward projection by less than 4 % (8², 16², 32², 24 threads), so the CPU keeps data
  order and the backend's workgroups. A count of the cache lines per load instruction of a
  wavefront (32 lanes) gives about 17 in both orders. The cause of the gain was not measured.
- Ray setup is negligible: computing and clipping all rays of cone256 takes 0.8 ms, and a
  precomputed ray list (540 MB) gives the same times.
- From 128³ to 256³ the work grows 16× (8× the rays, each twice as long), our GPU times 17–21×
  forward (26–32× before) and 17–22× back.
- RadonKA, slice by slice, is faster than our parallel-beam forward projection at 256³ (91 ms
  against 102 ms), slower at 128³ (20 ms against 4.9 ms).
- Joseph's back projection takes 4–5× its forward projection, Siddon's about 2.3×.

### Masks (GPU)

AMD Radeon RX 7900 XT, Float32, cone256 (256³ volume, 256² detector, 360 views), minimum time,
2026-11-02, branch `nh/masks` (the GPU was also the desktop GPU in use: approximate). Image masks:
the circular field of view (`MaskCircle`) and a box of 64³ voxels in the center; ray mask: 20 % of
the rays masked at random.

| image mask | ray mask | Siddon forward / back | Joseph forward / back |
|---|---|---|---|
| none | none | 111 / 264 ms | 113 / 537 ms |
| none | 20 % masked | 96 / 212 ms | 108 / 438 ms |
| circle | none | 111 / 257 ms | 115 / 542 ms |
| box 64³ | none | 2.6 / 6.7 ms | 3.1 / 18.0 ms |
| box 64³ | 20 % masked | 2.2 / 5.5 ms | 2.9 / 15.0 ms |

Rays are clipped to the bounding box of the image mask: a small region of interest makes the
projections 30 to 40× faster; the circle fills its box and gains nothing.
Masked rays are not traced: the back projection gets faster by about the masked fraction.


### Several rays per cell (GPU)

AMD Radeon RX 7900 XT, Float32, 128³ volume, 128² detector, 180 views (circular cone beam),
minimum of 5 runs, 2026-11-03, branch `nh/multiray` (the GPU was also the desktop GPU in use:
approximate).

| geometry | Siddon forward / back | Joseph forward / back |
|---|---|---|
| plain | 6.0 / 13.7 ms | 6.8 / 31.0 ms |
| `MultiRay`, `RegularPoints(1)` | 5.8 / 13.5 ms | 6.8 / 31.0 ms |
| `MultiRay`, `RegularPoints(2)` | 24.7 / 53.3 ms | 29.9 / 123.7 ms |
| `MultiRay`, `JitteredPoints(2)` | 25.8 / 54.9 ms | 30.2 / 125.4 ms |
| `MultiRay`, `JitteredPoints(2)`, `GaussianWeights(0.4)` | 24.1 / 54.9 ms | 29.8 / 125.7 ms |

One ray per cell costs the same as the plain geometry; 4 rays per cell take about 4 times as long.
Random positions (a hash per ray) and Gaussian weights are not measurable.

### Voxel indices in Int32 (GPU)

AMD Radeon RX 7900 XT, Float32, 128³ volume, 128² detector, 180 views (circular cone beam), one ray
per cell, minimum of 15 runs, two runs per version, 2026-11-04 (the GPU was also the desktop GPU in
use: approximate).

| voxel indices | Siddon forward / back | Joseph forward / back |
|---|---|---|
| Int64 | 5.9–6.4 / 13.4–13.6 ms | 6.6–6.8 / 30.6 ms |
| Int32 | 6.0–6.1 / 13.3–13.4 ms | 5.7–6.0 / 30.5 ms |

### Float16 and BFloat16 images (GPU)

AMD Radeon RX 7900 XT, projectors in Float32, one ray per cell, random images, circular cone beam
(`n³` volume, `n²` detector, `dso = 2n`, `dsd = 4n`), minimum of 10 or 15 runs, 2026-11-05 to
2026-11-08 (the GPU was also the desktop GPU in use: approximate). Two runs per case at 256³.

| case | image | Siddon | Joseph |
|---|---|---|---|
| forward, 256³, 360 views | Float32 | 109.3, 109.3 ms | 98.5, 97.8 ms |
| forward, 256³, 360 views | Float16 | 93.2, 94.4 ms | 84.1, 84.1 ms |
| forward, 128³, 180 views | Float32 | 6.14 ms | 5.41 ms |
| forward, 128³, 180 views | Float16 | 5.16 ms | 5.08 ms |
| back, 128³, 180 views, from Float32 / Float16 data | Float32 | 13.69 / 13.36 ms | 30.53 / 30.51 ms |

The forward projection with a Float16 image took about 14 % less time at 256³; the back projection
did not change. The 256³ forward times, measured again with values converted on reading
(`nh/mixedprecision`), were 108.9 and 109.2 ms against 94.2 and 94.5 ms (Siddon) and 98.3 and
101.0 ms against 83.1 and 83.7 ms (Joseph).

BFloat16 (`nh/bfloat16`, 2026-11-09, same GPU, measured with a script: minimum of 3 runs, 128³, 180
views, random images and data). Images and data stored in the given type, projectors in Float32;
the last column is the back projection into an image of the given type (atomic additions in that
type):

| projector | storage | forward | back into Float32 | back into the storage type |
|---|---|---|---|---|
| Siddon | Float32 | 6.5 ms | 15.1 ms | – |
| Siddon | Float16 | 5.0 ms | 14.9 ms | 17.5 ms |
| Siddon | BFloat16 | 5.2 ms | 14.8 ms | 17.1 ms |
| Joseph | Float32 | 5.8 ms | 34.5 ms | – |
| Joseph | Float16 | 4.7 ms | 34.6 ms | 38.7 ms |
| Joseph | BFloat16 | 5.1 ms | 34.5 ms | 36.5 ms |
| distance driven | Float32 | 12.9 ms | 32.1 ms | – |
| distance driven | Float16 | 10.1 ms | 32.1 ms | 41.8 ms |
| distance driven | BFloat16 | 10.5 ms | 32.1 ms | 41.2 ms |

BFloat16 storage took about as long as Float16 storage; the back projection into a Float16 or
BFloat16 image took longer than into a Float32 image.

### Fused normal operator

`Aᵀ A x` with the fused normal operator (one pass over the rays) against forward projection into a
preallocated buffer followed by the back projection, Float32, one ray per cell, random images,
circular cone beam (`dso = 2n`, `dsd = 4n`), 2026-11-08, branch `nh/fusednormal`. CPU: 64³, 64²
detector, 90 views, all threads, minimum of 5 runs. GPU: AMD Radeon RX 7900 XT, 128³, 128²
detector, 180 views, minimum of 10 runs (also the desktop GPU in use: approximate). Two runs each.

| | separate | fused | separate, weights | fused, weights |
|---|---|---|---|---|
| CPU, Siddon | 20.6, 21.0 ms | 20.3, 19.6 ms | 20.9 ms | 19.6 ms |
| CPU, Joseph | 39.5, 37.3 ms | 37.7, 38.0 ms | 41.0 ms | 39.6 ms |
| GPU, Siddon | 20.0, 19.4 ms | 19.8, 20.0 ms | 19.5 ms | 19.8 ms |
| GPU, Joseph | 36.7, 36.6 ms | 35.0, 34.9 ms | 36.8 ms | 35.0 ms |

The buffer of the separate version holds the projection data (1.4 MiB on the CPU, 11.2 MiB on the
GPU here).
