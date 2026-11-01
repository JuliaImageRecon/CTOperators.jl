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
```

GPU libraries (`*-rocm`, AMD GPUs through AMDGPU.jl) run only when named with `--libs`.

Options: `--libs` (comma separated), `--threads` (comma separated, default all cores), `--budget`
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
  the relative error of the forward projection of the voxelised phantom against its analytic line
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
projected voxelised phantom against its analytic line integrals.

CTOperators' Joseph gives the same phantom error as parallelproj's (the same method) and runs at the
same speed; Siddon is about twice as fast as Joseph on the CPU (one memory read per visited voxel
against four per sample).

### GPU

AMD Radeon RX 7900 XT (ROCm 7.2, AMDGPU.jl 2.10; RadonKA with AMDGPU 2.8), Float32, CTOperators at
4d3b472 (benchmark adapters uncommitted at the time), RadonKA 0.6.3; standard suite, minimum time,
2026-11-01. The CPU columns (24 threads) were measured in the same session. The phantom errors on
the GPU equal those on the CPU.

| problem | library | forward, GPU | back, GPU | forward, CPU | back, CPU | speedup forward / back |
|---|---|---|---|---|---|---|
| cone128 | CTOperators (Siddon) | 5.6 ms | 13.0 ms | 77 ms | 160 ms | 14× / 12× |
| cone128 | CTOperators (Joseph) | 7.2 ms | 29.3 ms | 138 ms | 336 ms | 19× / 11× |
| parallel128 | CTOperators (Siddon) | 5.6 ms | 10.4 ms | 57 ms | 134 ms | 10× / 13× |
| parallel128 | CTOperators (Joseph) | 7.2 ms | 27.9 ms | 105 ms | 310 ms | 15× / 11× |
| parallel128 | RadonKA | 20.3 ms | 30.4 ms | 180 ms | 186 ms | 9× / 6× |
| cone256 | CTOperators (Siddon) | 147 ms | 247 ms | 1.21 s | 2.48 s | 8× / 10× |
| cone256 | CTOperators (Joseph) | 231 ms | 543 ms | 2.26 s | 5.15 s | 10× / 9× |
| helix256 | CTOperators (Siddon) | 61 ms | 112 ms | 0.65 s | 1.29 s | 11× / 11× |
| helix256 | CTOperators (Joseph) | 104 ms | 272 ms | 1.30 s | 2.82 s | 12× / 10× |
| parallel256 | CTOperators (Siddon) | 166 ms | 218 ms | 0.89 s | 2.05 s | 5× / 9× |
| parallel256 | CTOperators (Joseph) | 224 ms | 517 ms | 1.74 s | 4.85 s | 8× / 9× |
| parallel256 | RadonKA | 91 ms | 252 ms | 2.96 s | 3.02 s | 33× / 12× |

cone256: 256³ volume, cone beam with a 256² detector and 360 views (23.6 M rays); helix256: 256³
volume, helical cone beam with 540 views over three turns; parallel256: 256³ volume, parallel beam
with a 256² detector and 360 views.

Observations (starting points for tuning, not yet investigated):

- From 128³ to 256³ the work grows 16× (8× the rays, each twice as long), our GPU times 26–32×. A
  128³ Float32 image (8 MB) fits into the GPU's 80 MB last-level cache, a 256³ image (64 MB) hardly;
  the order in which rays reach the cache (which rays run in one workgroup) then matters.
- RadonKA, slice by slice, is faster than our parallel-beam forward projection at 256³ (91 ms
  against 166 ms), slower at 128³ (20 ms against 5.6 ms).
- Joseph's back projection takes 2–4× its forward projection (four atomic additions per sample),
  Siddon's about 2× (one per voxel).

