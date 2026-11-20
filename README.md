# ELMM adapter for the Fortran-to-CUDA compiler

This directory builds a GPU-resident ELMM from pinned upstream sources. The
compiler does the general work: it extracts the numerical loop nests, compiles
them to CUDA entries, splices the entry calls back into the sources, and
supplies the runtime and the numerical build options. This adapter holds what
is specific to ELMM: which sources and procedures take part, where the resident
epoch starts and ends, the pressure solver connection, the wall model, output
and time-step integration, and the build of the application.

Generated sources, builds and results are Git-ignored. Dependencies and input
paths are explicit; the adapter never looks for a sibling checkout and never
fetches anything.

## Resident build

```sh
python /path/to/elmm-pipeline/pipeline.py resident-build \
  --elmm /path/to/elmm --poisfft /path/to/PoisFFT \
  --compiler /path/to/compiler-repository --solver /path/to/poisson-solver \
  --vkfft /path/to/pinned/VkFFT --output /path/to/new-build \
  --fortran /usr/bin/gfortran-15 --cxx /usr/bin/g++-14 \
  --nvcc /usr/local/cuda/bin/nvcc --architecture 86 --jobs 2
```

The command needs CMake 4+, Ninja and the compiler's Python dependencies
(`requirements-local.txt`). The ELMM and PoisFFT revisions are pinned in
`dependencies.json`. VkFFT must match `pressure-dependencies.json`, and the
solver must descend from the revision recorded there. Local solver edits are
copied and hashed, then compiled out of tree with the same Fortran compiler.
The supplied source trees are not modified.

The output directory must be new. It receives:

- `build/ELMM` and `build/ELMM-native`, the resident and the native executable;
  besides ELMM's own timings both print the wall clock time of initialization,
  of the time-stepping loop and of finalization;
- `build/ELMM-full20` and `build/ELMM-native-full20`, which also dump complete
  fields for comparison;
- `prepared.json`, `packages.json` and `built.json`, the records later steps
  read, and the build commands and logs.

`--prepare-only` stops after source generation, which needs only GNU Fortran
and Python. Repeat the command with `--build-prepared` and the toolchain
options to build that prepared source. `--precision sp` builds single
precision.

A successful build says nothing about GPU execution or performance. The
container runner (`containers/resident_comparison.py` in the parent repository)
runs both executables, checks that the resident one executed its compiled
entries on the device, and compares outputs and complete fields.

### What is supported

One configuration runs resident: GNU Fortran, OpenMP, one CUDA device, a
uniform grid, Sigma subgrid model, explicit second-order diffusion, and
periodic/no-slip channel boundaries. The check is made at run time; any other
configuration takes the original native branches of the same executable.

Native and resident builds compile the same unchanged ELMM sources. The sums
ELMM marks with an OpenMP reduction (the pressure reference, the mean friction
velocity) run on the device in a fixed order of their own, so they differ from
a native run in rounding, as native runs with different thread counts do. No
aggressive floating-point options are enabled.

### Output and the solver cache

VTK output uses two device/pinned snapshot pairs and a serial writer, so a
frame is captured on the device and written while the simulation continues.
Device snapshots count against the resident workspace budget; pinned snapshots
have a 256 MiB cap. Unsupported frame types use coherent native access after
queued frames have been written.

The solver can keep compiled kernels in a persistent cache. Set
`POISSON_SOLVER_KERNEL_CACHE_DIR` to an absolute owner-only directory to use
it, and record that choice when comparing timings. See the
[solver cache contract](../poisson-solver/README.md#installed-device-api).

## How the adapter is organised

`resident_build.py` describes ELMM to `compiler.resident`: the source units
(read from ELMM's `SConstruct`), six selections of procedures with their
coverage floors, the loop rewrites, the domain object and the build. The
boundary selection also declares the boundary types of the supported
configuration (periodic in x and y, no-slip in z), which lets the compiler
make the six faces of a boundary application one entry, and has the three
calls that copy corners and edges replaced by their statements, so that the
copies of the three velocity components share launches. The diagnostics
selection holds what a step reads of the flow without changing it: the CFL
loop and the mean friction velocities, which the compiler turns into device
loops that bring back only their results. The compiler prepares
the sources from that description. ELMM's own residency
code is added in one hook, by these modules:

- `gpu_hot_path.py`: the device Poisson solver in place of the host one and its
  scratch arrays. The halo of the solution, the pressure reference and the
  pressure halo are entries compiled from `Bound_Phi`, `FixPressureToReference`
  and `Bound_Pr`;
- `resident_rk.py`: the epoch around the Runge-Kutta stages, the run-time
  check of the supported configuration, host access for the remaining native
  code, and the one change to ELMM's time step: the velocity boundaries are
  not applied a second time after `IBMomentum` when there are no
  immersed-boundary points and nothing changed in between;
- `resident_time_loop.py` and `resident_output.py`: the epoch across time
  steps, the averaging kernel, snapshot output;
- `resident_walls.py`: the stress kernels of the flat-wall model, one loop over each list of
  wall points. The friction velocity and fluxes stay fields of ELMM's own point lists, and
  the wall corrections in the momentum diffusion are compiled from ELMM's own loops. The mean
  friction velocity of the U and V points gets a procedure that names the four lists of a
  flat-wall geometry, because ELMM's own loop reaches them through pointers;
- `resident_workspace.py`: storage for the RK constants and the filter
  snapshot that outlives one call.

`gpu_pressure_sources/` holds the hand-written Fortran and CUDA of the domain
object and the solver connection.

### Host accesses

The compiler audits the selected procedures for host statements that touch an
array an entry keeps on the device. A statement that only passes whole arrays
to compiled procedures, directly or through wrappers, is not reported. In ELMM
eighteen calls remain, all from `Convection` and `PostPoisson` to procedures
that stay host code. `elmm_pipeline/host_accesses.json` lists them, each with
the `reason` it is safe, as reviewed on 2026-11-21: seven are behind conditions
the supported configuration excludes, six are wrappers once this adapter's
edit is in, and five publish the device values before the host code runs. The
review found one gap, a configuration with buoyant gases and no transported
scalar, which the supported-configuration check now excludes. A statement that is not in the
list stops `resident-build`, so a new host access cannot appear unnoticed. After checking a new one, run the build once with
`--record-host-accesses`, copy `host-accesses.json` from the output over
the recorded file and add the reason.

## Run helpers

The container runner uses a few modules from here:

- `elmm_pipeline.configuration` copies a case directory and applies explicit
  overrides (steps, grid, domain size, CFL, advection scheme, Reynolds number).
- `elmm_pipeline.launch` builds the environment of an application run.
- `elmm_pipeline.validation` checks an application log and compares VTK output.
- `elmm_pipeline.fields` reads and compares the complete-field dumps.
- `elmm_pipeline.guard` runs one command with a memory guard and cleans up its
  descendants.
- `elmm_pipeline.gpu_toolchain` finds `nvcc`, its host compiler and the device.

## Tests

```sh
python3 -m pytest -q tests
```

`tests/conftest.py` puts this checkout and a compiler checkout on the import
path: `ELMM_COMPILER_ROOT` when set, otherwise the enclosing repository when
this directory is its `elmm-pipeline` submodule. The tests need fparser and
GNU Fortran. `tests/test_gpu_pressure_preflight.py` additionally needs
`ELMM_PRESSURE_RUNTIME` and `ELMM_POISSON_SOURCE` and is skipped without them.
