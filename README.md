# Structural Fortran transformation and ELMM benchmarks

This portable directory contains a generic region discovery and transformation
pipeline, plus an ELMM build/run adapter. Optimization decisions use program
structure, declarations, and dependence checks. There are no application routine
names, prescribed helper-call sequences, component names, or fixed array bounds
in the transformation policy. Source annotations are optional.

Downloaded repositories, compiler snapshots, Python environments, generated
sources, builds, and results are Git-ignored. Project code never discovers a
parent/sibling checkout. Dependencies and input paths are explicit.

## Setup and run

Requirements: Linux, Python 3.11+ with venv/pip, Git, GNU Fortran, GCC/G++, FFTW
(single/double precision and OpenMP libraries), BLAS and LAPACK. Bootstrap detects
native libraries and creates private linker aliases; it does not install system
packages. Initial provisioning needs network access.

GNU Fortran 15 works with the pinned ELMM revision. GNU Fortran 16 rejects an
upstream pure geometry constructor, so select a compatible compiler explicitly.

```sh
cd elmm-pipeline
python3 pipeline.py setup --fc gfortran-15 \
  --local-compiler /explicit/path/to/compiler-checkout

# Original channel configuration; scan the complete application source tree.
python3 pipeline.py run --example channel --steps 10 --threads 4 \
  --repeats 3 --warmups 1 --fc gfortran-15

# Any configuration directory, including paths containing spaces.
python3 /path/to/elmm-pipeline/pipeline.py run \
  --config '/path/to/my case' --steps 20 --fc gfortran-15
```

The compiler checkout must contain `compiler/__main__.py`. Setup snapshots it in
`.cache/repos/local-compiler`, including its content hash and Git dirty state.
Subsequent runs use that snapshot. Repeat setup with the explicit source path to
update it. `--tools baseline,loki,psyclone` omits this dependency; the same option
must be used at setup and run. Every comparison includes the native baseline.

The complete configuration directory and relative assets are copied. Each run
starts in a fresh copy and forces `output_dir='output/'`. Implicit `ELMM_*`
environment overrides are removed. Configurations must enable final velocity
and pressure in `output/out.vtk`; missing validation fields fail the run.

Omitting `--steps` preserves the original termination settings. `--steps N`
explicitly caps timesteps; it does not represent the full configured physical
simulation. `--advection N` remains an explicit optional configuration override.
Advection method and grid settings do **not** decide transformation eligibility.

For compiled custom geometry/hooks, use repeated `--build-arg name=value` options,
for example `--build-arg customSB=yes --build-arg SB=/path/to/custom.f90`.
Configurations using CGAL also need its native dependencies and upstream build
option. The ELMM build adapter currently controls double precision and CPU/OpenMP;
MPI and GPU offload are outside this implementation.
Supported custom-hook switches supply the same preprocessor definitions to
discovery and compilation. Arbitrary `extra=` compiler flags and overrides of
the controlled precision, compiler family, or OpenMP mode are rejected, so
analysis cannot silently describe a different program from the native build.

## Discovery and selection

The default run scans all free-form `.f90` files below ELMM's `src/`. It records
eligible regions and reasons for rejected candidates. It does not require a
particular source filename, routine name, advection method, rank, or annotation.

Optional hints restrict that search without supplying an optimization recipe:

```sh
python3 pipeline.py run --config /path/to/case --steps 10 --fc gfortran-15 \
  --source-pattern 'physics/*.f90' --select '*::advance*' --exclude '*diagnostic*'

# Explicit opt-in selection when source procedures carry !kernel comments:
python3 pipeline.py run --config /path/to/case --marked-only --fc gfortran-15
```

To inspect an unrelated source tree without building ELMM:

```sh
python3 pipeline.py analyze --source-tree /path/to/fortran/sources \
  --python "$PWD/.cache/envs/local/bin/python" --output /path/to/new-analysis
```

`--source-pattern` is repeatable; `--define SYMBOL` selects active preprocessing
symbols for analysis. The parser interpreter is explicit and needs fparser.
Use `--fc` to choose the GNU preprocessor; analysis uses `-O3 -fopenmp`, matching
the application build.
Discovery resolves module imports and supported kind constants from the selected
source tree, uses compiler preprocessing with source mapping, and makes no input
edits. The analysis manifest contains generated module inputs, source spans,
captures, legality information, and rejection diagnostics.

The local compiler also accepts unannotated entries and same-module helpers:

```sh
python -m compiler --input /path/to/module.f90 --kernel module_name::entry \
  --output-dir /path/to/generated
python -m compiler --input /path/to/module.f90 --list-candidates --json
```

Those commands run from the explicitly chosen compiler checkout/environment.
`--require-markers` retains strict annotated selection. The pipeline supplies
selected generated entries automatically; users need not name kernels. Unknown
precision aliases are rejected rather than inferred from their spelling.

## Transformation contract

Discovery currently accepts rectangular unit-stride loop regions with proved
injective pointwise writes, affine array accesses, supported explicit scalar
captures, and temporary values whose lifetimes fit inside the region. It can
group adjacent eligible loop nests. Each array is normalized using its own lower
bounds while induction-variable values remain unchanged, so indices used as data
keep their meaning.

An eligible region becomes a standalone module and a call at its original source
location. Native initialization, calls, boundary handling, tiling, and surrounding
OpenMP worksharing stay in place. This implementation preserves outer tiling; it
does not discard arbitrary tile-setup calls or assume tile widths are positive.
Loops required by existing worksharing/collapse directives are preserved.

| Backend | Generic processing |
|---|---|
| baseline | Original application source. |
| Loki | Eligible helper inlining, invariant scalar motion, and checked loop fusion using Loki's IR/transforms. |
| PSyclone | Eligible helper inlining, scalar motion, and loop fusion with dependence checks enabled. |
| local | General compiler lowering, dependence checks, optimization, and C++ generation with a Fortran interface. |

Backends report actual pass activity. Parsing/printing a module is not counted as
optimization. A single outlined loop can require no additional Loki/PSyclone
transformation; the report distinguishes discovery, outlining, backend changes,
and executed regions. Generated C++ is serial within a preserved native parallel
context to avoid creating a team per tile.
Inside such contexts, the local backend also requires proved contiguous array
captures: its interface must not create per-thread array temporaries whose
copyback could overwrite another thread's updates.

Unsupported cases remain native with diagnostics. Examples include uncertain
aliasing/storage association, unknown effects, unsupported precision, scalar
reductions, scratch arrays, nonrectangular/control-flow regions, and PURE or
ELEMENTAL procedures lacking a suitable generated-call contract. Optional
captures, TARGET aliases, persistent local state, and alternate ENTRY procedures
also remain native, as do procedures with NAMELIST or contained procedures whose
implicit local-variable reads need further analysis. The compiler
and structural outliner have different capability boundaries; backend rejection
is recorded separately. This is conservative automation, not whole-language
Fortran compilation or a guarantee of optimal performance.

## Results, execution evidence, and validation

Results go to `results/<UTC timestamp>/`, or a new `--output PATH`. They include:

- `report.md` and `report.json`: timings, per-run numerical checks, tool versions,
  compiler/build settings, source hashes, selection, and coverage.
- `config/`: effective input snapshot and its recorded overrides.
- `sources/<variant>/native/`: changed application files; generated modules,
  transformation/discovery manifests, and build logs accompany them.
- `runs/<variant>/<repeat>/`: fresh case copy, full output, and run log.

Automatic execution markers identify discovered regions by generated IDs. They
are inserted into staged source and need no user annotations. Thread-local
marker state avoids races inside native OpenMP loops. Regions not exercised by a
configuration are reported separately. A variant that executes no outlined
region is labeled `not_exercised`, not a successful optimization measurement.
These timings include the transformed variants' marker guards and first-call
output; the baseline has no coverage instrumentation.

`--no-coverage` disables inserted markers; the report explicitly labels execution
coverage unverified. This control is separate from optional `!kernel` hints.

All variants use the same double-precision native build flags and thread count;
OpenMP runtime scheduling is static and BLAS threads are one. Compilation finishes
before timed execution. Run order rotates between repetitions. External wall
time includes startup/final I/O; ELMM's internal timestep timer includes periodic
output and has 0.01-second resolution. Short capped runs establish operation;
longer repeated measurements are needed for performance conclusions.

Every warmup and measured repetition validates finite final VTK coordinates and
fields, dimensions, field names, and saved values against the first native run:
`abs(actual-reference) <= atol + rtol*abs(reference)`. Defaults are `atol=1e-6` and
`rtol=1e-5`. ELMM writes VTK values as float32 even in double-precision builds;
this does not establish double-precision bitwise equivalence. Direct native
integration tests also compare unrelated transformed kernels and preserved halos.
Every repetition must complete the same number of timesteps as the first baseline.

ELMM can return status zero after Fortran STOP. The pipeline checks failure
markers, positive timesteps, the final save confirmation, expected completion,
and required output fields. Build failures, timeouts, or numerical mismatches
fail the run and retain diagnostics.

## Portability and tests

Copy this directory's non-ignored files into a new repository, then run setup at
the destination. Disposable environments, binaries, and linker aliases should be
recreated. Supply the compiler source explicitly when provisioning a clean copy.
Paths in old reports are provenance, not implicit lookup locations.

Pinned repository URLs/revisions are in `dependencies.json`. Private Python
requirements pin the tool environments. Builds are cached by source, compiler,
backend, selection, and adapter identity. Upstream sources retain their licenses.

```sh
python3 -m unittest discover -s tests -v
```

Parser tests require fparser in the test interpreter; backend integration tests
use explicit `ELMM_LOKI_PYTHON` and `ELMM_PSYCLONE_PYTHON` paths. End-to-end generic
adaptation tests additionally use `ELMM_TEST_COMPILER` and `ELMM_TEST_PYTHON`.
Tests exercise unrelated unannotated programs, renamed symbols, varying ranks
and lower bounds, preserved native context, actual numerical equivalence, and
rejection of unsafe transformations.
