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
the optional GPU target offloads eligible regions while retaining that host
configuration. MPI is outside this implementation.
Supported custom-hook switches supply the same preprocessor definitions to
discovery and compilation. Arbitrary `extra=` compiler flags and overrides of
the controlled precision, compiler family, or OpenMP mode are rejected, so
analysis cannot silently describe a different program from the native build.

### GPU execution

`--target cpu` remains the default. Select `--target gpu` to compare the original
GNU Fortran/OpenMP application with generated local CUDA and Loki/PSyclone
OpenACC variants. The host application and interface modules still compile with
GNU Fortran; `nvcc` compiles CUDA, and NVIDIA HPC SDK `nvfortran` compiles OpenACC.
The compiler boundary uses C bindings with explicit array extents, so NVIDIA
Fortran modules and descriptors do not enter the GNU host build.

GPU runs additionally require a visible NVIDIA GPU/driver, CUDA for the local
backend, and NVIDIA HPC SDK for Loki/PSyclone. Compilers are discovered on PATH
or in standard system installations, with explicit overrides available:

```sh
python3 pipeline.py run --target gpu --example channel --steps 10 \
  --threads 4 --repeats 3 --warmups 1 --fc gfortran-15 \
  --nvcc /usr/local/cuda/bin/nvcc --cuda-host g++-14 \
  --nvfortran /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/nvfortran \
  --gpu-arch 86
```

The paths and compute capability above are examples; select those for your
installation and GPU. `--cuda-host` chooses a C++ compiler supported by `nvcc`
(discovery prefers `g++-14` when installed). `--gpu-arch` defaults to the visible
GPU's compute capability. It does not bypass the GPU availability check.
`--tools` selects the same variants as CPU runs, always including the native
baseline. GPU source discovery also supports `analyze --target gpu`.


### Local runtime policies and hardware calibration

The local backend accepts `--gpu-policy always|sections|auto|chunked|hybrid`.
`always` preserves the existing forced-offload path. The other policies use the
selected compiler's public placement and transfer implementation; unsupported
cases can execute the original native region. Loki and PSyclone currently use
`always` only.

Create reusable calibration in the explicit compiler checkout, then supply it
to the pipeline. Calibration measures hardware without an application workload:

```sh
python -m compiler.offload.calibrate --output /path/to/hardware.json \
  --threads 4 --precision 64 --cuda-host-cxx /usr/bin/g++-14

# From this pipeline checkout, after setup with that compiler snapshot:
python3 pipeline.py run --tools local --target gpu --gpu-policy auto \
  --calibration-profile /path/to/hardware.json --threads 4 \
  --example channel --steps 50 --repeats 3 --warmups 1
```

The profile records launch and transfer latency, pageable and pinned transfer
bandwidth, packing costs, CPU/GPU compute and memory throughput, and CPU worker
throughput with one thread reserved for a GPU pump. Hardware, toolchain,
precision, and thread-budget compatibility are checked by the compiler/runtime.
No application profile or online tuning determines placement. Runtime decisions
use the compiler's model and current dimensions. Missing or incompatible
calibration can keep cost-driven execution native rather than inventing rates.

The pipeline copies supplied calibration into each local build and result
artifact directory. Its content hash, selected policy, and host thread budget
are part of build identity. Original collective OpenMP regions synchronize
before allocation/placement queries. Compiler collective entries reuse the
existing team; a native decision executes the original worksharing region.
Allocation checks precede descriptor arguments and policy queries.

Validation reports native, GPU, mixed, hybrid, and chunked decisions separately
from successful CUDA launches. An intentional all-native policy run remains a
valid numerically checked result and makes no GPU execution claim.

### Opt-in shared source scopes

`--memory-model call` remains the default. The local CUDA backend additionally
accepts `--memory-model scoped` with `sections` or `auto`. Supply an entry, the
original entry source, its available source closure, and explicit capture facts:

```sh
python3 pipeline.py run --tools local --target gpu --gpu-policy sections \
  --memory-model scoped --scope-entry module_name::advance \
  --scope-source advance.f90 --scope-source-file numerical.f90 \
  --scope-source-file shared_arrays.f90 --scope-facts /path/to/captures.json \
  --config /path/to/case --threads 4 --repeats 3 --warmups 1
```

Source paths are relative to the selected application's `src/` tree. Repeat
`--scope-source-file` for the original modules available to the compiler, including
imports needed for capture and effect analysis. Call-region selection filters
(`--select`, `--exclude`, `--marked-only`) do not apply to this mode. The entry
names and source paths identify available input; they supply no grouping recipe.

`--scope-execution reached` opts into the compiler's experimental reached-owner
implementation; `bounded` remains the default. The adapter forwards the option,
includes it in build identity, and applies the returned public artifacts. The
compiler decides which original child regions, lexical coordinators and native
operations can share a context. Consult its manifest for reached boundaries and
available numerical work; selecting this mode is not evidence of a resident
application chain or improved performance.

Capture facts use the compiler's versioned public JSON contract. They declare
participation and canonical resource roots with storage lifetime, escape,
allocation-change, and physical initialization facts. Supply `whole`, `none`, or
explicit initialized sections according to the actual source contract; array
`INTENT` alone does not establish full initialization. The pipeline does not
invent these facts. Optional `sources` hashes may use paths relative to `src/`,
or absolute paths inside the cached original `src/` tree; hashes remain exact
when paths are rebound to an isolated build copy. Optional
`--scope-effect-contracts` passes a public compiler effect-contract file.

Collective source companions additionally require explicit capture schema 2:
`participation.kind=omp_full_team`, `dispatch=qualified_companion`, a qualified
entry, level 1, the configured host-thread budget, and unconditional original
call-site/team spans with their hashes. Shared array descriptor and immutable
control assertions belong to the supplied capture facts. The pipeline then
passes `--gpu-collective`; the compiler proves which callers and native helpers
can participate and returns public participation schema 2 inside the existing
scope manifest. Source paths in call sites and optional `native_participation`
assertions move with build copies; original spans, hashes and semantic facts
stay intact. No caller proof or native execution role is inferred by the adapter.

The selected parser environment extracts configured sources and capture-safe
numerical leaves. The compiler owns grouping, effects, numerical eligibility,
transfers and placement. Approved replacements are hash-checked before any
application source is edited. Public build roles supply interfaces, kernels and
one common runtime to the ordinary application build. Capture facts, source
selection, extraction implementation, policy, calibration and thread budget
participate in build identity. No parent checkout, compiler IR or generated CUDA
inspection is used.

Independent compiler-approved roots can share one application build through
`elmm_pipeline.local_adapter.compose_scope_packages(packages, destinations)`.
Prepare each package with `transform_scopes` against pristine sources and use a
separate artifact directory for each root. `destinations` maps original source
identities to the application copy. The helper validates every package before
applying replacements, rejects conflicting replacements of the same file, and
links exactly one matching common runtime. Shared build artifacts deduplicate
only when their public path, role, language and hash agree. Every colliding
supporting artifact path, including helper headers, must have identical bytes;
original-source replacements compose by destination. Incompatible runtime
identities or artifacts are errors. It does not merge Fortran bodies or choose
implementations.

```python
bundle = compose_scope_packages([first_package, second_package], destinations)
binary = build(...,
               extra_sources=[Path(path) for path in bundle["extra_sources"]],
               patched_sources=[Path(path) for path in bundle["patched_sources"]],
               gpu_sources=bundle["gpu_sources"])
```

Keep `bundle["scope_packages"]` in the build receipt: the original compiler
manifests remain authoritative separately. Retain each artifact directory for
relative includes. The CLI continues to accept one scope entry per invocation;
batch composition is an explicit Python API. Preparing or applying independent
packages sequentially to already edited sources is unsupported.
Pass these packages to `gpu_activity(..., scope_packages=bundle["scope_packages"])`
for combined diagnostic validation. Campaign adapters transport the same field.
Approved entry identities are checked across the packages; an all-native
contract applies only when every package establishes it. GPU activity and
transfer counters still come from the separate runtime trace.

Scoped runtime traces report successful launches, upload/download bytes,
allocations and waits during separate validation runs. Adapter call markers are
not available for source scopes, so transformed-call coverage is explicitly
unverified. A zero-scope result still builds and validates the native application.
The current compiler's `auto` source-scope mode selects the original native span
when coherent estimates are unavailable; this is a compiled manifest decision,
not an invented runtime decision count. Complete runs still compare final fields
against the unmodified baseline and include required CPU boundaries and exports.

The opt-in native integration check builds a small generic
producer→native helper→consumer application from independent compiler/pipeline
copies using the public adaptation and build APIs. It checks every field after
shape changes and reallocation, expected section traffic, and a valid zero-CUDA
automatic decision:

```sh
ELMM_SCOPED_NATIVE_TESTS=1 \
ELMM_TEST_COMPILER_ROOT=/explicit/path/to/compiler-checkout \
ELMM_TEST_SCONS_PYTHON=/explicit/path/to/scons-environment/bin/python \
python -m pytest -q tests/test_scoped_adaptation.py -k real_normal_scopes
```

This check requires CUDA with architecture 86, `g++-14` and `gfortran-15` on
`PATH`; builds run sequentially with one job and a four-thread host budget.

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

# Broader local-GPU candidates, checked by the explicitly selected compiler:
python3 pipeline.py analyze --source-tree /path/to/fortran/sources --target gpu \
  --local-compiler /path/to/compiler-checkout \
  --python "$PWD/.cache/envs/local/bin/python" --output /path/to/new-gpu-analysis
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

For the local GPU target, discovery instead checks broader normalized candidates
through the selected compiler's public `--list-candidates --json` interface.
The compiler owns numerical legality, per-kernel independence, fusion, and the
ordered execution/memory plan. Candidate checks participate in grouping, so a
rejected group can still leave eligible smaller regions. The pipeline retains
source-span, capture-lifetime, native OpenMP dispatch, and interface checks.
Loki and PSyclone retain the conservative discovery path, which does not require
the local compiler.

Local GPU discovery also proposes adjacent direct calls to leaf module
subroutines from a serial caller. Helpers remain calls in the normalized source;
the compiler inlines them and proves the combined computation. Whole-variable
arguments, constant assumed-shape lower bounds, visible shared captures, and
supported combined OpenMP loops are required. Scalar preparation stays inside
each helper. Existing strip-mining proofs can lift helper loops, with stride
tests evaluated after their original preparation; negative tile strides retain
empty point ranges. Opaque or recursive calls, inaccessible captures, scalar
outputs, uncertain aliases, and synchronization boundaries stay native. The
rules do not inspect application or routine names.

These larger entries use the existing whole-entry CUDA allocation scope. The
section, automatic, and chunk policies currently require a flat device plan;
helper scalar setup or host branches can therefore make them select native
execution for the group. Independently outlined callee loops may still offload
separately. The compiler's public `offload.analysis` explains this limitation.
Discovery acceptance alone does not establish execution on the GPU or a speedup.

This separation admits ordered dependent kernels in one data region, conditional
bodies and imperfect nests supported by the compiler, and stable allocated host
array captures. Bounds and branches inside a candidate retain their original
evaluation positions. Allocation status is checked before generated descriptors
or capture bounds are evaluated; an unavailable capture uses the original native
code, preserving valid empty-loop paths. Allocation/deallocation inside regions
and local scratch arrays remain outside this contract.

An eligible region becomes a standalone module and a call at its original source
location. Native initialization, calls, boundary handling, tiling, and surrounding
OpenMP worksharing stay in place for the CPU target. CPU outlining preserves
outer tiling; it does not discard arbitrary tile-setup calls or assume tile widths are positive.
Loops required by existing worksharing/collapse directives are preserved.

| Backend | CPU target | GPU target |
|---|---|---|
| baseline | Original application source. | Same original GNU Fortran/OpenMP application. |
| Loki | Eligible helper inlining, invariant scalar motion, and checked loop fusion using Loki's IR/transforms. | OpenACC parallel loops and an enclosing data region generated through Loki's IR. |
| PSyclone | Eligible helper inlining, scalar motion, and loop fusion with dependence checks enabled. | OpenACC parallel loops and an enclosing data region generated through PSyclone's IR. |
| local | General compiler lowering, dependence checks, optimization, and C++ generation with a Fortran interface. | CUDA kernels and a shared per-entry device allocation plan. |

GPU discovery can lift a proved strip-mined loop nest into its complete logical
iteration domain and replace compatible OpenMP worksharing with one collective
dispatch. It proves tile bounds, index use, storage access, and dispatch safety;
unsupported tiled or parallel contexts remain native with diagnostics. It does
not fall back to launching a GPU kernel for every CPU tile. Optional markers
restrict selection but supply no routine-specific transformation recipes.

Device data live for **one normalized entry invocation**. All kernels inside
that entry share its allocations, including a lifted tile domain. Written
arrays are copied in and out to preserve unchanged halos. Data are not retained
between separate entries, calls, or timesteps. Transfers, allocation overhead,
and remaining native work therefore remain part of complete application timing.

Adjacent eligible GPU loop nests can share one entry and data region. This
uploads their common input arrays once, while retaining the original kernel
order and each loop's own bounds. Grouping requires the same control path and
compatible OpenMP dispatch and capture proofs. Calls, intervening executable
statements, conditionals, and explicit barriers end a group. Unsupported OpenMP
contexts remain native. If combining eligible loops fails the existing
dependence checks, they remain separate.
Reports count both outlined regions and their constituent loop nests.

Loop nests within a local GPU group need not be pointwise independent of each
other: each generated kernel must be safe to parallelize, and the compiler
preserves dependencies by keeping kernels ordered when fusion is unsafe. One invocation shares device
storage across those kernels. The original source remains the fallback for
unsupported dynamic calling contexts.

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
reductions, scratch arrays, control flow unsupported by the selected backend, and PURE or
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
Application timings include the transformed variants' marker guards and first-call
output; the baseline has no coverage instrumentation.

`--no-coverage` disables inserted markers; the report explicitly labels execution
coverage unverified. This control is separate from optional `!kernel` hints.

GPU runs first execute a separate, numerically checked `gpu-validation` case.
Local CUDA uses `FORT_RUNTIME_TRACE=1`; OpenACC uses
`NVCOMPILER_ACC_NOTIFY=31` plus an in-device NVIDIA check. Reports distinguish
computation launches from OpenACC proof kernels. An offload variant needs actual
computation on the GPU to claim GPU execution; intentional native policy decisions
are reported separately and remain valid. Compilation or a host entry
marker alone is insufficient, including with `--no-coverage`.

`--gpu-profile nsys` is the default and profiles only that validation case with
Nsight Systems. The retained `.nsys-rep`, SQLite export, and profile logs support
kernel duration/count and transfer duration/byte totals from CUDA device
activities. Totals sum activity durations; overlapping work need not equal wall
time. Profiler wall time includes collection/export overhead and is excluded
from benchmark samples. Missing profiler support or activity records are
reported as unavailable, not zero cost. `--gpu-profile none` disables Nsight
collection while preserving GPU execution proof and numerical validation.
Verbose GPU tracing, profiling, and forced synchronization overrides are removed
from ordinary warmups and timed repetitions.

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

The compiler and this pipeline can live in separate repositories. The pipeline
never imports compiler implementation modules: it invokes the explicitly chosen
snapshot through `python -m compiler` and consumes its JSON decisions and emitted
sources. The compiler runs independently on standalone entries and has no
dependency on this pipeline, ELMM, Loki, or PSyclone. Normal generation with
`--json` reports the execution/memory plans and actual parallel-region count;
the adapter uses that report instead of inferring kernel counts from CUDA text.

Pinned repository URLs/revisions are in `dependencies.json`. Private Python
requirements pin the tool environments, including older versions required by
Loki and PSyclone. After changing a repository pin, repeat setup; it fetches and
checks out the requested revision only in a clean cache and preserves local
edits. Builds are cached by source, compiler,
backend, selection, and adapter identity. Upstream sources retain their licenses.

```sh
python3 -m pytest -q tests
```

Parser tests require fparser in the test interpreter; backend integration tests
use explicit `ELMM_LOKI_PYTHON` and `ELMM_PSYCLONE_PYTHON` paths. End-to-end generic
adaptation tests additionally use `ELMM_TEST_COMPILER` and `ELMM_TEST_PYTHON`.
Tests exercise unrelated unannotated programs, renamed symbols, varying ranks
and lower bounds, preserved native context, actual numerical equivalence, and
rejection of unsafe transformations.

`ELMM_GPU_NATIVE_TESTS=1 python3 -m unittest tests.test_gpu_build -v` additionally
builds and executes small GNU-host CUDA/OpenACC interoperability tests on a
visible GPU. GPU tooling and profiler parsers have separate tests that run
without GPU access.

## Reproducible application campaigns

`python -m elmm_pipeline.campaigns` maintains the comparison workflow. It uses
explicit JSON input paths; it neither imports compiler internals nor looks for
older result directories. Every preparation freezes selected **committed**
compiler/pipeline files, approved binaries, build receipts, capture evidence,
calibration, configurations and full-field references. Unrelated worktree edits
are preserved. The receipt records the compatible commit pair and hashes external
runtime libraries and tools. Reusing a verified binary requires its exact hash
and retained build provenance; timing samples are always fresh.

```sh
python -m pip install -r requirements-campaign.txt
python -m elmm_pipeline.campaigns prepare --config /path/to/comparison.json --output /path/to/new-campaign
python -m elmm_pipeline.campaigns run --campaign /path/to/new-campaign verification
python -m elmm_pipeline.campaigns run --campaign /path/to/new-campaign halos
python -m elmm_pipeline.campaigns run --campaign /path/to/new-campaign timings
# Run each additional explicitly configured phase, e.g. the six-policy suite.
python -m elmm_pipeline.campaigns run --campaign /path/to/new-campaign ordinary
python -m elmm_pipeline.campaigns plot --campaign /path/to/new-campaign
python -m elmm_pipeline.campaigns seal --campaign /path/to/new-campaign --output /path/to/new-snapshot
```

Configuration schema 1 requires `threads: 4`, `runtime_id`, `profile`, `title`,
`previous_snapshot`, `repositories` (explicit paths and selected tracked files),
`tools` (executable/version-query argument lists), `modes` and `cases`. Each mode
supplies a unique `name`, `label`, approved `binary`/`sha256`, and optionally its
`adapter` and `build_receipt`. `baseline_modes` identifies the frozen applicable
controls; `candidate_mode` identifies one policy shared by every case. Cases
supply copied configuration directories, complete VTK references, original
runtime arguments and expected timestep counts. `halo_drivers` supplies explicit
application validation sources and linker inputs, complete field counts and
unchanged tolerances. Their application-specific calls do not affect compiler
eligibility. `external_files` binds shared libraries, objects and public build
artifacts to their reviewed hashes.

An application validation driver can set `application_case` to a configured case
name. It then runs its diagnostic main with that case's original inputs and
step/VTK checks; these runs never contribute timing samples. The explicit linker
inputs retain the original numerical objects. Set `metadata_format` to
`storage-shape-bounds-v1` to require a `fields-NAME.shape` companion to each
binary dump: element bits, extents, lower bounds and upper bounds, in that order.
Validation checks precision, exact original bounds, file size and every element,
including halos and empty fields. Drivers without this option retain their
existing fixed-layout dump format.

Optional `commands` configure additional maintained tools as argument lists,
working directories, explicit environment and expected report paths. The only
substitutions are `{campaign}` and `{python}`; commands never pass through a
shell. Toolchain and architecture choices belong in this configuration, so the
runner does not assume a GPU generation or CUDA installation directory.

Execution uses the committed pipeline snapshot. The memory guard serializes
heavy jobs, requires 3 GiB available memory and tracks detached descendants
through cleanup. Reports retain incomplete or failed attempts. Each phase is
immutable; prepare a new campaign after changing inputs. Timing includes process
startup, native work, transfers, output and teardown, with one warmup and three
measurements (seven near a gate, or `repetitions: 7`). Diagnostics run separately;
planning construction, validation and selection have separate counters. Sealing
requires successful guards, complete fields, halo checks, configured reports
and current plots. A sealed snapshot is evidence, not automatic promotion.

`python -m elmm_pipeline.campaigns.profile --campaign ... --case ... --mode ...
--nsys /explicit/nsys --seconds 2 --output /new/diagnostic` runs a bounded
CUDA/NVTX capture and validates the complete application output. Run it through
`elmm_pipeline.campaigns.guard`, after timing jobs finish. The capture can last
1–30 seconds; the application continues to completion after collection ends.
CPU sampling and synchronizing phase timers are disabled. Recorded device
durations remain separate from application wall time; a short trace is never
extrapolated into a full-run speedup or overlap claim.
