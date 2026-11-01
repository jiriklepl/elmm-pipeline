# Programmer Documentation

This document explains the structure of `poisson-solver`, its main components and how data moves through a solve.
It covers both the local build and the distributed build.
For API documentation, see the generated Doxygen output.
The [examples README](../examples/README.md) explains how to generate it.

## Contents

1. [Overview and repository map](#1-overview-and-repository-map)
2. [The public API](#2-the-public-api)
3. [Plan / Core / Policies split](#3-plan--core--policies-split)
4. [Boundary-condition dispatch table](#4-boundary-condition-dispatch-table)
5. [Data-flow traces](#5-data-flow-traces)
6. [Distributed decomposition and heFFTe communication](#6-distributed-decomposition-and-heffte-communication)
7. [Bindings](#7-bindings)
8. [Build system](#8-build-system)
9. [Tests](#9-tests)
10. [Repository quirk: the heFFTe fork](#10-repository-quirk-the-heffte-fork)

---

## 1. Overview and repository map

`poisson-solver` solves `Δφ = b` on 1D, 2D, and 3D grids with FFT-based methods.
It follows the design of [`PoisFFT`](https://github.com/LadaF/PoisFFT).
See [motivation](./motivation.md) for background on the design.

The library splits every solve into three pieces.
A **plan** decides *what* to do, based on the requested boundary conditions.
A **core** owns the buffers and handles that the plan needs.
A set of **policies** decides *how* FFTs and inter-rank redistribution happen.
You cannot see this split if you only read the public `include/poisson-solver/solver.hpp` header.
The rest of this document makes the split visible.

Two conventions apply everywhere in the codebase.

- **Grid layout.** The first index is the fastest-varying one.
  The library always orders boundary conditions (BCs) as `{{west, east}, {south, north}, {bottom, top}}`, that is, per axis, in X, Y, Z order.
- **Supported types.** The solver works only with `float` and `double` element types.
  Grid dimensionality `Dim` is 1, 2, or 3.

### Repository map

| Path | Contents |
|---|---|
| `include/poisson-solver/` | Public headers: `solver.hpp`, `boundary_conditions.hpp`, `aliases.hpp`, `approximations.hpp`, `mpi_utils.hpp`, `c_binding.h` |
| `include/poisson-solver/detail/` | Internal headers: `solver_plan.hpp`, `solver_core.hpp`, `axis_strategies.hpp`, `fft_policies.hpp`, `reshaper.hpp`, `helpers.hpp`, `cuch.hpp`, `mpi_check.hpp`, `c_binding_for_fortran.h` |
| `include/poisson-solver/detail/kernels/*.cuh` | `__global__` CUDA kernels |
| `include/poisson-solver/detail/runners/*.hpp` | Host-callable launch-wrapper declarations for those kernels |
| `src/detail/runners/*.cu` | Launch-wrapper definitions + explicit template instantiations |
| `src/` | `c_binding.cpp`, `detail/c_binding_for_fortran.cpp`, `fortran_binding.f90`, `mpi_utils.cpp` |
| `tests/` | doctest-based C++/C tests, Fortran tests, benchmark executables |
| `benchmark-runners/` | Shell scripts that launch the benchmark executables on the cluster |

The kernel/runner split keeps CUDA `__global__` syntax out of the public header (`solver.hpp`).
Kernels live in `.cuh` files.
Only `.cu` translation units include these `.cuh` files.
The runners that launch the kernels use ordinary host-callable signatures.

---

## 2. The public API

This section covers the API in `include/poisson-solver/solver.hpp`.

- **`ISolver<T, Dim>`** — the abstract base class for both concrete solvers.
  Its interface has three methods: `Execute` (host buffers), `ExecuteGPU` (device buffers), and `GetTotalGridSize`.
  `ExecuteGPU` saves a host-device copy, but a device-device copy can still occur because of ghost-cell slicing.
  The C++ API has no `Finalize` method, by design.
  The object lifetime follows RAII rules instead.

- **`Solver<T, Dim>`** — the solver for a uniform grid, where `Dim` is 1, 2, or 3.
- **`NonuniformZSolver<T>`** — a 3D-only solver whose Z axis has non-uniform cell spacing.

Each concrete solver has exactly one constructor, chosen at compile time by `POISSON_SOLVER_USE_MPI`.
`POISSON_SOLVER_USE_MPI=OFF` exposes the **local** constructor, it takes grid sizes, lengths, and BCs.
`POISSON_SOLVER_USE_MPI=ON` exposes the **distributed** constructor, it additionally takes the global grid size, this rank's offset into it, and an `MPI_Comm` from `mpi_utils::InitMPIGrid` (see [Distributed decomposition and heFFTe communication](#6-distributed-decomposition-and-heffte-communication)).

`Execute` and `ExecuteGPU` take two parameters, `ngs_phi` and `ngs_rhs`.
Each gives the number of ghost cells on one side, per dimension, around the active grid in the caller's `phi` or `rhs` buffer.
Before it solves, the solver strips the ghost shell into a contiguous device workspace with `UnpackPaddedGrid`.
After it solves, `PackPaddedGrid` writes the result back into the padded shape.

Every `Execute` or `ExecuteGPU` call returns an `ExecutionTimes` struct with four fields: `copy_time_ms`, `reshape_time_ms`, `fft_time_ms`, and `kernel_time_ms`.
Two RAII scoped timers collect these values:

- `CUDAScopedTimer` — CUDA events, used around kernels and device copies.
- `HybridScopedTimer` — a device sync, then `std::chrono::steady_clock`.
  Use it around FFT and reshape calls.
  These calls can span multiple streams or execute nontrivial CPU code.

Both timers measure something only when you build the library with `POISSON_SOLVER_ENABLE_BENCHMARKS` (see [Build system](#8-build-system)).
Otherwise, `ExecutionTimes` is always all-zero.
The timer types then become zero-cost no-ops.

**Approximations.** `IApproximation<T>` (`approximations.hpp`) sets how the solver approximates the continuous Laplacian eigenvalue on the discrete grid.
Three options exist: `SpectralApproximation` (exact spectral eigenvalues), `FiniteDifference2Approximation`, and `FiniteDifference4Approximation`.
This choice affects only the per-axis eigenvalue computation in [Plan / Core / Policies Split](#3-plan--core--policies-split).
It does not change which BCs the solver supports.

---

## 3. Plan / Core / Policies split

The codebase is split mainly into these three components - plan, core and policies.
It ensures the code is readable and maintainable while the solving is done with little overhead.

1. **Plan** - what has to happen, _given the requested BCs_?
2. **Core** - what buffers, handles, and precomputed data does that imply?
3. **Policies** - what methods should be called based on the compile-time configuration?

Each `Solver` or `NonuniformZSolver` instance owns a plan and a core, as private members.

### The plan — `detail/solver_plan.hpp`

`ISolverPlan<T, FFTDim>` is the common base class.
`Solver` uses `SolverPlan<T, FFTDim>`.
`NonuniformZSolver` uses `NonuniformZSolverPlan<T>`.
A plan validates **all** BCs and chooses values for the following:

| Field                   | Type                                              | Meaning                                                                                                                     |
|-------------------------|---------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| `axis_strategies`       | array of `IAxisStrategy` per FFT-transformed axis | per-axis grid spacing / eigenvalue argument / normalization ([Axis strategies](#axis-strategies--detailaxis_strategieshpp)) |
| `fft_layout`            | `VariantLayout` (`FullLayout` or `SplitLayout`)   | whether one transform covers the whole grid, or the grid is split into a 1D + 2D stage                                      |
| `requires_cmplx_wspace` | `bool`                                            | whether the core must allocate a complex device workspace                                                                   |
| `reshape_types`         | sequence of `ReshapeType`                         | which redistributions (`ToPencils`, `ToSlabs`) the reshaper pipeline runs, and in what order                                |

`SolverPlan` also stores a `solve_method`.
This is a pointer to a member function, and it selects which `Solve*` method on `Solver` runs the solve.
The plan chooses it once, at construction, based on the BCs.
See [Boundary-condition dispatch table](#4-boundary-condition-dispatch-table) for the exact mapping from BCs to FFTs and solve methods.

### Axis strategies — `detail/axis_strategies.hpp`

`IAxisStrategy<T>` groups everything that depends on a single axis's BC pair.
This includes the grid spacing (`GetGridDx`), the eigenvalue argument for the chosen approximation function (`GetApproximationInput`), and the axis's share of the overall normalization factor (`ComputeNormalizationFactor`).


### The core — `detail/solver_core.hpp`

`SolverCore<T, Dim, FFTDim = Dim>` owns everything the plan's decisions imply:

- **FFT plan** (`fft_plan`) — the active `FFTPolicy` builds this from the plan's `fft_layout`.
- **Reshaper** (`reshaper`) — a `DummyReshaper` in the local build, or a heFFTe-backed `Reshaper` in the distributed build ([Distributed decomposition and heFFTe communication](#6-distributed-decomposition-and-heffte-communication)).
- **Ping-pong device workspaces**: `cu_real_workspace1/2` and `cu_cmplx_workspace1/2`.
  The solver alternates between the "1" buffer and the "2" buffer at each step of a `Solve*` method.
  It uses workspace 1, then reshapes or transforms into workspace 2, then uses workspace 2, and so on.
  Three booleans decide which of these get a fresh allocation, and which just alias each other.

  The booleans are `requires_cmplx_wspace`, whether the layout is full, and whether the build includes MPI.
  The comment above `AllocDeviceWorkspaces` spells this out in a table.
  In short: without MPI, workspace 2 always aliases workspace 1, because a local solve never redistributes data and needs no second buffer.
  With MPI and a split layout, workspace 2 is a separate allocation, because redistribution there is out-of-place.

- **Per-axis eigenvalue arrays** (`cu_denoms`) — one array per FFT-transformed axis.
  The solver computes these once, on the host, from the chosen `IApproximation` and axis strategy, then copies them to the device at construction time.
  `cu_denom_observers` holds raw pointers into these arrays, for the kernels to read.
- **`norm_factor`** — the scalar that divides every result at the end of a solve.
  It is the product, over all FFT-transformed axes, of that axis strategy's `ComputeNormalizationFactor`.

- **Lazily sized padded host-transfer buffers** — the staging buffers `Execute` uses for the ghost-cell-padded `phi` and `rhs`.
  The solver reallocates these only when the requested ghost-cell count changes.

### FFT policies — `detail/fft_policies.hpp`

`POISSON_SOLVER_USE_MPI` selects `FFTPolicy<T, Dim>`, a compile-time alias: `VkFFTPolicy` for the local build, `MpiFftPolicy` for the distributed build.
Both expose the same static API: `Forward`/`Backward` for `FullLayout` plans, and `Forward1D`/`Backward1D` plus `Forward2D`/`Backward2D` for `SplitLayout` plans.
Because of this, `solver.hpp` defines every `Solve*` method just once, and it compiles under either policy.

- **`FullLayout`** carries a single `FFTTransformType`.
  It covers the whole domain, up to 3D, in one transform.
- **`SplitLayout`** carries two transform types, `forward_type_1d` and `forward_type_2d`.
  It also carries an `isolated_dim_idx` (0 for X, 1 for Y, 2 for Z), which names the axis the 1D stage handles.
  The 2D stage covers the remaining two axes.

The policies need a `SplitLayout` for one reason.
A single FFT configuration can express only one transform type across the whole domain.
When the BCs differ by axis-group, the plan builds *two* configurations.
Both configurations cover the full 3D grid, but each excludes the axes the other one owns.

### Kernels and runners

`__global__` kernels live in headers under `detail/kernels/*.cuh`.
`detail/runners/*.hpp` declares thin host-callable launch wrappers, called "runners".
`src/detail/runners/*.cu` defines these runners.
Two mechanisms make the kernel code reusable, across real and complex data, and across dimensionality:

- `ApplyStrideLoop` — a grid-stride loop over up to three dimensions, with X as the innermost, fastest loop.
  This matches the grid-layout convention from [Overview and repository map](#1-overview-and-repository-map).
- A small set of operators (`operator/=`, `operator*`, `operator-`, and others) work on `cuComplex` and `cuDoubleComplex`.
  Because of these operators, generic kernels like `DivideByDenoms` and `TridiagZ` need only one template.
  The same kernel code then compiles for both real and complex data.

Each `.cu` file ends with `INSTANTIATE_*` macro blocks that instantiate every runner for `float`/`double`, and for every dimensionality it needs.

---

## 4. Boundary-condition dispatch table
This section describes the mapping between BCs and components of the codebase.

### Per-axis: BC pair → FFT transform type
The tables below use the notation `axis(A)` where A = X, Y, or Z.
`axis(A)` denotes the FFT transform type for the boundary-condition pair on that axis (see the table below).
`ISolverPlan::GetAxisFFTType` creates this mapping.
An unsupported pair throws an error.

| BC pair                       | Transform     |
|-------------------------------|---------------|
| Periodic / Periodic           | Complex (C2C) |
| Dirichlet / Dirichlet         | DST-I         |
| DirichletStag / DirichletStag | DST-II        |
| Neumann / Neumann             | DCT-I         |
| NeumannStag / NeumannStag     | DCT-II        |
| Dirichlet / Neumann           | DST-III       |
| Neumann / Dirichlet           | DCT-III       |
| DirichletStag / NeumannStag   | DST-IV        |
| NeumannStag / DirichletStag   | DCT-IV        |

### Whole-solver dispatch (`SolverPlan`'s constructor)

The constructor tries, in order:

1. **MPI-only 3D full-real detour** (`TryConstructFromFullRealBCsSplit`)
   The code reaches this branch only in an MPI build, and only when `FFTDim == 3`.
   Before trying anything else, an MPI 3D solver plan checks whether **all** BCs are the same real, non-periodic type: all-Dirichlet, all-DirichletStag, all-Neumann, or all-NeumannStag.
   If so, it deliberately avoids heFFTe's own whole-grid FFT plan.
   Instead, it builds a `SplitLayout` that does the transform itself, one axis-group at a time, with the reshaper handling redistribution between the stages.
   heFFTe was measured to be significantly slower than manual reshapes plus VkFFT for these transforms, so the solver takes this path instead.
   Fully periodic 3D solves, and all 1D or 2D solves, are unaffected.
   They stay on heFFTe's own whole-grid transform.

2. **All BCs identical** (`TryConstructFromFullBCs`) — one `FullLayout` transform over the whole grid, empty `reshape_types`:

   | BCs (all axes) | `solve_method`       | Transform |
   |----------------|----------------------|-----------|
   | Periodic       | `SolveFullPeriodic`  | Complex   |
   | Dirichlet      | `SolveFullDirichlet` | DST-I     |
   | DirichletStag  | `SolveFullDirichlet` | DST-II    |
   | Neumann        | `SolveFullNeumann`   | DCT-I     |
   | NeumannStag    | `SolveFullNeumann`   | DCT-II    |

3. **Non-full BCs** (`ConstructFromNonFullBCs`), by dimensionality:
   - **1D**: any remaining BC pair maps to `SolveFullDirichlet`, on a `FullLayout`, with that pair's transform type.
   - **2D**: always throws an error.
     The solver supports only identical BCs in 2D.

   - **3D**: see the branch table below (`ConstructFromNonFullBCs3D`).
     Every branch there produces a `SplitLayout`.

   | BCs (X, Y, Z)                                          | `solve_method`    | 1D transform | 2D transform | `isolated_dim_idx` | `reshape_types`    | complex workspace? |
   |--------------------------------------------------------|-------------------|--------------|--------------|--------------------|--------------------|--------------------|
   | Periodic, Periodic, Neumann(Stag)                      | `SolvePPNs`       | axis(Z)      | Complex      | 2                  | ToPencils, ToSlabs | yes                |
   | Periodic, Periodic, Dirichlet(Stag)                    | `SolvePPD`        | axis(Z)      | Complex      | 2                  | ToPencils, ToSlabs | yes                |
   | Neumann(Stag), Periodic, Periodic                      | `SolveNsPP`       | axis(X)      | Complex      | 0                  | ToPencils, ToSlabs | yes                |
   | Periodic, Neumann(Stag), Neumann(Stag)                 | `SolvePNsNs`      | Complex      | axis(Y)      | 0                  | ToSlabs, ToPencils | yes                |
   | Periodic, Neumann(Stag), Periodic                      | `SolvePNsP`       | axis(Y)      | Complex      | 1                  | ToPencils, ToSlabs | yes                |
   | Dirichlet(Stag), Dirichlet(Stag), Periodic             | `SolveDDP`        | Complex      | axis(X)      | 2                  | ToSlabs, ToPencils | yes                |
   | no Periodic, X BC == Y BC ≠ Z BC, some Dirichlet(Stag) | `Solve2Real1Real` | axis(Z)      | axis(X)      | 2                  | ToSlabs, ToPencils | no                 |
   | no Periodic, Y BC == Z BC ≠ X BC, some Dirichlet(Stag) | `Solve1Real2Real` | axis(X)      | axis(Y)      | 0                  | ToSlabs, ToPencils | no                 |

   Anything else throws an error.

`isolated_dim_idx` always names the axis the **1D** stage owns.
The 2D stage always owns the other two axes.

### `NonuniformZSolverPlan`

`NonuniformZSolverPlan` builds from only the X and Y BCs (`ISolverPlan<T, 2>`).
A tridiagonal solve handles Z axis instead of an FFT, so Z never enters `axis_strategies`.
`isolated_dim_idx` is always 2, and `reshape_types` is always `{ToSlabs, ToPencils}`.
The plan accepts exactly two BC combinations:

| BCs (X, Y, Z)                                  | `solve_method`     | 2D transform | complex workspace? |
|------------------------------------------------|--------------------|--------------|--------------------|
| Neumann(Stag), Neumann(Stag), (any accepted Z) | `SolveFullNeumann` | axis(X)      | no                 |
| Periodic, Periodic, Neumann(Stag)              | `SolvePPNs`        | Complex      | yes                |

---

## 5. Data-flow traces

Below are a few data-flow traces to help you visualize the steps performed during solver setup and execution.

### Setup phase

Two traces below show what happens once, at construction, before any `Execute` call.

**(a) Local, fully periodic 3D**

1. `SolverPlan` constructs first.
   It builds one `PeriodicStrategy` per axis, matches the all-Periodic branch, and sets `solve_method = SolveFullPeriodic`, `fft_layout = FullLayout{Complex}`, and an empty `reshape_types`.
2. `SolverCore` delegates to `BuildLocalBackend`, which builds a `DummyReshaper` and one in-place 3D C2C VkFFT plan.
3. `AllocDeviceWorkspaces` allocates `cu_real_workspace1` and `cu_cmplx_workspace1` (workspace 2 aliases workspace 1, since there is no MPI).
   `AllocDeviceDenoms` allocates one `cu_denoms[d]` array per axis.
4. `GetEigenvalues` computes each axis's eigenvalues on the host, from the chosen `IApproximation`.
   `CopyDenomsToDevice` copies them once, and `CreateDenomObservers` builds the raw-pointer array the kernels read at solve time.
5. Construction ends here.
   The padded host-transfer buffers for `phi` and `rhs` do not exist yet.
   `Execute` allocates them lazily, on first use.

**(b) Distributed split layout**

1. `SolverPlan` constructs first, as above, picking a `SplitLayout` and a `reshape_types` sequence.
2. `SolverCore` delegates to `BuildDistributedBackend`, which builds the reshaper pipeline first.
3. Only then does it build the FFT plan.
   For a `SplitLayout`, the constructor reads `reshaper->PencilLayout().ns` and `reshaper->SlabLayout().ns`.
   These size the local VkFFT sub-plans for the pencil and slab shapes, which differ from the original decomposition.
4. `AllocDeviceWorkspaces` now allocates workspace 2 as a genuinely separate buffer, since redistribution under MPI is out-of-place, not an alias.
5. Eigenvalue setup proceeds as in the local case.

### Execution phase
**(a) Local, fully periodic 3D**

`Solver<double, 3>::Execute(phi, rhs, ngs_phi, ngs_rhs)`, all BCs Periodic:

| # | Operation                               | Effect                                                                        |
|---|-----------------------------------------|-------------------------------------------------------------------------------|
| 1 | H2D copy `rhs` → padded staging buffer  | whole padded buffer, ghosts included                                          |
| 2 | `UnpackPaddedGrid`                      | strips ghosts → `cu_real_workspace1` holds the contiguous interior RHS        |
| 3 | `common_cmplx::Init`                    | real → complex, into `cu_cmplx_workspace1`                                    |
| 4 | `FFTPolicy::Forward`                    | in-place forward 3D C2C transform                                             |
| 5 | `DivideByDenoms`                        | divide by the summed per-axis eigenvalues                                     |
| 6 | `FFTPolicy::Backward`                   | in-place inverse 3D C2C transform                                             |
| 7 | `common_cmplx::Normalize`               | complex → real, divided by `norm_factor`. This is the solution `φ`            |
| 8 | `PackPaddedGrid` + D2H copy             | writes the interior into the padded output buffer and copies it back to `phi` |

No reshape calls happen at all: `FullLayout` solves never touch the reshaper.
`ExecuteGPU` is identical, except for step 1 and the D2H half of step 8, because the caller already supplies device buffers.

**(b) Distributed split layout — `SolvePPD` (BCs Periodic, Periodic, DirichletStag)**

| #  | Operation                       | Effect                                                                                                                                                            |
|----|---------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1  | ReshapeReal: original → pencils | redistributes so Z is fully local per rank                                                                                                                        |
| 2  | `Forward1D`                     | real DST-II transform along Z only                                                                                                                                |
| 3  | ReshapeReal: pencils → slabs    | redistributes so X,Y are fully local per rank                                                                                                                     |
| 4  | `common_cmplx::Init`            | real → complex                                                                                                                                                    |
| 5  | `Forward2D`                     | complex C2C transform over X, Y                                                                                                                                   |
| 6  | `DivideByDenoms`                | uses the *reshaped* layout's local sizes and offsets — after redistribution both the local block shape and its global offset change                               |
| 7  | `Backward2D`                    | inverse C2C over X, Y                                                                                                                                             |
| 8  | `common_cmplx::Normalize`       | complex → real                                                                                                                                                    |
| 9  | ReshapeReal: slabs → pencils    | back to a Z-local layout                                                                                                                                          |
| 10 | `Backward1D`                    | inverse DST-II along Z                                                                                                                                            |
| 11 | ReshapeReal: pencils → original | back to the solver's original decomposition                                                                                                                       |

---

## 6. Distributed decomposition and heFFTe communication

### Process grid

`mpi_utils::InitMPIGrid(comm, {nps_y, nps_z})` builds a non-periodic Cartesian communicator.
Internally, it hands the process counts to MPI in `{Z, Y, X}` order.
This order matters because MPI treats its *last* array dimension as fastest-varying, while this library treats its *first* dimension as fastest-varying.
`nps_x` is hard-coded to `1`, so **the X axis is never distributed.**
This is deliberate, the library keeps every X-line fully local on one rank, mirroring `PoisFFT`.

`mpi_utils::GetLocalGridInfo(global_ns, cart_comm)` tells a rank its own slice.
It returns this rank's local sizes and offset.

`mpi_utils::SelectLocalDevice(comm)` assigns each process a device.

### Decompositions
In a distributed solve the data moves through three decompositions:

- **Original brick** — the layout the caller gives to the solver's constructor.
  X is entirely local.
  The process grid distributes Y and Z over `nps_y × nps_z`.
- **Pencil** — spans the *entire* world along `isolated_dim_idx`, so that axis is fully local.
  The full process grid splits it over the other two axes.
  It feeds the **1D** FFT stage.
- **Slab** — spans the entire world along the two axes that are *not* isolated.
  The process grid cuts it only along the isolated axis, across all ranks.
  It feeds the **2D** FFT stage.

The plan sets which axis is isolated, and which two axes form the slab.

### Moving between decompositions
A **reshaper** is what performs changes between decompositions.
To build a reshaper, specify exactly one `ToPencils` step and one `ToSlabs` step, in either order.
If you specify either step twice, or omit one, the build fails.
This produces a sequence of three decompositions: `[original, decomp1, decomp2]`.
From the three decompositions, it derives four heFFTe redistribution objects: `original→decomp1`, `decomp1→decomp2`, and their two inverses, `decomp2→decomp1` and `decomp1→original`.
It tracks internally which of the four runs next.
Each `ReshapeReal` or `ReshapeCmplx` call runs exactly the next one in sequence, and returns the resulting local grid layout:

| Call # | Redistribution applied |
|--------|------------------------|
| 1      | original → decomp1     |
| 2      | decomp1 → decomp2      |
| 3      | decomp2 → decomp1      |
| 4      | decomp1 → original     |

The counter is **never reset** between solves.
**Every split-layout `Solve*` method must issue exactly four `ReshapeReal`/`ReshapeCmplx` calls per `Execute`.**

### What heFFTe is actually used for

heFFTe serves two separate purposes here:

- **Redistribution.** For `SplitLayout`, the reshaper builds heFFTe's low-level redistribution objects directly and the FFT is handled via VkFFT.

- **Whole-grid transforms.** On the `FullLayout` path, heFFTe's own 3D FFT plan does the entire transform, including whatever internal redistribution it needs.
  This path applies when every axis has identical BCs, **in MPI builds, except for the** [full-real detour](#whole-solver-dispatch-solverplans-constructor).
  This library's reshaper never gets involved.

### Why local and distributed builds share one `Solve*` body

In the local build, replaying `reshape_types` produces a reshaper whose "redistribution" is always a no-op (`DummyReshaper`): it returns the grid unchanged.
As a result, every `Solve*` method in `solver.hpp` that touches the reshaper exists **once**, in source code.
It costs nothing extra in a build without MPI.
The distributed and local builds are not separate code paths for these methods, only separate runtime behaviors of the same code.

### FFT scaling

The code never asks heFFTe or VkFFT to scale its own output.
The solver uses both unscaled, and applies normalization exactly once, at the end of each `Solve*` method, by dividing by `norm_factor`.
For situations where heFFTe's unscaled convention differs from VkFFT's, a correction is applied (see `axis_strategies.hpp`).

### Distributed 1D

Rather than leave 1D unsupported, the distributed 1D handle gathers the entire line onto rank 0.
It runs an ordinary local, VkFFT, 1D transform there, then scatters the result back out.
This mirrors an equivalent functionality in PoisFFT, and it exists purely for completeness.

---

## 7. Bindings

Four layers wrap the C++ core.

1. **`include/poisson-solver/solver.hpp`** — the C++ API from [The public API](#2-the-public-api).

2. **`src/c_binding.cpp` / `include/poisson-solver/c_binding.h`** — the C API.
   `PoissonSolver` is an opaque struct.
   It has a `void *handle` that points at the heap-allocated C++ solver.
   It also has `grid_size`, `dim`, and per-axis sizes `nx`, `ny`, `nz` (the code sets unused axes to 1).

   The `DEFINE_EXECUTE` and `DEFINE_FINALIZE` macros generate the per-dimension dispatch, instead of hand-written code for each dimension.
   This dispatch calls into the right `Solver<T, Dim>` instantiation.
   Single-precision entry points carry the `Float` suffix, always last, after any dimension suffix.

3. **`src/detail/c_binding_for_fortran.cpp`** — a thin wrapper layer, with names suffixed `*ForFortran`.
   Its sole job is to translate Fortran's `MPI_Fint` communicator handles into a C++ `MPI_Comm`.
   For the utilities that return one, it also translates back the other way.
   Then it forwards the call to the plain C API.
   Only the solver constructors and the MPI setup utilities need this translation, so only they have wrappers.
   `Execute`, `ExecuteGPU`, and `Finalize` take no communicator, so Fortran binds straight to the plain C symbols and skips this layer entirely.

4. **`src/fortran_binding.f90`** — the Fortran module, named **`PoisSolver`**.
   Constants such as `PoisSolver_Periodic` live here.
   It exposes a `PoissonSolver` derived type, with generic interfaces that dispatch on the rank of the array arguments.
   Because of this, callers do not need separate 1D, 2D, and 3D subroutine names.

   Fortran does not pass ghost-cell counts explicitly.
   Instead, it infers them from the shape of the caller's array.
   It compares that shape against the `nx`, `ny`, `nz` fields recorded in the underlying C struct.

---

## 8. Build system

Build system is mostly described in [examples](../examples/) but internal details of it are discussed only here.

**Fetched dependencies.** The build fetches all of these at configure time, with `FetchContent`:

- **VkFFT** — the local FFT library.
  The build selects the CUDA backend explicitly.
  Every build needs VkFFT, including MPI builds, because split-layout distributed solves still run local VkFFT transforms on pencils and slabs.

- **A fork of heFFTe** — only in MPI builds, with its CUDA backend and GPU-aware MPI support enabled.
  See [Repository quirk: the heFFTe fork](#10-repository-quirk-the-heffte-fork) for details.
- **doctest** — the C++ test framework.
- **doxygen-awesome-css** — styling for the generated Doxygen output.
  The build fetches this only when CMake finds Doxygen on the system.

---

## 9. Tests

**This section assumes that you understand how poisson-solver is used in the** [container environment](../container-environment/).
If you do not, please, explore the [container environment directory](../container-environment/) first and then return to this section.

Tests run on the `chimera` cluster via `ctest`, inside the [container environment](../container-environment/), using Charliecloud.
The relative paths of the Charliecloud invocation resolve against the described container environment layout.

**Build target.** `cmake --build build/ --target tests` builds every test executable.

**Running them.** On the cluster head node, navigate into `poisson-solver/build` and run `ctest -V`.
Each test's `srun` command handles its own allocation, so do not run `ctest` itself from inside an already-allocated GPU node.
**If you already hold a GPU allocation for another reason, you can instead bypass `srun` and invoke a binary directly.**

**Scoring.** doctest prints nothing until a test fails, so several minutes of silence is normal.
Fortran tests are scored by matching `FAIL` in their output (`FAIL_REGULAR_EXPRESSION`), not by exit code.

### Acknowledgments
The Fortran tests are adapted from the [PoisFFT](https://github.com/LadaF/PoisFFT) library.
They have been modified to use the `poisson-solver` API and perform additional tests.
When using this library for your research please cite the `PoisFFT` article.

Vladimir Fuka: PoisFFT - a free parallel fast Poisson solver, Applied Mathematics and Computation, 2015

available online: http://dx.doi.org/10.1016/j.amc.2015.03.011

preprint: http://arxiv.org/abs/1409.8116

---

## 10. Repository quirk: the heFFTe fork

The root `CMakeLists.txt` fetches a **fork** of heFFTe, `mntjams/heffte`, not the upstream `icl-utk-edu/heffte`.
`FetchContent` pins this fork to a fixed commit.

Upstream heFFTe's cuFFT backend supports only complex-to-complex, real-to-complex, DCT-II, and DST-II transforms.
This solver needs real-to-real transforms that upstream does not supply.
So the fork adds cuFFT-backed executors for:

- **DST-I**
- **DCT-IV**
- **DST-IV**

Without these, you could not build the all-Dirichlet solvers, or the distributed solvers for staggered mixed BCs from [Boundary-condition dispatch table](#4-boundary-condition-dispatch-table), against upstream heFFTe at all.
