# poisson-solver

## Description
`poisson-solver` is a modern GPU-based library for solving the equation $`\Delta \phi = b`$ where $`\phi`$ and $`b`$ are functions of 1, 2 or 3 spatial dimensions.
It allows for a selection from a wide range of domain configurations and can be employed to work on a single-node (local) or multi-node (distributed) configurations.

It is based on the [PoisFFT](https://github.com/LadaF/PoisFFT) library and provides the same functionalities.

## Requirements
`poisson-solver` requires the following compilers, libraries and hardware (those that are **completely optional**, **fetched automatically** or **required only in certain build configurations** are marked accordingly):
- CMake >= 4.0,
- C++ compiler supporting C++20,
- Fortran compiler (**optional** - see [examples](./examples/README.md) for information about disabling Fortran binding compilation),
- `VkFFT` for computation of local FFTs (**fetched automatically**),
- `heFFTe` for computation of distributed FFTs (**only for distributed use but fetched automatically in that configuration**),
- GPU-aware `MPI` with C++ API (**only for distributed use**),
- Fortran API of that same `MPI` library (**optional for distributed use, not required for local use** - see [examples](./examples/README.md) to disable the Fortran binding),
- NVIDIA GPU,
- CUDA Toolkit (`nvcc` for the build) with these runtime libraries:
  - CUDA runtime,
  - CUDA driver API,
  - NVRTC,
  - cuFFT (**only for distributed use** - `heFFTe` uses it for the distributed FFTs).

## Detailed documentation
For motivation behind this solver see [this](./docs/motivation.md).
It is highly recommended that you explore the [examples](./examples/) directory as that contains the solver usage guide and more useful information (e.g., how to build).
We have included a dockerfile (inside [container-environment](./container-environment/)) that contains all the necessary software required to run the solver.
See the [container-environment](./container-environment/) directory to learn more about the deployment of the solver inside our container.
[benchmark-results](./benchmark-results/) contains the performance comparison of `poisson-solver` against the original `PoisFFT` library.
At last, see [docs](./docs/) for the inner code structure of `poisson-solver`.

## Installed device API

The [status-returning device API](./docs/device-api.md) accepts device pointers
and a caller-supplied CUDA stream, with matching Fortran `c_ptr` bindings. Its
initial scope is local uniform double-precision FD2/FD4. The `release-sm86`
preset pins the local compiler/architecture configuration; dependency commits
are recorded in [dependencies.lock.json](./dependencies.lock.json). CMake install
exports headers, the library, Fortran modules and `PoissonSolver::poisson-solver`.

## AI usage
Claude and Google Gemini were used to:
- assist with writing and editing the documentation, and
- assist with parts of the implementation (code generation to help iterate through ideas).

After the use of these tools, the code and documentation were reviewed manually, the authors take full responsibility for the final form of the project.
