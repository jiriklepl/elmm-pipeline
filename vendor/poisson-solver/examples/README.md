# Examples

## How to build

### Configuring the build
Navigate to the project root directory.
Use one of the prepared CMake presets:
1. `release` - local solver (this is what you usually want for local use)
2. `release-mpi` - distributed solver (this is what you usually want for distributed use)
3. `release-benchmarks` - local solver with benchmarking code (use this if you want to measure where the solver spends time)
4. `release-mpi-benchmarks` - distributed solver with benchmarking code (similar use-case to the `release-benchmarks` preset)
5. `debug` - local solver debug build
6. `debug-mpi` - distributed solver debug build


As an alternative approach, you can control the CMake options directly.
You can also combine the approaches and use presets while specifying some options directly.
The options are:
1. `POISSON_SOLVER_BUILD_FORTRAN` - whether to build a Fortran binding
2. `POISSON_SOLVER_USE_MPI` - determines the type of build - local/distributed
3. `POISSON_SOLVER_ENABLE_BENCHMARKS` - whether to include benchmarking code (timers, benchmarks, etc.)

Beware that the option `BUILD_SHARED_LIBS` is `ON` by default.

Some examples of CMake invocation are (from the project root directory):
```bash
# Local release config
cmake --preset release

# Distributed release config
cmake --preset release-mpi

# Local release config with no Fortran binding
cmake --preset release -DPOISSON_SOLVER_BUILD_FORTRAN=OFF

# Manual config of a static, distributed library with no Fortran binding
cmake -S . -B build/ -DBUILD_SHARED_LIBS=OFF -DPOISSON_SOLVER_USE_MPI=ON -DPOISSON_SOLVER_BUILD_FORTRAN=OFF -DCMAKE_BUILD_TYPE=Release
```

### Building the library
To then build the library, use:
```bash
cmake --build build/
```
This creates a library inside the `build/lib/` directory (the debug library has a `_debug` suffix).
If you build the Fortran binding, the `.mod` files land in `build/modules/`.

### Other build targets
#### Examples
Use the `examples` target to build the example programs.
```bash
cmake --build build/ --target examples
```
The binaries land in `build/examples/`.
**Your selected preset from the previous step affects which examples are built (local/distributed).**
See [How to use](#how-to-use) below for more information.

#### API docs
Use the `doc` target to generate the doxygen documentation _for the current configuration_.
**There are differences in documentation for configurations with `POISSON_SOLVER_USE_MPI=ON` vs those with `POISSON_SOLVER_USE_MPI=OFF`**.
The generated documentation will be in `build/docs/html/`.

#### Tests
There is also the `tests` target for development and (maybe) illustratory purposes.
You can read more about tests in the [documentation](../docs/docs.md#9-tests).

#### Benchmarks
The last target is `benchmarks`.
You can read about the benchmarks (performance of this library, how to run them, etc.) in [benchmark-results](../benchmark-results/).

## How to use

`poisson-solver` supports C++, C and Fortran.
This directory contains example source code for all these languages along with explanatory comments.
The six examples are:
1. **local uniform solver** - `local_uniform.{cpp,c,f90}`
2. **local nonuniform solver** - `local_nonuniform.{cpp,c,f90}`
3. **distributed uniform solver** - `distributed_uniform.{cpp,c,f90}`
4. **distributed nonuniform solver** - `distributed_nonuniform.{cpp,c,f90}`
5. **local uniform solver with device buffers** - `local_uniform_nocopy.{cpp,c,f90}`
6. **local 2D solver printing its result** - `local_with_pretty_print.cpp` (C++ only)

**These are meant to be read!**
It is recommended to explore them in the given order.
But feel free to explore only those relevant to your language or usecase.

### Which preset builds which example
The build is either local or distributed, never both:

| Preset                                               | Examples built                                                                         |
|------------------------------------------------------|----------------------------------------------------------------------------------------|
| `release`, `debug`, `release-benchmarks`             | `local_uniform`, `local_nonuniform`, `local_uniform_nocopy`, `local_with_pretty_print` |
| `release-mpi`, `debug-mpi`, `release-mpi-benchmarks` | `distributed_uniform`, `distributed_nonuniform`                                        |

### Fortran targets
Each example gives one target per language: `-cpp`, `-c` and `-f90`.
The `-f90` targets exist only when `POISSON_SOLVER_BUILD_FORTRAN` is `ON` (the default).
`local_with_pretty_print` is the only exception.
It is written in C++ only, so it gives just the `local_with_pretty_print-cpp` target.
Our [PoisFFT fork](https://github.com/mntjams/PoisFFT) contains the same example, with the same configuration so you can compile it and compare the results.
Assuming you use our [container environment](../container-environment/) and you have our PoisFFT fork cloned inside `mapped/`, allocate a node and run `srun ch-run -b mapped:/opt/build imgdir -- /bin/bash -c "cd /opt/build/PoisFFT/src && unset CXX && unset CC && scons test"`.
The example can then be run with `srun ch-run -b mapped:/opt/build imgdir -- /bin/bash -c "/opt/build/PoisFFT/bin/gcc/local_with_pretty_print"`.

### Running the local examples
A local example needs one GPU. Run the binary with no arguments:
```bash
./build/examples/local_uniform-cpp
```

### Running the distributed examples
If you don't have the required software, you can use our [container environment](../container-environment/).
A distributed example needs exactly four processes, at least one GPU, and stops with an error message on any other process count.
Allocate that many processes using your cluster manager.
Then, run the example.
For `SLURM` this could look like (using our container):
```
# Start in the container directory (has `mapped/` and `imgdir/`)
salloc -p <partition-name> -N 2 -n 4 --gpus 4 -w <node-names>

# Set the env. var per guidance in container-environment
UCX_POSIX_USE_PROC_LINK=n srun --mpi=pmix ch-run -b mapped:/opt/build imgdir -- /bin/bash -c "cd /opt/build/poisson-solver/build/examples && ./distributed_uniform-cpp"

# Invocation of another example
UCX_POSIX_USE_PROC_LINK=n srun --mpi=pmix ch-run -b mapped:/opt/build imgdir -- /bin/bash -c "cd /opt/build/poisson-solver/build/examples && ./distributed_nonuniform-c"
exit
```

### What success looks like
An example solves the equation, then prints `Successfully finished example <name>` and returns 0.
`<name>` is the example's base name, for instance `local_uniform`.
Read the source file to see the API calls it makes.
