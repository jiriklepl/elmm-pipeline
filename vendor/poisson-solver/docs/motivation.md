# Motivation Behind poisson-solver

`PoisFFT` is a Fortran library that solves the Poisson equation using the Fast Fourier Transform.
It is a part of ELMM (Extended Large Eddy Microscale Model), a LES (Large Eddy Simulation) model for atmospheric flows.
`PoisFFT` is written entirely in Fortran and utilizes only CPUs.
It is parallelized using `OpenMP` in a single-node (local) configuration and using `MPI` in a multi-node (distributed) configuration.
Furthermore, it offers a hybrid `MPI` + `OpenMP` parallelization.

`PoisFFT` requires the users to first define a domain (number of grid cells, grid lengths, boundary conditions, etc.) and then allows them to repeatedly solve the Poisson equation with this domain configuration by supplying the right-hand side of the equation $`\Delta \phi = b`$.

Usually, the equation solving can be summarized as the following pipeline:
1. Apply a forward FFT on the whole RHS,
2. divide each RHS element by a sum of pre-computed eigenvalues,
3. apply a backward FFT on the whole RHS and
4. normalize the result to get $`\phi`$.

`PoisFFT` also enables the use of solvers that do not have uniform lengths of individual grid cells on the Z axis.
These are called `nonuniform-z` solvers and for them the previously mentioned pipeline differs in the following:
1. Forward FFT is applied only in the X and Y directions,
2. for each row in the Z direction a tridiagonal system is solved via a simplified form of Gaussian elimination,
3. backward FFT is applied only in the X and Y directions and
4. normalization factor depends only on dimensions transformed via the FFT.

For different combinations of BCs (boundary conditions) the pipeline may slightly vary (e.g., the real RHS is first converted into a complex RHS or different FFT types are used across the directions).
For distributed solvers this is even more complicated due to the requirement of data locality before FFT steps.
FFT computation in some direction requires that data is not distributed in said direction.
This enforces `MPI` communication before the FFT steps.

`PoisFFT` is a great candidate for a rewrite into C++ and CUDA to take advantage of the GPUs available on the MFF cluster `chimera`.
`poisson-solver` is exactly this rewrite which we’ve done as a part of Research Project (NPRG070) at MFF UK.
We've also modified `ELMM` (and named it `gpu-elmm`) to link with our `poisson-solver`.
More details on how to run our `poisson-solver` can be found [here](../examples/) and information on how to run `gpu-elmm` with our solver can be found [here](../container-environment/).
