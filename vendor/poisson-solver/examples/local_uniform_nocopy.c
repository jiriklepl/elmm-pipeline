#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime_api.h>

// You only need to include c_binding.h
#include <poisson-solver/c_binding.h>

// Define the grid side sizes
#define SIDE_SIZE 64

int main(void) {
    // We will construct a 3D solver
    const int ndims = 3;

    int ns[3] = {SIDE_SIZE, SIDE_SIZE, SIDE_SIZE};

    // Define the grid side lengths
    float lens[3] = {1.0f, 1.0f, 1.0f};

    // Define the boundary conditions
    // in the order of {west, east, south, north, bottom, top}
    BoundaryCondition bcs[6] = {NeumannStag,
                                NeumannStag,
                                NeumannStag,
                                NeumannStag,
                                NeumannStag,
                                NeumannStag};

    // Initialize the solver
    PoissonSolver solver;
    PoissonSolverNewFloat(&solver, ndims, ns, lens, bcs, FD2, NULL, NULL, NULL);

    // Grid will have two ghost cells on every side
    // One axis will therefore have four ghost cells
    // (two on the left, two on the right)
    int ngs[3] = {2, 2, 2};

    // The buffers hold the grid and the ghost cells on both sides
    size_t buffer_size = 1;
    for (int d = 0; d < ndims; ++d) {
        buffer_size *= (size_t)(ns[d] + 2 * ngs[d]);
    }
    size_t bytes = buffer_size * sizeof(float);

    // Allocate the input and result buffers directly on the device.
    float *d_phi = NULL;
    float *d_rhs = NULL;
    cudaMalloc((void **)&d_phi, bytes);
    cudaMalloc((void **)&d_rhs, bytes);

    // A real application would already have the right-hand side on the
    // device (for example, produced there by a previous kernel). Here we
    // fill a host buffer instead and copy it over, purely to have some
    // input data for the example.
    float *rhs = malloc(bytes);
    for (size_t i = 0; i < buffer_size; ++i) {
        rhs[i] = 1.0f;
    }
    cudaMemcpy(d_rhs, rhs, bytes, cudaMemcpyHostToDevice);
    free(rhs);

    // Solve the equation directly on the device buffers.
    // No host<->device copy of phi or rhs happens inside the solver call
    // itself.
    // (phi and rhs have the same amount of ghost cells)
    PoissonSolverExecuteGPUFloat(&solver, d_phi, d_rhs, ngs, ngs);

    // Copy the result back to the host to inspect it.
    float *phi = malloc(bytes);
    cudaMemcpy(phi, d_phi, bytes, cudaMemcpyDeviceToHost);
    free(phi);

    // Explicit finalization
    PoissonSolverFinalizeFloat(&solver);

    cudaFree(d_phi);
    cudaFree(d_rhs);

    printf("Successfully finished example local_uniform_nocopy\n");

    return 0;
}
