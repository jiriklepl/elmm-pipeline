#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

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

    // Prepare input
    // Fill it with desired values (here, we fill 1.0)
    float *rhs = malloc(buffer_size * sizeof(float));
    for (size_t i = 0; i < buffer_size; ++i) {
        rhs[i] = 1.0f;
    }

    // Prepare result buffer
    float *phi = calloc(buffer_size, sizeof(float));

    // Solve the equation
    // (phi and rhs have the same amount of ghost cells)
    PoissonSolverExecuteFloat(&solver, phi, rhs, ngs, ngs);

    // Use double precision instead by dropping the Float suffix on every
    // call (PoissonSolverNew, PoissonSolverExecute, PoissonSolverFinalize)

    // Explicit finalization
    PoissonSolverFinalizeFloat(&solver);

    free(rhs);
    free(phi);

    printf("Successfully finished example local_uniform\n");

    return 0;
}
