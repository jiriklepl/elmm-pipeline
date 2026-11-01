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

    // Define the grid side lengths in X and Y
    // The Z length instead comes from the z and z_u arrays (see below)
    float lens[2] = {1.0f, 1.0f};

    // Define the boundary conditions
    // in the order of {west, east, south, north, bottom, top}
    BoundaryCondition bcs[6] = {NeumannStag,
                                NeumannStag,
                                NeumannStag,
                                NeumannStag,
                                NeumannStag,
                                NeumannStag};

    // The Z axis is nonuniform, so two arrays describe it.
    // z_u holds the cell faces: SIDE_SIZE + 1 values, from the bottom up.
    // z holds the cell centers: SIDE_SIZE values, one per cell.
    // Cell k reaches from z_u[k] to z_u[k + 1], and z[k] lies between them.
    //
    // Even spacing keeps this example short.
    // A real case may space the faces unevenly.
    const float len_z = 1.0f;
    const float dz = len_z / SIDE_SIZE;

    float z_u[SIDE_SIZE + 1];
    for (size_t k = 0; k < SIDE_SIZE + 1; ++k) {
        z_u[k] = (float)k * dz;
    }

    float z[SIDE_SIZE];
    for (size_t k = 0; k < SIDE_SIZE; ++k) {
        z[k] = (z_u[k] + z_u[k + 1]) / 2;
    }

    // Initialize the solver
    PoissonSolver solver;
    PoissonSolverNonuniformZNewFloat(
        &solver, ns, lens, z, z_u, bcs, FD2, NULL, NULL, NULL);

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
    PoissonSolverNonuniformZExecuteFloat(&solver, phi, rhs, ngs, ngs);

    // Explicit finalization
    PoissonSolverNonuniformZFinalizeFloat(&solver);

    free(rhs);
    free(phi);

    printf("Successfully finished example local_nonuniform\n");

    return 0;
}
