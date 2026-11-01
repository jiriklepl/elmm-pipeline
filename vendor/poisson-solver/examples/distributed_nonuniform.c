#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

// Don't forget to include MPI
#include <mpi.h>

// c_binding.h declares the MPI helper functions
// (InitMPIGrid, GetLocalGridInfo, SelectLocalDevice)
// directly, when built with POISSON_SOLVER_USE_MPI
#include <poisson-solver/c_binding.h>

// Define the global grid side sizes
#define GLOBAL_SIDE_SIZE 64

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    // We will construct a 3D solver
    const int ndims = 3;

    int comm_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    if (comm_size != 4) {
        fprintf(stderr, "This example requires exactly 4 processes\n");
        MPI_Finalize();
        return 1;
    }

    // Decomposition into a 1x2x2 process grid
    int nps[2] = {2, 2};
    MPI_Comm cart_comm;
    PoissonSolverInitMPIGrid(MPI_COMM_WORLD, nps, &cart_comm);

    // Selects local device using round-robin approach
    // Omit this if you already selected a local device
    // (e.g. via your own mechanism)
    int local_device;
    PoissonSolverSelectLocalDevice(MPI_COMM_WORLD, &local_device);

    int gns[3] = {GLOBAL_SIDE_SIZE, GLOBAL_SIDE_SIZE, GLOBAL_SIDE_SIZE};

    // Define the global grid side lengths in X and Y
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

    // ns and offs receive the local grid size and offset
    // of the local grid from the origin
    // Use these to initialize the solver
    int ns[3];
    int offs[3];
    PoissonSolverGetLocalGridInfo(gns, cart_comm, ns, offs);

    // The Z axis is nonuniform, so two arrays describe it.
    // z_u holds the cell faces: gns[2] + 1 values, from the bottom up.
    // z holds the cell centers: gns[2] values, one per cell.
    // Cell k reaches from z_u[k] to z_u[k + 1], and z[k] lies between them.
    // Both arrays describe the whole global Z axis: every rank builds the
    // same values here.
    //
    // Even spacing keeps this example short.
    // A real case may space the faces unevenly.
    const float len_z = 1.0f;
    const float dz = len_z / (float)gns[2];

    float z_u[GLOBAL_SIDE_SIZE + 1];
    for (size_t k = 0; k < GLOBAL_SIDE_SIZE + 1; ++k) {
        z_u[k] = (float)k * dz;
    }

    float z[GLOBAL_SIDE_SIZE];
    for (size_t k = 0; k < GLOBAL_SIDE_SIZE; ++k) {
        z[k] = (z_u[k] + z_u[k + 1]) / 2;
    }

    // Grid will have two ghost cells on every side
    // One axis will therefore have four ghost cells
    // (two on the left, two on the right)
    int ngs[3] = {2, 2, 2};

    // The buffers hold the local grid and the ghost cells on both sides
    size_t buffer_size = 1;
    for (int d = 0; d < ndims; ++d) {
        buffer_size *= (size_t)(ns[d] + 2 * ngs[d]);
    }

    // Prepare input
    // Fill it with desired values (here, we fill 1.0)
    float *local_rhs = malloc(buffer_size * sizeof(float));
    for (size_t i = 0; i < buffer_size; ++i) {
        local_rhs[i] = 1.0f;
    }

    // Prepare result buffer
    float *local_phi = calloc(buffer_size, sizeof(float));

    // Initialize the solver
    // Use the local grid information here
    PoissonSolver solver;
    PoissonSolverNonuniformZNewFloat(
        &solver, ns, lens, z, z_u, bcs, FD2, gns, offs, &cart_comm);

    // Solve the equation
    // (local_phi and local_rhs have the same amount of ghost cells)
    PoissonSolverNonuniformZExecuteFloat(&solver, local_phi, local_rhs, ngs, ngs);

    // Explicit finalization
    // Finalize the solver before MPI_Comm_free and MPI_Finalize below
    PoissonSolverNonuniformZFinalizeFloat(&solver);

    free(local_rhs);
    free(local_phi);

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();

    printf("Successfully finished example distributed_nonuniform\n");

    return 0;
}
