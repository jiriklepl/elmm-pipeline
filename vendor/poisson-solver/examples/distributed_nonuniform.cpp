#include <array>
#include <cstddef>
#include <vector>
#include <iostream>

// Don't forget to include MPI
#include <mpi.h>

// mpi_utils are required to correctly set up the distributed grid
#include <poisson-solver/solver.hpp>
#include <poisson-solver/mpi_utils.hpp>

using Prec = float;
using namespace poisson_solver;

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    // We will construct a 3D solver
    constexpr std::size_t dim = 3;

    int comm_size{0};
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    if (comm_size != 4) {
        std::cerr << "This example requires exactly 4 processes\n";
        MPI_Finalize();
        return 1;
    }

    // Decomposition into a 1x2x2 process grid
    MPI_Comm cart_comm = mpi_utils::InitMPIGrid(MPI_COMM_WORLD, {2, 2});

    // Selects local device using round-robin approach
    // Omit this if you already selected a local device
    // (e.g. via your own mechanism)
    mpi_utils::SelectLocalDevice(MPI_COMM_WORLD);

    // Define the global grid side sizes
    constexpr std::size_t global_side_size = 64;
    GridSizes<dim> gns{global_side_size, global_side_size, global_side_size};

    // Define the global grid side lengths in X and Y
    // The Z length instead comes from the z and z_u vectors (see below)
    std::array<Prec, 2> lens{1.0, 1.0};

    // Define the boundary conditions
    // in the order of {west, east, south, north, bottom, top}
    constexpr auto Ns = BC::NeumannStag;
    BCs<dim> bcs{Ns, Ns, Ns, Ns, Ns, Ns};

    // local_info contains the local grid size and offset
    // of the local grid from the origin
    // Use this to initialize the solver
    auto local_info = mpi_utils::GetLocalGridInfo(gns, cart_comm);

    // The Z axis is nonuniform, so two vectors describe it.
    // `z_u` holds the cell faces: gns[2] + 1 values, from the bottom up.
    // `z` holds the cell centers: gns[2] values, one per cell.
    // Cell k reaches from z_u[k] to z_u[k + 1], and z[k] lies between them.
    // Both vectors describe the whole global Z axis: every rank builds the
    // same values here.
    constexpr Prec len_z = 1.0;

    // Even spacing keeps this example short.
    // A real case may space the faces unevenly.
    const Prec dz = len_z / static_cast<Prec>(gns[2]);

    std::vector<Prec> z_u(gns[2] + 1);
    for (std::size_t k{0}; k < z_u.size(); ++k) {
        z_u[k] = static_cast<Prec>(k) * dz;
    }

    std::vector<Prec> z(gns[2]);
    for (std::size_t k{0}; k < z.size(); ++k) {
        z[k] = (z_u[k] + z_u[k + 1]) / 2;
    }

    // Grid will have two ghost cells on every side
    // One axis will therefore have four ghost cells
    // (two on the left, two on the right)
    GridSizes<dim> ngs{2, 2, 2};

    // The buffers hold the local grid and the ghost cells on both sides
    std::size_t buffer_size{1};
    for (std::size_t d{0}; d < dim; ++d) {
        buffer_size *= local_info.ns[d] + 2 * ngs[d];
    }

    // Prepare input
    // Fill it with desired values (here, we fill 1.0)
    std::vector<Prec> local_rhs(buffer_size, 1.0);

    // Prepare result buffer
    std::vector<Prec> local_phi(buffer_size, 0.0);

    {
        // Initialize the solver
        // The approximation is optional and SpectralApproximation as default
        // Use the local grid information here
        NonuniformZSolver<Prec> solver{
            local_info.ns, lens, z, z_u, bcs, gns,
            local_info.offs, cart_comm, FiniteDifference2Approximation<Prec>{}};

        // Solve the equation
        // (local_phi and local_rhs have the same amount of ghost cells)
        solver.Execute(local_phi, local_rhs, ngs, ngs);

        // Solver finalization handled via RAII, before MPI_Finalize below
    }

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();

    std::cout << "Successfully finished example distributed_nonuniform\n";
}
