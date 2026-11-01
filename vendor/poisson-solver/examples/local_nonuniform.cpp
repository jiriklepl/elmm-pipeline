#include <array>
#include <cstddef>
#include <iostream>
#include <vector>

// You only need to include solver.hpp
#include <poisson-solver/solver.hpp>

using Prec = float;
using namespace poisson_solver;

int main() {
    // We will construct a 3D solver
    constexpr std::size_t dim = 3;

    // Define the grid side sizes
    constexpr std::size_t side_size = 64;
    GridSizes<dim> ns{side_size, side_size, side_size};

    // Define the grid side lengths in X and Y
    // The Z length instead comes from the z_u vector below
    std::array<Prec, 2> lens{1.0, 1.0};

    // Define the boundary conditions
    // in the order of {west, east, south, north, bottom, top}
    constexpr auto Ns = BC::NeumannStag;
    BCs<dim> bcs{Ns, Ns, Ns, Ns, Ns, Ns};

    // The Z axis is nonuniform, so two vectors describe it.
    // `z_u` holds the cell faces: side_size + 1 values, from the bottom up.
    // `z` holds the cell centers: side_size values, one per cell.
    // Cell k reaches from z_u[k] to z_u[k + 1], and z[k] lies between them.
    constexpr Prec len_z = 1.0;
    constexpr Prec dz = len_z / side_size;

    // Even spacing keeps this example short.
    // A real case may space the faces unevenly.
    std::vector<Prec> z_u(side_size + 1);
    for (std::size_t k{0}; k < z_u.size(); ++k) {
        z_u[k] = static_cast<Prec>(k) * dz;
    }

    std::vector<Prec> z(side_size);
    for (std::size_t k{0}; k < z.size(); ++k) {
        z[k] = (z_u[k] + z_u[k + 1]) / 2;
    }

    // Initialize the solver
    // The approximation is optional and SpectralApproximation as default
    NonuniformZSolver<Prec> solver{
        ns, lens, z, z_u, bcs, FiniteDifference2Approximation<Prec>{}};

    // Grid will have two ghost cells on every side
    // One axis will therefore have four ghost cells
    // (two on the left, two on the right)
    GridSizes<dim> ngs{2, 2, 2};

    // The buffers hold the grid and the ghost cells on both sides
    std::size_t buffer_size{1};
    for (std::size_t d{0}; d < dim; ++d) {
        buffer_size *= ns[d] + 2 * ngs[d];
    }

    // Prepare input
    // Fill it with desired values (here, we fill 1.0)
    std::vector<Prec> rhs(buffer_size, 1.0);

    // Prepare result buffer
    std::vector<Prec> phi(buffer_size, 0.0);

    // Solve the equation
    // (phi and rhs have the same amount of ghost cells)
    solver.Execute(phi, rhs, ngs, ngs);

    // Finalization handled via RAII

    std::cout << "Successfully finished example local_nonuniform\n";
}
