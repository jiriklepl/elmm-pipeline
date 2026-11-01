#include <array>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>

// You only need to include solver.hpp
#include <poisson-solver/solver.hpp>

using Prec = double;
using namespace poisson_solver;

int main() {
    constexpr std::size_t dim = 2;

    // Define the grid side sizes
    constexpr std::size_t side_size = 4;
    GridSizes<dim> ns{side_size, side_size};

    // Define the grid side lengths
    std::array<Prec, dim> lens{1.0, 1.0};

    // Define the boundary conditions
    // in the order of {west, east, south, north}
    constexpr auto Ns = BC::NeumannStag;
    BCs<dim> bcs{Ns, Ns, Ns, Ns};

    // Initialize the solver
    Solver<Prec, dim> solver{
        ns, lens, bcs, FiniteDifference2Approximation<Prec>{}};

    // Grid will have one ghost cell on every side
    // One axis will therefore have two ghost cells
    // (one on the left, one on the right)
    GridSizes<dim> ngs{1, 1};

    // The buffers hold the grid and the ghost cells on both sides
    std::size_t buffer_size{1};
    for (std::size_t d{0}; d < dim; ++d) {
        buffer_size *= ns[d] + 2 * ngs[d];
    }

    const std::size_t stride{ns[0] + 2 * ngs[0]};

    // The index of the grid cell (i, j) inside a padded buffer
    auto idx = [stride, ngs](std::size_t i, std::size_t j) {
        return (i + ngs[0]) + stride * (j + ngs[1]);
    };

    // Some arbitrary input values
    // clang-format off
    constexpr std::array<Prec, side_size * side_size> values{
         0.42, -1.17,  0.83,  0.05,
        -0.61,  1.34, -0.27,  0.92,
         0.18, -0.74,  1.05, -0.39,
         0.66, -0.21, -1.02,  0.31};
    // clang-format on

    // Delta phi = b is solvable only if b has zero mean.
    // The random values above do not, so subtract their
    // mean before solving.
    Prec mean{0.0};
    for (const Prec v : values) {
        mean += v;
    }
    mean /= static_cast<Prec>(values.size());

    // Prepare input
    std::vector<Prec> rhs(buffer_size, 0.0);
    for (std::size_t j{0}; j < ns[1]; ++j) {
        for (std::size_t i{0}; i < ns[0]; ++i) {
            rhs[idx(i, j)] = values[i + side_size * j] - mean;
        }
    }

    // Prepare result buffer
    std::vector<Prec> phi(buffer_size, 0.0);

    // Solve the equation
    // (phi and rhs have the same amount of ghost cells)
    solver.Execute(phi, rhs, ngs, ngs);

    std::cout << std::fixed << std::setprecision(6);

    const auto print_grid = [&](std::string_view label,
                                const std::vector<Prec> &buffer) {
        std::cout << label << " (interior " << ns[0] << "x" << ns[1] << "):\n";
        for (std::size_t j{0}; j < ns[1]; ++j) {
            for (std::size_t i{0}; i < ns[0]; ++i) {
                std::cout << std::setw(10) << buffer[idx(i, j)];
            }
            std::cout << "\n";
        }
    };

    print_grid("rhs", rhs);
    print_grid("phi", phi);

    // Finalization handled via RAII

    std::cout << "Successfully finished example local_with_pretty_print\n";
}
