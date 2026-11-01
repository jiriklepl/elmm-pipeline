#ifndef BOUNDARY_CONDITIONS_HPP
#define BOUNDARY_CONDITIONS_HPP

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>

/**
 * @file
 * @brief Definition of boundary conditions and related helper functions.
 */

namespace poisson_solver {
// === Enums ===
enum class BoundaryCondition {
    Periodic,
    Dirichlet,
    Neumann,
    DirichletStag,
    NeumannStag
};

// === Structs ===
struct BCPair {
    BoundaryCondition first{};
    BoundaryCondition second{};

    auto operator==(const BCPair &) const -> bool = default;
};

// === Aliases ===
using BC = BoundaryCondition;

template <std::size_t Dim>
using BCs = std::array<BCPair, Dim>;
}  // namespace poisson_solver

namespace poisson_solver::detail {
// === Functions ===
constexpr auto IsPeriodic(BCPair bcp) -> bool {
    return bcp.first == BC::Periodic && bcp.second == BC::Periodic;
}

constexpr auto IsDirichlet(BCPair bcp) -> bool {
    return bcp.first == BC::Dirichlet && bcp.second == BC::Dirichlet;
}

constexpr auto IsNeumann(BCPair bcp) -> bool {
    return bcp.first == BC::Neumann && bcp.second == BC::Neumann;
}

constexpr auto IsDirichletStag(BCPair bcp) -> bool {
    return bcp.first == BC::DirichletStag && bcp.second == BC::DirichletStag;
}

constexpr auto IsNeumannStag(BCPair bcp) -> bool {
    return bcp.first == BC::NeumannStag && bcp.second == BC::NeumannStag;
}

constexpr auto IsDirichletNeumann(BCPair bcp) -> bool {
    return bcp.first == BC::Dirichlet && bcp.second == BC::Neumann;
}

constexpr auto IsNeumannDirichlet(BCPair bcp) -> bool {
    return bcp.first == BC::Neumann && bcp.second == BC::Dirichlet;
}

constexpr auto IsDirichletStagNeumannStag(BCPair bcp) -> bool {
    return bcp.first == BC::DirichletStag && bcp.second == BC::NeumannStag;
}

constexpr auto IsNeumannStagDirichletStag(BCPair bcp) -> bool {
    return bcp.first == BC::NeumannStag && bcp.second == BC::DirichletStag;
}

// (D, D) or (DS, DS)
constexpr auto IsDirichletOrStag(BCPair bcp) -> bool {
    return IsDirichlet(bcp) || IsDirichletStag(bcp);
}

// (N, N) or (NS, NS)
constexpr auto IsNeumannOrStag(BCPair bcp) -> bool {
    return IsNeumann(bcp) || IsNeumannStag(bcp);
}

constexpr auto HasPeriodic(BCPair bcp) -> bool {
    return bcp.first == BC::Periodic || bcp.second == BC::Periodic;
}

constexpr auto HasDirichlet(BCPair bcp) -> bool {
    return bcp.first == BC::Dirichlet || bcp.second == BC::Dirichlet;
}

constexpr auto HasDirichletStag(BCPair bcp) -> bool {
    return bcp.first == BC::DirichletStag || bcp.second == BC::DirichletStag;
}

constexpr auto HasDirichletOrStag(BCPair bcp) -> bool {
    return HasDirichlet(bcp) || HasDirichletStag(bcp);
}

constexpr auto ToString(BC bc) -> std::string {
    // This could probably be done in a prettier
    // and more robust way but it's not really
    // a part of core functionality
    // so I'll leave it like this.
    switch (bc) {
        case BoundaryCondition::Periodic:
            return "P";
        case BoundaryCondition::Dirichlet:
            return "D";
        case BoundaryCondition::Neumann:
            return "N";
        case BoundaryCondition::DirichletStag:
            return "Ds";
        case BoundaryCondition::NeumannStag:
            return "Ns";
        default:
            assert(false && "Should be unreachable");
    }
    return "WE SHOULD NEVER GET HERE!";
}
}  // namespace poisson_solver::detail

#endif  // BOUNDARY_CONDITIONS_HPP
