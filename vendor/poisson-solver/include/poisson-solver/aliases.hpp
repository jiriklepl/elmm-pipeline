#ifndef ALIASES_HPP
#define ALIASES_HPP

#include <array>
#include <cstddef>

/**
 * @file
 * @brief Aliases utilized in the library C++ API.
 */

namespace poisson_solver {
template <std::size_t Dim>
using GridSizes = std::array<std::size_t, Dim>;

template <std::size_t Dim>
using GridOffsets = std::array<std::size_t, Dim>;
}  // namespace poisson_solver

#endif  // ALIASES_HPP
