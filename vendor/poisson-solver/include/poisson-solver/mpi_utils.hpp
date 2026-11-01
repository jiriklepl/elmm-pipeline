#ifndef MPI_UTILS_HPP
#define MPI_UTILS_HPP
#include "poisson-solver/aliases.hpp"
#include "poisson-solver/detail/helpers.hpp"

#include <array>
#include <cstddef>

namespace poisson_solver::mpi_utils {
// === Aliases ===
template <std::size_t Dim>
using ProcsPerDim = std::array<int, Dim>;

// === Structs ===
/**
 * struct LocalGridInfo - Stores information about local grid.
 * This includes its size and its offset from the origin.
 */
template <std::size_t Dim>
struct LocalGridInfo {
    GridSizes<Dim> ns{};
    GridOffsets<Dim> offs{};
};
}  // namespace poisson_solver::mpi_utils

#ifdef POISSON_SOLVER_USE_MPI
#include <mpi.h>

namespace poisson_solver::mpi_utils {

// === Functions ===
/**
 * @brief Retrieves process grid of the communicator.
 *
 * @param cart_comm Communicator created via a call to InitMPIGrid.
 * @return Number of processes in each dimension.
 */
auto GetProcGrid(MPI_Comm cart_comm) -> ProcsPerDim<3>;

/**
 * @brief Initializes MPI cartesian grid
 * with at most two axes of decomposition (the X axis is local).
 *
 * @param comm_old The old communicator.
 * @param nps Desired number of processes per dimension.
 * nps[0] corresponds to the Y axis and nps[1] to the Z axis.
 * For 1D decomposition, leave the non-decomposed axis as 1.
 * @return Cartesian grid with the specified decomposition.
 */
auto InitMPIGrid(MPI_Comm comm_old, ProcsPerDim<2> nps_yz) -> MPI_Comm;

/**
 * @brief Retrieves information about the local grid.
 *
 * @param gns Global sizes of the grid in each dimension.
 * @param cart_comm Communicator created via a call to InitMPIGrid.
 * @return Information about the local grid.
 */
auto GetLocalGridInfo(GridSizes<3> gns, MPI_Comm cart_comm) -> LocalGridInfo<3>;

/**
 * @brief Selects one CUDA device per rank based on this process's rank
 * within its node, and makes it the active device via cudaSetDevice().
 *
 * @param comm A communicator containing (at least) all ranks that share
 * this node.
 * @return The CUDA device ordinal selected for this rank.
 */
auto SelectLocalDevice(MPI_Comm comm) -> int;
}  // namespace poisson_solver::mpi_utils

#endif  // POISSON_SOLVER_USE_MPI
#endif  // MPI_UTILS_HPP
