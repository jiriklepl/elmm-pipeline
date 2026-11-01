#ifdef POISSON_SOLVER_USE_MPI
#include "poisson-solver/mpi_utils.hpp"

#include "poisson-solver/detail/cuch.hpp"
#include "poisson-solver/detail/mpi_check.hpp"

#include <cstddef>
#include <functional>
#include <heffte.h>
#include <mpi.h>
#include <numeric>
#include <stdexcept>
#include <string>

namespace poisson_solver::mpi_utils {
auto ConvertToLocalGridInfo(const heffte::box3d<> &box)
    -> LocalGridInfo<3> {
    LocalGridInfo<3> lgi{};

    for (std::size_t d{0}; d < 3; ++d) {
        lgi.ns[d] = box.size[d];
        lgi.offs[d] = box.low[d];
    }

    return lgi;
}

auto GetProcGrid(MPI_Comm cart_comm) -> ProcsPerDim<3> {
    ProcsPerDim<3> nps_zyx{};
    std::array<int, 3> periods{};
    std::array<int, 3> coords{};

    // Extract process grid
    MPI_CHECK(MPI_Cart_get(
        cart_comm, 3, nps_zyx.data(), periods.data(), coords.data()));

    // MPI returns {Z, Y, X}, convert it to {X, Y, Z}
    return ProcsPerDim<3>{nps_zyx[2], nps_zyx[1], nps_zyx[0]};
}

void CheckCartCommCanBeCreated(MPI_Comm comm, ProcsPerDim<2> nps) {
    auto size{0};
    MPI_CHECK(MPI_Comm_size(comm, &size));
    if (std::accumulate(
            nps.begin(), nps.end(), std::size_t{1}, std::multiplies<>()) !=
        static_cast<std::size_t>(size)) {
        throw std::invalid_argument(
            "Communicator size differs from the product of all processes "
            "into which the communicator should be decomposed");
    }
}

auto InitMPIGrid(MPI_Comm comm_old, ProcsPerDim<2> nps_yz) -> MPI_Comm {
    CheckCartCommCanBeCreated(comm_old, nps_yz);

    MPI_Comm cart_comm{};
    // MPIs fastest changing dimensions is last, ours is first
    ProcsPerDim<3> nps_xyz{nps_yz[1], nps_yz[0], 1};
    std::array<int, 3> periods{false, false, false};

    MPI_CHECK(MPI_Cart_create(
        comm_old, 3, nps_xyz.data(), periods.data(), false, &cart_comm));

    return cart_comm;
}

auto GetMyBox(MPI_Comm comm,
              const std::vector<heffte::box3d<>> &boxes) -> heffte::box3d<> {
    int rank;
    MPI_CHECK(MPI_Comm_rank(comm, &rank));

    return boxes[rank];
}

auto GetLocalGridInfo(GridSizes<3> gns, MPI_Comm cart_comm) -> LocalGridInfo<3> {
    auto nps = GetProcGrid(cart_comm);

    heffte::box3d<> world = {{0, 0, 0},
                             {static_cast<int>(gns[0]) - 1,
                              static_cast<int>(gns[1]) - 1,
                              static_cast<int>(gns[2]) - 1}};

    auto boxes = heffte::split_world(world, nps);

    auto my_box = GetMyBox(cart_comm, boxes);

    return ConvertToLocalGridInfo(my_box);
}

auto SelectLocalDevice(MPI_Comm comm) -> int {
    MPI_Comm local_comm{};
    MPI_CHECK(MPI_Comm_split_type(
        comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm));

    int local_rank{};
    MPI_CHECK(MPI_Comm_rank(local_comm, &local_rank));
    MPI_CHECK(MPI_Comm_free(&local_comm));

    int device_count{};
    CUCH(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        throw std::runtime_error(
                "No CUDA devices are visible to this process.");
    }

    int device = local_rank % device_count;
    CUCH(cudaSetDevice(device));

    return device;
}
}  // namespace poisson_solver::mpi_utils

#endif  // POISSON_SOLVER_USE_MPI
