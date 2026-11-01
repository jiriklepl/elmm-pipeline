#ifndef RESHAPER_HPP
#define RESHAPER_HPP

#include "poisson-solver/aliases.hpp"
#include "poisson-solver/detail/helpers.hpp"
#include "poisson-solver/mpi_utils.hpp"

#include <concepts>
#include <cstddef>
#include <functional>
#include <memory>
#include <numeric>
#include <utility>

#ifdef POISSON_SOLVER_USE_MPI
#include "poisson-solver/detail/cuch.hpp"

#include <complex>
#include <mpi.h>
#include <vector>
#endif  // POISSON_SOLVER_USE_MPI
/**
 * @file
 * @brief Reshapers provide a simple interface for exchanging parts of
 * a distributed grid.
 */

namespace poisson_solver::detail {
// === Enums ===
enum class ReshapeType { ToPencils, ToSlabs };

// === Interfaces ===
/**
 * @brief Performs reshapes from the reshape pipeline he was
 * constructed from.
 *
 * Specifically, the reshape calls first follow the reshapes from the pipeline
 * and then reshape back into the original grid.
 * If our pipeline was OriginalGrid, ToPencils, ToSlabs, the reshape calls
 * would perform: OriginalGrid->Pencils, Pencils->Slabs, Slabs->Pencils,
 * Pencils->OriginalGrid (and then looped back into OriginalGrid->Pencils, ...).
 *
 * Each ReshapeReal() or ReshapeCmplx() call performs the next reshape from
 * the pipeline and returns the LocalGridInfo of the new local grid.
 * These methods differ only in the grid element types. The user is
 * responsible for calling the correct method on his grid (i.e., calling
 * ReshapeReal() on a complex grid will likely crash).
 */
template <std::floating_point T, std::size_t Dim>
class IReshaper {
public:
    virtual ~IReshaper() = default;

    /**
     * @param dest Device pointer where the reshaped grid shall be.
     * @param src Device pointer where the original grid is.
     */
    virtual auto ReshapeReal(T *dest, const T *src) -> mpi_utils::LocalGridInfo<Dim> = 0;

    /**
     * @param dest Device pointer where the reshaped grid shall be.
     * @param src Device pointer where the original grid is.
     */
    virtual auto ReshapeCmplx(CuComplexTrait<T> *dest,
                              const CuComplexTrait<T> *src) -> mpi_utils::LocalGridInfo<Dim> = 0;
    virtual auto MaxLocalElements() const -> std::size_t = 0;

    /**
     * @brief Returns the local grid layout of
     * the grid when reshaped into pencils.
     */
    virtual auto PencilLayout() const -> mpi_utils::LocalGridInfo<Dim> = 0;

    /**
     * @brief Returns the local grid layout of
     * the grid when reshaped into slabs.
     */
    virtual auto SlabLayout() const -> mpi_utils::LocalGridInfo<Dim> = 0;
};

/**
 * The user first builds a pipeline and specified the order of reshapes
 * from the original grid into pencils (for 1D FFTs) and slabs (for 2D FFTs).
 * There must be exactly one reshape to pencils and one to slabs.
 * By calling Build(), an appropriate reshaper is returned.
 */
template <std::floating_point T, std::size_t Dim>
class IReshapePipeline {
public:
    virtual ~IReshapePipeline() = default;
    virtual auto ToPencils(std::size_t dimension) -> IReshapePipeline & = 0;
    virtual auto ToSlabs(std::size_t dimension1,
                         std::size_t dimension2) -> IReshapePipeline & = 0;
    virtual auto Build() -> std::unique_ptr<IReshaper<T, Dim>> = 0;
};

// === Local Forward Declarations ===
template <std::floating_point T, std::size_t Dim>
class DummyReshaper;

// === Local Implementations ===
/**
 * @brief Acts as a sink of build operatins (i.e., does no-op) so the
 * source code of distributed vs local solver doesn't need to change
 * but the local solver buids no reshape pipeline and reshapes nothing.
 */
template <std::floating_point T, std::size_t Dim>
class DummyReshapePipeline : public IReshapePipeline<T, Dim> {
private:
    /**
     * struct ConstructorKey - Key to the ctor.
     */
    struct ConstructorKey {};

public:
    friend class DummyReshaper<T, Dim>;

    /**
     * @brief Publicly exposed private ctor.
     * Only callable from inside of this class using ConstructorKey.
     */
    DummyReshapePipeline(ConstructorKey /*unused*/,
                         GridSizes<Dim> ns,
                         GridOffsets<Dim> offs)
        : ns_{ns},
          offs_{offs} {}

    static auto From(GridSizes<Dim> ns)
        -> std::unique_ptr<IReshapePipeline<T, Dim>> {
        return std::make_unique<DummyReshapePipeline<T, Dim>>(
            ConstructorKey{}, ns, GridOffsets<Dim>{});
    }

    static auto From(GridSizes<Dim> ns, GridOffsets<Dim> offs)
        -> std::unique_ptr<IReshapePipeline<T, Dim>> {
        return std::make_unique<DummyReshapePipeline<T, Dim>>(
            ConstructorKey{}, ns, offs);
    }

    auto ToPencils([[maybe_unused]] std::size_t dimension)
        -> DummyReshapePipeline & override {
        // nop
        return *this;
    }

    auto ToSlabs([[maybe_unused]] std::size_t dimension1,
                 [[maybe_unused]] std::size_t dimension2)
        -> DummyReshapePipeline & override {
        // nop
        return *this;
    }

    auto Build() -> std::unique_ptr<IReshaper<T, Dim>> override {
        return std::make_unique<DummyReshaper<T, Dim>>(std::move(*this));
    }

private:
    GridSizes<Dim> ns_;
    GridOffsets<Dim> offs_{};
};

template <std::floating_point T, std::size_t Dim>
class DummyReshaper : public IReshaper<T, Dim> {
public:
    explicit DummyReshaper(DummyReshapePipeline<T, Dim> &&rp)
        : ns_{std::move(rp.ns_)},
          offs_{std::move(rp.offs_)} {};

    auto ReshapeReal([[maybe_unused]] T *dest,
                     [[maybe_unused]] const T *src) -> mpi_utils::LocalGridInfo<Dim> override {
        // nop if aliased
        if (src != dest) {
            CUCH(cudaMemcpyAsync(dest,
                            src,
                            MaxLocalElements() * sizeof(T),
                            cudaMemcpyDeviceToDevice, detail::execution_stream));
        }
        return {ns_, offs_};
    }

    auto ReshapeCmplx([[maybe_unused]] CuComplexTrait<T> *dest,
                      [[maybe_unused]] const CuComplexTrait<T> *src)
        -> mpi_utils::LocalGridInfo<Dim> override {
        // nop if aliased
        if (src != dest) {
            CUCH(cudaMemcpyAsync(dest,
                            src,
                            MaxLocalElements() * sizeof(CuComplexTrait<T>),
                            cudaMemcpyDeviceToDevice, detail::execution_stream));
        }
        return {ns_, offs_};
    }

    auto MaxLocalElements() const -> std::size_t override {
        return std::accumulate(
            ns_.begin(), ns_.end(), std::size_t{1}, std::multiplies<>());
    }

    auto PencilLayout() const -> mpi_utils::LocalGridInfo<Dim> override {
        return mpi_utils::LocalGridInfo<Dim>{};
    }

    auto SlabLayout() const -> mpi_utils::LocalGridInfo<Dim> override {
        return mpi_utils::LocalGridInfo<Dim>{};
    }

private:
    GridSizes<Dim> ns_{};
    GridOffsets<Dim> offs_{};
};

#ifdef POISSON_SOLVER_USE_MPI
// === Distributed Forward Declarations ===
template <std::floating_point T>
class Reshaper;

// === Distributed Helper Structs ===
template <typename Workspace>
struct HeFFTeWorkspaceTraitImpl;

template <>
struct HeFFTeWorkspaceTraitImpl<float> {
    using Type = float;
};

template <>
struct HeFFTeWorkspaceTraitImpl<CuComplexTrait<float>> {
    using Type = std::complex<float>;
};

template <>
struct HeFFTeWorkspaceTraitImpl<double> {
    using Type = double;
};

template <>
struct HeFFTeWorkspaceTraitImpl<CuComplexTrait<double>> {
    using Type = std::complex<double>;
};

// === Aliases ===
template <typename Workspace>
using HeFFTeWorkspaceTrait = HeFFTeWorkspaceTraitImpl<Workspace>::Type;

// === Distributed Implementations ===
template <std::floating_point T, std::size_t Dim>
class ReshapePipeline : public IReshapePipeline<T, Dim> {
private:
    /**
     * struct ConstructorKey - Key to the ctor.
     */
    struct ConstructorKey {};

    using Decomposition = std::vector<heffte::box3d<>>;

public:
    friend class Reshaper<T>;

    /**
     * @brief Publicly exposed private ctor.
     * Only callable from inside of this class using ConstructorKey.
     */
    ReshapePipeline(ConstructorKey,
                    Decomposition &&original_boxes,
                    mpi_utils::ProcsPerDim<2> nps_yz,
                    MPI_Comm comm)
        : world_(heffte::find_world(original_boxes)),
          nps_yz_{nps_yz},
          comm_{comm} {
        MPI_CHECK(MPI_Comm_rank(comm, &rank_));
        decompositions_.emplace_back(std::move(original_boxes));
    }

    static auto From(GridSizes<Dim> ns,
                     GridOffsets<Dim> offs,
                     MPI_Comm cart_comm,
                     bool is_full_solver)
        -> std::unique_ptr<IReshapePipeline<T, Dim>> {
        if constexpr (Dim == 1 || Dim == 2) {
            return DummyReshapePipeline<T, Dim>::From(ns, offs);
        } else {
            if (is_full_solver) {
                return DummyReshapePipeline<T, Dim>::From(ns, offs);
            }
            heffte::box3d<> my_box = GetGridBox(ns, offs);
            auto all_boxes =
                heffte::mpi::gather_boxes(my_box, my_box, cart_comm).in;

            auto nps_xyz = mpi_utils::GetProcGrid(cart_comm);
            mpi_utils::ProcsPerDim<2> nps_yz = {nps_xyz[1], nps_xyz[2]};

            return std::make_unique<ReshapePipeline<T, 3>>(
                ConstructorKey{}, std::move(all_boxes), nps_yz, cart_comm);
        }
    }

    auto ToPencils(std::size_t dimension) -> ReshapePipeline & override {
        if (contains_reshape_to_pencils_) {
            throw std::runtime_error(
                "Cannot reshape to pencils more than once.");
        }
        contains_reshape_to_pencils_ = true;

        // Resulting decomp by reshaping last decomp into pencils
        // in the direction of `dimension`
        decompositions_.emplace_back(heffte::make_pencils(
            world_, nps_yz_, dimension, decompositions_.back(), {0, 1, 2}));

        auto my_pencil = decompositions_.back()[rank_];
        auto my_pencil_sizes = my_pencil.size;
        auto my_pencil_offs = my_pencil.low;
        pencil_sizes_ = {static_cast<std::size_t>(my_pencil_sizes[0]),
                         static_cast<std::size_t>(my_pencil_sizes[1]),
                         static_cast<std::size_t>(my_pencil_sizes[2])};
        pencil_offs_ = {static_cast<std::size_t>(my_pencil_offs[0]),
                        static_cast<std::size_t>(my_pencil_offs[1]),
                        static_cast<std::size_t>(my_pencil_offs[2])};

        return *this;
    }

    auto ToSlabs(std::size_t dimension1,
                 std::size_t dimension2) -> ReshapePipeline & override {
        if (contains_reshape_to_slabs_) {
            throw std::runtime_error("Cannot reshape to slabs more than once.");
        }
        contains_reshape_to_slabs_ = true;

        auto num_procs = nps_yz_[0] * nps_yz_[1];

        // Resulting decomp by reshaping last decomp into slabs
        // that span the whole `dimension1` x `dimension2`
        decompositions_.emplace_back(heffte::make_slabs(world_,
                                                        num_procs,
                                                        dimension1,
                                                        dimension2,
                                                        decompositions_.back(),
                                                        {0, 1, 2},
                                                        heffte::rank_remap{}));

        auto my_slab = decompositions_.back()[rank_];
        auto my_slab_sizes = my_slab.size;
        auto my_slab_offs = my_slab.low;
        slab_sizes_ = {static_cast<std::size_t>(my_slab_sizes[0]),
                       static_cast<std::size_t>(my_slab_sizes[1]),
                       static_cast<std::size_t>(my_slab_sizes[2])};
        slab_offs_ = {static_cast<std::size_t>(my_slab_offs[0]),
                      static_cast<std::size_t>(my_slab_offs[1]),
                      static_cast<std::size_t>(my_slab_offs[2])};

        return *this;
    }

    auto Build() -> std::unique_ptr<IReshaper<T, Dim>> override {
        if (!(contains_reshape_to_pencils_ && contains_reshape_to_slabs_)) {
            throw std::runtime_error(
                "Not all reshapes were specified in the pipeline (pencil, slab "
                "or both missing).");
        }
        return std::make_unique<Reshaper<T>>(std::move(*this));
    }

private:
    bool contains_reshape_to_pencils_{false};
    bool contains_reshape_to_slabs_{false};

    GridSizes<Dim> pencil_sizes_{};
    GridOffsets<Dim> pencil_offs_{};
    GridSizes<Dim> slab_sizes_{};
    GridOffsets<Dim> slab_offs_{};

    heffte::box3d<> world_;
    mpi_utils::ProcsPerDim<2> nps_yz_{};

    MPI_Comm comm_;
    int rank_{-1};

    std::vector<Decomposition> decompositions_{};
};

template <std::floating_point T>
class Reshaper : public IReshaper<T, 3> {
public:
    Reshaper(ReshapePipeline<T, 3> &&rp)
        : rp_{std::move(rp)} {
        AssignReshapes(rp_.decompositions_);

        // std::complex<double> is the largest possible type
        // we can receive, so this is definitely enough but in some
        // cases this much is not needed (std::complex<float>, double, float).
        workspace_ = MakeCudaUnique<std::byte>(MaxWorkspaceSize(reshapes_) *
                                               sizeof(std::complex<double>));
    };

    auto ReshapeReal(T *dest, const T *src) -> mpi_utils::LocalGridInfo<3> override {
        ApplyReshape(dest, src);

        auto &my_new_box =
            rp_.decompositions_[next_reshape_box_idx_][rp_.rank_];

        UpdateIdxs();

        GridSizes<3> new_ns = {static_cast<std::size_t>(my_new_box.size[0]),
                               static_cast<std::size_t>(my_new_box.size[1]),
                               static_cast<std::size_t>(my_new_box.size[2])};
        GridOffsets<3> new_offs = {static_cast<std::size_t>(my_new_box.low[0]),
                                   static_cast<std::size_t>(my_new_box.low[1]),
                                   static_cast<std::size_t>(my_new_box.low[2])};
        return mpi_utils::LocalGridInfo{new_ns, new_offs};
    }

    auto ReshapeCmplx(CuComplexTrait<T> *dest,
                      const CuComplexTrait<T> *src) -> mpi_utils::LocalGridInfo<3> override {
        ApplyReshape(dest, src);

        auto &my_new_box =
            rp_.decompositions_[next_reshape_box_idx_][rp_.rank_];

        UpdateIdxs();

        GridSizes<3> new_ns = {static_cast<std::size_t>(my_new_box.size[0]),
                               static_cast<std::size_t>(my_new_box.size[1]),
                               static_cast<std::size_t>(my_new_box.size[2])};
        GridOffsets<3> new_offs = {static_cast<std::size_t>(my_new_box.low[0]),
                                   static_cast<std::size_t>(my_new_box.low[1]),
                                   static_cast<std::size_t>(my_new_box.low[2])};
        return mpi_utils::LocalGridInfo{new_ns, new_offs};
    }

    auto MaxLocalElements() const -> std::size_t override {
        std::size_t max{0};

        for (const auto &decomp : rp_.decompositions_) {
            max = std::max(max,
                           static_cast<std::size_t>(decomp[rp_.rank_].count()));
        }

        return max;
    }

    auto PencilLayout() const -> mpi_utils::LocalGridInfo<3> override {
        return {rp_.pencil_sizes_, rp_.pencil_offs_};
    }

    auto SlabLayout() const -> mpi_utils::LocalGridInfo<3> override {
        return {rp_.slab_sizes_, rp_.slab_offs_};
    }

private:
    using Reshapes = std::vector<std::unique_ptr<heffte::reshape3d_base<int>>>;

    ReshapePipeline<T, 3> rp_;
    Reshapes reshapes_{};
    std::size_t next_reshape_idx_{0};
    std::size_t next_reshape_box_idx_{1};

    CudaUniquePtr<std::byte> workspace_{};

    void AssignReshapes(
        const std::vector<typename ReshapePipeline<T, 3>::Decomposition>
            &decomps) {
        using BackendTag = heffte::backend::cufft;

        // For both forward reshapes and backward reshapes
        reshapes_.resize(2 * (decomps.size() - 1));

        auto options = heffte::default_options<BackendTag>();
        options.algorithm = heffte::reshape_algorithm::p2p_plined;

        for (std::size_t i{0}; i < decomps.size() - 1; ++i) {
            auto decomp_from = decomps[i];
            auto decomp_to = decomps[i + 1];

            // Forward reshape
            reshapes_[i] = heffte::make_reshape3d<BackendTag>(
                nullptr, decomp_from, decomp_to, rp_.comm_, options);

            // Backward reshape
            reshapes_[reshapes_.size() - 1 - i] =
                heffte::make_reshape3d<BackendTag>(
                    nullptr, decomp_to, decomp_from, rp_.comm_, options);
        }
    }

    auto MaxWorkspaceSize(const Reshapes &reshapes) const -> std::size_t {
        std::size_t max{0};

        for (const auto &reshape : reshapes) {
            if (reshape) {
                max = std::max(max, reshape->size_workspace());
            }
        }

        return max;
    }

    template <ValidWorkspace<T> Workspace>
    void ApplyReshape(Workspace *dest, const Workspace *src) {
        auto &next_reshape = reshapes_[next_reshape_idx_];
        if (next_reshape) {
            next_reshape->apply(
                1,
                reinterpret_cast<const HeFFTeWorkspaceTrait<Workspace> *>(src),
                reinterpret_cast<HeFFTeWorkspaceTrait<Workspace> *>(dest),
                reinterpret_cast<HeFFTeWorkspaceTrait<Workspace> *>(
                    workspace_.get()));
        } else if (src != dest) {
            // If no transformation (not even transposition) is
            // required, this branch is taken.
            // In that case all the data is already
            // locally available in the same shape and size.
            auto my_new_box =
                rp_.decompositions_[next_reshape_box_idx_][rp_.rank_];
            auto elems_to_copy = static_cast<std::size_t>(my_new_box.count());
            CUCH(cudaMemcpyAsync(dest,
                            src,
                            elems_to_copy * sizeof(Workspace),
                            cudaMemcpyDeviceToDevice, detail::execution_stream));
        }
    }

    void UpdateIdxs() {
        bool next_reshape_is_forward =
            next_reshape_idx_ < (reshapes_.size() / 2) - 1;
        bool next_reshape_is_the_last =
            next_reshape_idx_ == reshapes_.size() - 1;

        if (next_reshape_is_forward || next_reshape_is_the_last) {
            next_reshape_box_idx_++;
        } else {
            next_reshape_box_idx_--;
        }
        next_reshape_idx_++;
        next_reshape_idx_ %= reshapes_.size();
    }
};
#endif  // POISSON_SOLVER_USE_MPI
}  // namespace poisson_solver::detail
#endif  // RESHAPER_HPP
