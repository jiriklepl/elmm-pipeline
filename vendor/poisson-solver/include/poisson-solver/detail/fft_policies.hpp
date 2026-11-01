#ifndef FFT_POLICIES_HPP
#define FFT_POLICIES_HPP

#include "poisson-solver/aliases.hpp"
#include "poisson-solver/detail/cuch.hpp"
#include "poisson-solver/detail/helpers.hpp"

#include <cassert>
#include <concepts>
#include <cstddef>
#include <cuda.h>
#include <cuda_runtime.h>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <variant>
#include "poisson-solver/detail/nvrtc_cache.hpp"
// Account every device buffer allocated by the pinned VkFFT headers, including
// LUT/Bluestein scratch. Driver context/module/code allocations are excluded.
#define cudaMalloc ::poisson_solver::detail::DeviceAllocate
#define cudaFree ::poisson_solver::detail::DeviceFree
#define nvrtcCreateProgram ::poisson_solver::detail::nvrtc_cache::create
#define nvrtcCompileProgram ::poisson_solver::detail::nvrtc_cache::compile
#define nvrtcGetCUBINSize ::poisson_solver::detail::nvrtc_cache::cubin_size
#define nvrtcGetCUBIN ::poisson_solver::detail::nvrtc_cache::cubin
#define nvrtcDestroyProgram ::poisson_solver::detail::nvrtc_cache::destroy
#include <vkFFT.h>
#undef nvrtcCreateProgram
#undef nvrtcCompileProgram
#undef nvrtcGetCUBINSize
#undef nvrtcGetCUBIN
#undef nvrtcDestroyProgram
#undef cudaFree
#undef cudaMalloc

#ifdef POISSON_SOLVER_USE_MPI
#include "poisson-solver/detail/mpi_check.hpp"

#include <algorithm>
#include <array>
#include <complex>
#include <heffte.h>
#include <mpi.h>
#include <utility>
#endif  // POISSON_SOLVER_USE_MPI

/**
 * @file
 * @brief Classes providing API for forward and backward FFTs.
 *
 * Furthermore, there is API for forward and backward
 * FFTs only in specific dimensions (this is required in some solvers).
 *
 * The FFTPolicy is an alias of the current FFT policy based on the
 * use of POISSON_SOLVER_USE_MPI.
 */

namespace poisson_solver::detail {
// === Enums ===
enum class FFTTransformType {
    Complex,

    DCT1,
    DCT2,
    DCT3,
    DCT4,

    DST1,
    DST2,
    DST3,
    DST4
};

// === Forward Declarations ===
struct FullLayout;
struct SplitLayout;

struct VkFFTFullHandle;
struct VkFFTSplitHandle;

// === Aliases ===
using VariantLayout = std::variant<std::monostate, FullLayout, SplitLayout>;

using VkFFTVariantHandle =
    std::variant<std::monostate, VkFFTFullHandle, VkFFTSplitHandle>;

// === Structs ===
struct VkFFTApplicationDeleter {
    void operator()(VkFFTApplication *app) const {
        // It's possible that `app` wasn't even initialized, calling
        // `deleteVkFFT` on that could be dangerous.
        if (app != nullptr) {
            deleteVkFFT(app);
            delete app;
        }
    }
};

/**
 * struct VkFFTHandle - Is configured to perform one type of transformation.
 */
struct VkFFTHandle {
    VkFFTConfiguration config{};
    std::unique_ptr<VkFFTApplication, VkFFTApplicationDeleter> app{};
};

/**
 * struct VkFFTFullHandle - Full solvers require only one transform of the
 * whole (at most 3D) domain.
 */
struct VkFFTFullHandle {
    VkFFTHandle handle{};
};

/**
 * struct VkFFTSplitHandle - Split solvers perform two different
 * transformations.
 */
struct VkFFTSplitHandle {
    VkFFTHandle handle_1d{};
    VkFFTHandle handle_2d{};
};

struct VkFFTPlan {
    std::unique_ptr<CUdevice> cu_device{};
    std::unique_ptr<cudaStream_t> stream{};
    VkFFTVariantHandle handle{};
};

// Holds values of `performDCT` and `performDST`
// used in VkFFT configuration.
struct VkFFTTypeToPerform {
    std::size_t dct{};
    std::size_t dst{};
};

// === Layout Structs ===
// Different layouts of FFTs are used based on BCs.
// Only the forward types are stored thanks
// to libraries handling the backward types automatically.

struct FullLayout {
    FFTTransformType forward_type{};
};

struct SplitLayout {
    FFTTransformType forward_type_1d{};
    FFTTransformType forward_type_2d{};

    // 0 = X, 1 = Y, 2 = Z
    std::size_t isolated_dim_idx{};
};

// === Functions and Macros ===
inline void VkFFTCheck(VkFFTResult res,
                       int line,
                       const char *src_file,
                       const char *err_msg = nullptr) {
    if (res != VKFFT_SUCCESS) {
        std::stringstream err_stream{};
        err_stream << "VkFFT Error (" << res << ")\n"
                   << "at " << src_file << "[" << line << "]:" << err_msg;
        throw std::runtime_error(err_stream.str());
    }
}

#define VKFFT_CHECK(res) \
    ::poisson_solver::detail::VkFFTCheck(res, __LINE__, __FILE__, #res)

// === Classes ===
/**
 * struct VkFFTPolicy - Local FFT policy (transformations on local grids only).
 *
 * @tparam T Float type.
 * @tparam Dim Number of grid dimensions transformed via the FFT.
 */
template <std::floating_point T, std::size_t Dim>
struct VkFFTPolicy {
public:
    using Plan = VkFFTPlan;

    static auto Init(GridSizes<Dim> ns, VariantLayout layout) -> Plan {
        nvrtc_cache::CreationScope compile_scope;
        Plan plan{};

        // Shared among all layouts.
        plan.cu_device = GetCudaDevice();
        plan.stream = std::make_unique<cudaStream_t>(nullptr);

        std::visit(Overloaded{[&plan, ns](FullLayout l) {
                                  plan.handle = GetFullHandle(
                                      l, ns, plan.cu_device.get(), plan.stream.get());
                              },
                              [&plan, ns](SplitLayout l) {
                                  plan.handle = GetSplitHandle(
                                      l, ns, plan.cu_device.get(), plan.stream.get());
                              },
                              [](std::monostate) {
                                  assert(false && "Visited a monostate.");
                              }},
                   layout);

        return plan;
    }

    static void SetStream(Plan &plan, cudaStream_t stream) { *plan.stream = stream; }

    template <ValidWorkspace<T> Workspace>
    static void Forward(Workspace *dev_buffer, Plan &plan) {
        assert(std::holds_alternative<VkFFTFullHandle>(plan.handle));
        auto &handle = std::get<VkFFTFullHandle>(plan.handle).handle;

        VkFFTLaunchParams launch_params{};
        void *void_ptr_dev_buff = dev_buffer;
        launch_params.buffer = &void_ptr_dev_buff;

        VKFFT_CHECK(
            VkFFTAppend(handle.app.get(), forward_direction_, &launch_params));
    }

    template <ValidWorkspace<T> Workspace>
    static void Backward(Workspace *dev_buffer, Plan &plan) {
        assert(std::holds_alternative<VkFFTFullHandle>(plan.handle));
        auto &handle = std::get<VkFFTFullHandle>(plan.handle).handle;

        VkFFTLaunchParams launch_params{};
        void *void_ptr_dev_buff = dev_buffer;
        launch_params.buffer = &void_ptr_dev_buff;

        VKFFT_CHECK(
            VkFFTAppend(handle.app.get(), backward_direction_, &launch_params));
    }

    template <ValidWorkspace<T> Workspace>
    static void Forward1D(Workspace *dev_buffer, Plan &plan) {
        assert(std::holds_alternative<VkFFTSplitHandle>(plan.handle));
        auto &handle = std::get<VkFFTSplitHandle>(plan.handle).handle_1d;

        VkFFTLaunchParams launch_params{};
        void *void_ptr_dev_buff = dev_buffer;
        launch_params.buffer = &void_ptr_dev_buff;

        VKFFT_CHECK(
            VkFFTAppend(handle.app.get(), forward_direction_, &launch_params));
    }

    template <ValidWorkspace<T> Workspace>
    static void Backward1D(Workspace *dev_buffer, Plan &plan) {
        assert(std::holds_alternative<VkFFTSplitHandle>(plan.handle));
        auto &handle = std::get<VkFFTSplitHandle>(plan.handle).handle_1d;

        VkFFTLaunchParams launch_params{};
        void *void_ptr_dev_buff = dev_buffer;
        launch_params.buffer = &void_ptr_dev_buff;

        VKFFT_CHECK(
            VkFFTAppend(handle.app.get(), backward_direction_, &launch_params));
    }

    template <ValidWorkspace<T> Workspace>
    static void Forward2D(Workspace *dev_buffer, Plan &plan) {
        assert(std::holds_alternative<VkFFTSplitHandle>(plan.handle));
        auto &handle = std::get<VkFFTSplitHandle>(plan.handle).handle_2d;

        VkFFTLaunchParams launch_params{};
        void *void_ptr_dev_buff = dev_buffer;
        launch_params.buffer = &void_ptr_dev_buff;

        VKFFT_CHECK(
            VkFFTAppend(handle.app.get(), forward_direction_, &launch_params));
    }

    template <ValidWorkspace<T> Workspace>
    static void Backward2D(Workspace *dev_buffer, Plan &plan) {
        assert(std::holds_alternative<VkFFTSplitHandle>(plan.handle));
        auto &handle = std::get<VkFFTSplitHandle>(plan.handle).handle_2d;

        VkFFTLaunchParams launch_params{};
        void *void_ptr_dev_buff = dev_buffer;
        launch_params.buffer = &void_ptr_dev_buff;

        VKFFT_CHECK(
            VkFFTAppend(handle.app.get(), backward_direction_, &launch_params));
    }

private:
    static constexpr int forward_direction_ = -1;
    static constexpr int backward_direction_ = 1;

    static auto GetCudaDevice() -> std::unique_ptr<CUdevice> {
        // Ensure CUDA runtime is initialized.
        // This is essentailly a nop but if runtime isn't init, then it does it.
        cudaFree(nullptr);

        int ordinal{};
        CUCH(cudaGetDevice(&ordinal));

        auto device = std::make_unique<CUdevice>();
        cuDeviceGet(device.get(), ordinal);

        return device;
    }

    // Returns initialized full handle.
    static auto GetFullHandle(FullLayout layout,
                              GridSizes<Dim> ns,
                              CUdevice *device_ptr, cudaStream_t *stream) -> VkFFTFullHandle {
        VkFFTFullHandle h{};
        auto &config = h.handle.config;

        config.device = device_ptr;
        config.stream = stream;
        config.num_streams = 1;

        // 0 for float, 1 for doubles.
        config.doublePrecision = !std::is_same_v<T, float>;

        config.FFTdim = Dim;
        for (std::size_t d = 0; d < Dim; ++d) {
            config.size[d] = ns[d];
        }
        auto fft_type = GetVkFFTTransformType(layout.forward_type);
        config.performDCT = fft_type.dct;
        config.performDST = fft_type.dst;

        auto &app = h.handle.app;

        // Need to use `reset` because `make_unique`
        // doesn't work with custom deleters.
        app.reset(new VkFFTApplication{});

        VKFFT_CHECK(initializeVkFFT(app.get(), config));

        return h;
    };

    // Returns initialized split handle.
    static auto GetSplitHandle(SplitLayout layout,
                               GridSizes<Dim> ns,
                               CUdevice *device_ptr, cudaStream_t *stream) -> VkFFTSplitHandle {
        VkFFTSplitHandle h{};
        auto &config_1d = h.handle_1d.config;
        auto &config_2d = h.handle_2d.config;

        config_1d.device = device_ptr;
        config_2d.device = device_ptr;
        config_1d.stream = config_2d.stream = stream;
        config_1d.num_streams = config_2d.num_streams = 1;

        // 0 for float, 1 for doubles.
        bool double_prec_used = !std::is_same_v<T, float>;
        config_1d.doublePrecision = static_cast<pfUINT>(double_prec_used);
        config_2d.doublePrecision = static_cast<pfUINT>(double_prec_used);

        // Both configs will know about the
        // grid being 3D but will omit the
        // appropriate dimensions.
        config_1d.FFTdim = Dim;
        config_2d.FFTdim = Dim;
        for (std::size_t d = 0; d < Dim; ++d) {
            config_1d.size[d] = ns[d];
            config_2d.size[d] = ns[d];
        }

        switch (layout.isolated_dim_idx) {
            case 0: {
                config_1d.omitDimension[1] = 1;
                config_1d.omitDimension[2] = 1;
                config_2d.omitDimension[0] = 1;
                break;
            }
            case 1: {
                config_1d.omitDimension[0] = 1;
                config_1d.omitDimension[2] = 1;
                config_2d.omitDimension[1] = 1;
                break;
            }
            case 2: {
                config_1d.omitDimension[0] = 1;
                config_1d.omitDimension[1] = 1;
                config_2d.omitDimension[2] = 1;
                break;
            }
            default: {
                assert(false && "Invalid index of an isolated axis.");
            }
        }

        auto fft_type_1d = GetVkFFTTransformType(layout.forward_type_1d);
        auto fft_type_2d = GetVkFFTTransformType(layout.forward_type_2d);

        config_1d.performDCT = fft_type_1d.dct;
        config_1d.performDST = fft_type_1d.dst;

        config_2d.performDCT = fft_type_2d.dct;
        config_2d.performDST = fft_type_2d.dst;

        auto &app_1d = h.handle_1d.app;
        auto &app_2d = h.handle_2d.app;

        // Need to use `reset` because `make_unique`
        // doesn't work with custom deleters.
        app_1d.reset(new VkFFTApplication{});
        app_2d.reset(new VkFFTApplication{});

        VKFFT_CHECK(initializeVkFFT(app_1d.get(), config_1d));
        VKFFT_CHECK(initializeVkFFT(app_2d.get(), config_2d));

        return h;
    }

    /**
     * @brief Maps values of the FFTTransformType enum
     * to the int values used by VkFFT.
     */
    static auto GetVkFFTTransformType(FFTTransformType t)
        -> VkFFTTypeToPerform {
        using enum FFTTransformType;

        switch (t) {
            case Complex:
                return {.dct = 0, .dst = 0};
            case DCT1:
                return {.dct = 1, .dst = 0};
            case DCT2:
                return {.dct = 2, .dst = 0};
            case DCT3:
                return {.dct = 3, .dst = 0};
            case DCT4:
                return {.dct = 4, .dst = 0};
            case DST1:
                return {.dct = 0, .dst = 1};
            case DST2:
                return {.dct = 0, .dst = 2};
            case DST3:
                return {.dct = 0, .dst = 3};
            case DST4:
                return {.dct = 0, .dst = 4};
            default:
                assert(false && "Unhandled VkFFT transform type mapping.");
                return {0, 0};
        }
    }
};

#ifdef POISSON_SOLVER_USE_MPI
// === Structs ===
// We have to use the type erasure idiom with heFFTe to
// transform the compile-time tags into run-time virtual calls.

template <std::floating_point T>
class IMpiFftHandle {
public:
    virtual ~IMpiFftHandle() = default;

    // Provide default asserts to catch invalid calls
    // (e.g., calling Forward1D on a Full plan)
    virtual void Forward(void *dev_buffer) {
        assert(false && "Not a full plan");
    }

    virtual void Backward(void *dev_buffer) {
        assert(false && "Not a full plan");
    }

    virtual void Forward1D(void *dev_buffer) {
        assert(false && "Not a split plan");
    }

    virtual void Backward1D(void *dev_buffer) {
        assert(false && "Not a split plan");
    }

    virtual void Forward2D(void *dev_buffer) {
        assert(false && "Not a split plan");
    }

    virtual void Backward2D(void *dev_buffer) {
        assert(false && "Not a split plan");
    }

protected:
    using HeFFTeBackendTagVariant = std::variant<heffte::backend::cufft,
                                                 heffte::backend::cufft_cos1,
                                                 heffte::backend::cufft_cos,
                                                 heffte::backend::cufft_cos4,
                                                 heffte::backend::cufft_sin1,
                                                 heffte::backend::cufft_sin,
                                                 heffte::backend::cufft_sin4>;

    using WorkspaceVariant = std::variant<T, CuComplexTrait<T>>;

    static constexpr auto AsWorkspace(FFTTransformType fft_type)
        // TODO - Produces compiler error
        // noexcept(std::is_nothrow_default_constructible_v<T> &&
        //          std::is_nothrow_default_constructible_v<CuComplexTrait<T>>)
        -> WorkspaceVariant {
        using enum FFTTransformType;
        if (fft_type == Complex) {
            return CuComplexTrait<T>{};
        } else {
            return T{};
        }
    }

    /**
     * @brief Transforms FFTTransformType into a different entity
     * via the provided mapping in \p as_what
     * and exposes an instance of that entity
     * in the supplied \p callable.
     *
     * This allows us to call `decltype(instance)` and effectively
     * map run-time enum values into types.
     *
     * @param as_what Mapping function taking one FFTTransformType param
     * that produces the desired entity.
     * @param fft_type The FFT type to transform into a different entity.
     * @param callable Callable object taking one templated param.
     *
     * Usage example:
     * @code{.cpp}
     * WithFFTTypeAs(AsWorkspace, run_time_fft_type, [](auto instance){
     *     using WorkspaceType = decltype(instance);
     *     // Use the WorkspaceType... e.g., for casting a received void*
     * });
     * @endcode
     *
     * @see AsWorkspace()
     */
    template <typename AsWhat, typename F>
    static constexpr auto WithFFTTypeAs(AsWhat &&as_what,
                                        FFTTransformType fft_type,
                                        F &&callable) {
        // std::forward makes a difference if the operator()
        // is ref-qualified (most likely not the case so we're just being
        // good boys here)
        auto target_variant_instance = std::forward<AsWhat>(as_what)(fft_type);
        return std::visit(
            [&callable](auto target_instance) {
                return std::forward<F>(callable)(target_instance);
            },
            target_variant_instance);
    }
};

/**
 * @brief Special handle for 1D FFTs in distributed code.
 * 
 * This normally doesn't make sense as at least one dimension
 * is always continuous in a distributed solver
 * but PoisFFT implements it and so do we.
 */
template <std::floating_point T>
class MpiFft1DHandle : public IMpiFftHandle<T> {
public:
    using LocalFFTPolicy = VkFFTPolicy<T, 1>;
    using LocalFFTPlan = typename LocalFFTPolicy::Plan;

    MpiFft1DHandle(FullLayout fl,
                   GridSizes<1> ns,
                   GridOffsets<1> offs,
                   MPI_Comm comm)
        : comm_{comm},
          fl_{fl} {
        using enum FFTTransformType;

        int num_procs;
        MPI_CHECK(MPI_Comm_rank(comm_, &rank_));
        MPI_CHECK(MPI_Comm_size(comm_, &num_procs));

        std::size_t elem_size{};
        this->WithFFTTypeAs(this->AsWorkspace,
                            fl.forward_type,
                            [&](auto workspace_elem_instance) {
                                using Workspace =
                                    decltype(workspace_elem_instance);

                                elem_size = sizeof(Workspace);
                            });

        counts_.resize(num_procs);
        displs_.resize(num_procs);

        int send_count = static_cast<int>(ns[0] * elem_size);

        MPI_CHECK(MPI_Allgather(
            &send_count, 1, MPI_INT, counts_.data(), 1, MPI_INT, comm_));

        displs_[0] = 0;
        for (int i = 1; i < num_procs; ++i) {
            displs_[i] = displs_[i - 1] + counts_[i - 1];
        }

        std::size_t global_bytes = displs_.back() + counts_.back();
        std::size_t global_elems = global_bytes / elem_size;

        if (rank_ == 0) {
            workspace_ = MakeCudaUnique<std::byte>(global_bytes);
            plan_ = std::make_unique<LocalFFTPlan>(
                LocalFFTPolicy::Init(GridSizes<1>{global_elems}, fl));
        }
    }

    void Forward(void *dev_buffer) override {
        MPI_CHECK(MPI_Gatherv(dev_buffer,
                              counts_[rank_],
                              MPI_BYTE,
                              rank_ == 0 ? this->workspace_.get() : nullptr,
                              counts_.data(),
                              displs_.data(),
                              MPI_BYTE,
                              0,
                              comm_));

        if (rank_ == 0) {
            this->WithFFTTypeAs(
                this->AsWorkspace,
                fl_.forward_type,
                [&](auto workspace_elem_instance) {
                    using Workspace = decltype(workspace_elem_instance);

                    LocalFFTPolicy::Forward(
                        reinterpret_cast<Workspace *>(this->workspace_.get()),
                        *plan_);
                });
        }

        MPI_CHECK(MPI_Scatterv(rank_ == 0 ? this->workspace_.get() : nullptr,
                               counts_.data(),
                               displs_.data(),
                               MPI_BYTE,
                               dev_buffer,
                               counts_[rank_],
                               MPI_BYTE,
                               0,
                               comm_));
    }

    void Backward(void *dev_buffer) override {
        MPI_CHECK(MPI_Gatherv(dev_buffer,
                              counts_[rank_],
                              MPI_BYTE,
                              rank_ == 0 ? this->workspace_.get() : nullptr,
                              counts_.data(),
                              displs_.data(),
                              MPI_BYTE,
                              0,
                              comm_));

        if (rank_ == 0) {
            this->WithFFTTypeAs(
                this->AsWorkspace,
                fl_.forward_type,
                [&](auto workspace_elem_instance) {
                    using Workspace = decltype(workspace_elem_instance);

                    LocalFFTPolicy::Backward(
                        reinterpret_cast<Workspace *>(this->workspace_.get()),
                        *plan_);
                });
        }

        MPI_CHECK(MPI_Scatterv(rank_ == 0 ? this->workspace_.get() : nullptr,
                               counts_.data(),
                               displs_.data(),
                               MPI_BYTE,
                               dev_buffer,
                               counts_[rank_],
                               MPI_BYTE,
                               0,
                               comm_));
    }

private:
    MPI_Comm comm_;
    int rank_;

    std::unique_ptr<LocalFFTPlan> plan_{};

    std::vector<int> counts_;
    std::vector<int> displs_;

    FullLayout fl_{};

    CudaUniquePtr<std::byte> workspace_{};
};

/**
 * struct MpiFftFullHandle - Handle for full solvers.
 *
 * Uses HeFFTe to perform FFTs on the whole grid.
 *
 * @tparam TransformTag HeFFTe backend transform tag.
 */
template <std::floating_point T, typename TransformTag>
struct MpiFftFullHandle : public IMpiFftHandle<T> {
public:
    // If using C2C transform, workspace is complex, otherwise it's real
    using Workspace =
        std::conditional_t<std::is_same_v<heffte::backend::cufft, TransformTag>,
                           CuComplexTrait<T>,
                           T>;

    MpiFftFullHandle(heffte::fft3d<TransformTag> &&plan)
        : plan_{std::move(plan)},
          workspace_{MakeCudaUnique<Workspace>(plan_.size_workspace())} {}

    void Forward(void *dev_buffer) override {
        plan_.forward(reinterpret_cast<Workspace *>(dev_buffer),
                      reinterpret_cast<Workspace *>(dev_buffer),
                      reinterpret_cast<Workspace *>(workspace_.get()));
    }

    void Backward(void *dev_buffer) override {
        plan_.backward(reinterpret_cast<Workspace *>(dev_buffer),
                       reinterpret_cast<Workspace *>(dev_buffer),
                       reinterpret_cast<Workspace *>(workspace_.get()));
    }

private:
    // Has to be initialized before workspace_!
    heffte::fft3d<TransformTag> plan_{};

    // If we didn't choose to provide a workspace, heFFTe
    // would allocate and deallocate it itself in every FFT call.
    CudaUniquePtr<Workspace> workspace_{};
};

/**
 * struct MpiFftSplitHandle - Handle for split layouts.
 *
 * Uses VkFFT to perform local FFTs along the split dimensions.
 * Requires a coordinated interleaving of reshapes in between the FFTs.
 */
template <std::floating_point T>
struct MpiFftSplitHandle : public IMpiFftHandle<T> {
public:
    using LocalFFTPolicy = VkFFTPolicy<T, 3>;
    using LocalFFTPlan = typename LocalFFTPolicy::Plan;

    MpiFftSplitHandle(SplitLayout sl,
                      GridSizes<3> original_sizes,
                      GridSizes<3> pencil_sizes,
                      GridSizes<3> slab_sizes,
                      MPI_Comm cart_comm)
        // Wspace size computed in the ctor body
        : sl_{sl} {

        // After a reshape the size of the resulting pencil or slab can
        // (and most likely will) differ from the local grid size
        // of the original grid and we therefore need to initialize
        // the plans with sizes of the reshaped grids.
        fft_plan_1d_ = LocalFFTPolicy::Init(pencil_sizes, sl);
        fft_plan_2d_ = LocalFFTPolicy::Init(slab_sizes, sl);
    }

    void Forward1D(void *dev_buffer) override {
        this->WithFFTTypeAs(
            this->AsWorkspace,
            sl_.forward_type_1d,
            [&](auto workspace_elem_instance) {
                using Workspace = decltype(workspace_elem_instance);

                LocalFFTPolicy::Forward1D(
                    reinterpret_cast<Workspace *>(dev_buffer), fft_plan_1d_);
            });
    }

    void Backward1D(void *dev_buffer) override {
        this->WithFFTTypeAs(
            this->AsWorkspace,
            // We only do C2C and R2R transforms so the forward
            // and backward workspace types are identical.
            sl_.forward_type_1d,
            [&](auto workspace_elem_instance) {
                using Workspace = decltype(workspace_elem_instance);

                LocalFFTPolicy::Backward1D(
                    reinterpret_cast<Workspace *>(dev_buffer), fft_plan_1d_);
            });
    }

    void Forward2D(void *dev_buffer) override {
        this->WithFFTTypeAs(
            this->AsWorkspace,
            sl_.forward_type_2d,
            [&](auto workspace_elem_instance) {
                using Workspace = decltype(workspace_elem_instance);

                LocalFFTPolicy::Forward2D(
                    reinterpret_cast<Workspace *>(dev_buffer), fft_plan_2d_);
            });
    }

    void Backward2D(void *dev_buffer) override {
        this->WithFFTTypeAs(
            this->AsWorkspace,
            // We only do C2C and R2R transforms so the forward
            // and backward workspace types are identical.
            sl_.forward_type_2d,
            [&](auto workspace_elem_instance) {
                using Workspace = decltype(workspace_elem_instance);

                LocalFFTPolicy::Backward2D(
                    reinterpret_cast<Workspace *>(dev_buffer), fft_plan_2d_);
            });
    }

private:
    // Only 1D methods should be called on the 1D plan
    // (same applies for 2D methods on 2D plan) because in the remaining
    // two dimensions (for the 1D plan, it's one for the 2D plan)
    // the grid will not be completely local.
    LocalFFTPlan fft_plan_1d_{};
    LocalFFTPlan fft_plan_2d_{};

    SplitLayout sl_{};
};

template <std::floating_point T>
struct HeFFTePlan {
public:
    std::unique_ptr<IMpiFftHandle<T>> handle{};
};

template <std::floating_point T, std::size_t Dim>
struct MpiFftPolicy {
public:
    using Plan = HeFFTePlan<T>;

    static auto Init(GridSizes<Dim> ns,
                     GridSizes<Dim> pencil_sizes,
                     GridSizes<Dim> slab_sizes,
                     MPI_Comm comm,
                     SplitLayout sl) -> Plan {
        if constexpr (Dim == 3) {
            return GetSplitPlan(sl, ns, pencil_sizes, slab_sizes, comm);
        } else {
            throw std::runtime_error("Getting a split plan when Dim != 3");
        }
    }

    static auto Init(GridSizes<Dim> ns,
                     GridOffsets<Dim> offs,
                     MPI_Comm comm,
                     FullLayout fl) -> Plan {
        return GetFullPlan(fl, ns, offs, comm);
    }

    template <ValidWorkspace<T> Workspace>
    static void Forward(Workspace *dev_buffer, Plan &plan) {
        plan.handle->Forward(dev_buffer);
    }

    template <ValidWorkspace<T> Workspace>
    static void Backward(Workspace *dev_buffer, Plan &plan) {
        plan.handle->Backward(dev_buffer);
    }

    template <ValidWorkspace<T> Workspace>
    static void Forward1D(Workspace *dev_buffer, Plan &plan) {
        plan.handle->Forward1D(dev_buffer);
    }

    template <ValidWorkspace<T> Workspace>
    static void Backward1D(Workspace *dev_buffer, Plan &plan) {
        plan.handle->Backward1D(dev_buffer);
    }

    template <ValidWorkspace<T> Workspace>
    static void Forward2D(Workspace *dev_buffer, Plan &plan) {
        plan.handle->Forward2D(dev_buffer);
    }

    template <ValidWorkspace<T> Workspace>
    static void Backward2D(Workspace *dev_buffer, Plan &plan) {
        plan.handle->Backward2D(dev_buffer);
    }

private:
    static auto Get1DPlan(FullLayout fl,
                          GridSizes<1> ns,
                          GridOffsets<1> offs,
                          MPI_Comm comm) -> Plan {
        return Plan{std::make_unique<MpiFft1DHandle<T>>(fl, ns, offs, comm)};
    }

    static auto GetFullPlan(FullLayout fl,
                            GridSizes<Dim> ns,
                            GridOffsets<Dim> offs,
                            MPI_Comm comm) -> Plan {
        using enum FFTTransformType;

        if constexpr (Dim == 1) {
            return Get1DPlan(fl, ns, offs, comm);
        }

        auto grid_box = GetGridBox(ns, offs);

        auto options = heffte::default_options<heffte::backend::cufft>();
        options.algorithm = heffte::reshape_algorithm::p2p_plined;
        switch (fl.forward_type) {
            case Complex: {
                using TransformTag = heffte::backend::cufft;
                auto plan =
                    heffte::make_fft3d<TransformTag>(grid_box, grid_box, comm, options);
                return Plan{std::make_unique<MpiFftFullHandle<T, TransformTag>>(
                    std::move(plan))};
            }
            case DCT1: {
                using TransformTag = heffte::backend::cufft_cos1;
                auto plan =
                    heffte::make_fft3d<TransformTag>(grid_box, grid_box, comm, options);
                return Plan{std::make_unique<MpiFftFullHandle<T, TransformTag>>(
                    std::move(plan))};
            }
            case DCT2: {
                using TransformTag = heffte::backend::cufft_cos;
                auto plan =
                    heffte::make_fft3d<TransformTag>(grid_box, grid_box, comm, options);
                return Plan{std::make_unique<MpiFftFullHandle<T, TransformTag>>(
                    std::move(plan))};
            }
            case DCT3:
                // There is never a situation where full solver has DCT3 set as
                // a forward FFT type.
                throw std::runtime_error(
                    "Internal error. Full solver shouldn't have DCT3 as "
                    "forward FFT.");
            case DCT4: {
                using TransformTag = heffte::backend::cufft_cos4;
                auto plan =
                    heffte::make_fft3d<TransformTag>(grid_box, grid_box, comm, options);
                return Plan{std::make_unique<MpiFftFullHandle<T, TransformTag>>(
                    std::move(plan))};
            }
            case DST1: {
                using TransformTag = heffte::backend::cufft_sin1;
                auto plan =
                    heffte::make_fft3d<TransformTag>(grid_box, grid_box, comm, options);
                return Plan{std::make_unique<MpiFftFullHandle<T, TransformTag>>(
                    std::move(plan))};
            }
            case DST2: {
                using TransformTag = heffte::backend::cufft_sin;
                auto plan =
                    heffte::make_fft3d<TransformTag>(grid_box, grid_box, comm, options);
                return Plan{std::make_unique<MpiFftFullHandle<T, TransformTag>>(
                    std::move(plan))};
            }
            case DST3:
                // There is never a situation where full solver has DST3 set as
                // a forward FFT type.
                throw std::runtime_error(
                    "Internal error. Full solver shouldn't have DST3 as "
                    "forward FFT.");
            case DST4: {
                using TransformTag = heffte::backend::cufft_sin4;
                auto plan =
                    heffte::make_fft3d<TransformTag>(grid_box, grid_box, comm, options);
                return Plan{std::make_unique<MpiFftFullHandle<T, TransformTag>>(
                    std::move(plan))};
            }
            default:
                throw std::runtime_error("Invalid FFT forward type.");
        }
    }

    static auto GetSplitPlan(SplitLayout sl,
                             GridSizes<3> ns,
                             GridSizes<3> pencil_sizes,
                             GridSizes<3> slab_sizes,
                             MPI_Comm comm) -> Plan {
        return Plan{std::make_unique<MpiFftSplitHandle<T>>(
            sl, ns, pencil_sizes, slab_sizes, comm)};
    }
};

template <std::floating_point T, std::size_t Dim>
using FFTPolicy = MpiFftPolicy<T, Dim>;

#else

template <std::floating_point T, std::size_t Dim>
using FFTPolicy = VkFFTPolicy<T, Dim>;

#endif  // POISSON_SOLVER_USE_MPI
}  // namespace poisson_solver::detail

#endif  // FFT_POLICIES_HPP
