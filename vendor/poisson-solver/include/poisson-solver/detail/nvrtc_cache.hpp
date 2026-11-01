#pragma once
#include <cstddef>
#include <nvrtc.h>

// These wrappers are used only at the vendored provider's VkFFT compile
// boundary. They do not interpose public NVRTC symbols or other libraries.
namespace poisson_solver::detail::nvrtc_cache {
class CreationScope {
public:
  CreationScope() noexcept;
  ~CreationScope();
  CreationScope(const CreationScope &) = delete;
  CreationScope &operator=(const CreationScope &) = delete;

private:
  CreationScope *previous_;
};
nvrtcResult create(nvrtcProgram *, const char *, const char *, int,
                   const char *const *, const char *const *) noexcept;
nvrtcResult compile(nvrtcProgram, int, const char *const *) noexcept;
nvrtcResult cubin_size(nvrtcProgram, std::size_t *) noexcept;
nvrtcResult cubin(nvrtcProgram, char *) noexcept;
nvrtcResult destroy(nvrtcProgram *) noexcept;
} // namespace poisson_solver::detail::nvrtc_cache
