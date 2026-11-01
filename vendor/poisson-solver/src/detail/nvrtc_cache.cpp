#include "poisson-solver/detail/nvrtc_cache.hpp"
#include "kernel_cache_build_identity.hpp"
#include "poisson-solver/detail/kernel_binary_cache.hpp"
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <fstream>
#include <link.h>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace poisson_solver::detail::nvrtc_cache {
namespace {
using kernel_cache::Digest;
constexpr std::size_t max_live_programs = 64;
constexpr std::size_t max_live_image_bytes = 64U << 20;
constexpr std::size_t max_source_bytes = 8U << 20;
constexpr std::size_t max_header_bytes = 32U << 20;
struct State {
  explicit State(std::string directory, Digest source, CreationScope *owner)
      : directory(std::move(directory)), source(source), owner(owner) {}
  std::mutex mutex;
  std::string directory;
  Digest source{}, key{};
  CreationScope *owner;
  std::vector<unsigned char> image;
  bool keyed = false, compiled = false, reserved = false;
  std::size_t binary_bytes = 0;
};
struct Registry {
  std::mutex mutex;
  std::map<nvrtcProgram, std::shared_ptr<State>> programs;
  std::size_t image_reservations = 0;
};
Registry &registry() {
  static Registry value;
  return value;
}
thread_local CreationScope *active_scope = nullptr;
std::shared_ptr<State> lookup(nvrtcProgram p) {
  auto &r = registry();
  std::lock_guard lock(r.mutex);
  auto i = r.programs.find(p);
  return i == r.programs.end() ? nullptr : i->second;
}
void release_image(State &state) {
  std::vector<unsigned char>().swap(state.image);
  if (state.reserved) {
    auto &r = registry();
    std::lock_guard lock(r.mutex);
    r.image_reservations -= kernel_cache::max_binary_bytes;
    state.reserved = false;
  }
}
std::string bounded(const char *text, std::size_t limit) {
  if (!text)
    throw std::invalid_argument("absent compiler text");
  auto size = strnlen(text, limit + 1);
  if (size > limit)
    throw std::invalid_argument("compiler text exceeds cache bound");
  return std::string(text, size);
}
bool self_contained(const std::string &source) {
  // The pinned generator uses no filesystem includes. Conservatively decline
  // even comments/identifiers containing "include", rather than pretending
  // that a partial preprocessor resolves arbitrary external dependencies.
  std::string joined;
  joined.reserve(source.size());
  for (std::size_t i = 0; i < source.size(); ++i) {
    if (source[i] == '\\' && i + 1 < source.size() && source[i + 1] == '\n') {
      ++i;
      continue;
    }
    if (source[i] == '\\' && i + 2 < source.size() && source[i + 1] == '\r' &&
        source[i + 2] == '\n') {
      i += 2;
      continue;
    }
    joined.push_back(source[i]);
  }
  return joined.find("include") == std::string::npos &&
         joined.find("embed") == std::string::npos &&
         joined.find("??") == std::string::npos &&
         joined.find("__TIME__") == std::string::npos &&
         joined.find("__DATE__") == std::string::npos &&
         joined.find("__TIMESTAMP__") == std::string::npos;
}
bool pinned_option(const std::string &option) {
  const std::string prefix = "--gpu-architecture=sm_";
  if (option == "-fmad=false")
    return true;
  if (!option.starts_with(prefix))
    return false;
  const auto suffix = option.substr(prefix.size());
  return !suffix.empty() && suffix.size() <= 4 &&
         suffix.find_first_not_of("0123456789af") == std::string::npos;
}
std::string library_digest(const void *function) {
  Dl_info info{};
  if (!dladdr(function, &info) || !info.dli_fname)
    throw std::runtime_error("compiler library unavailable");
  return kernel_cache::hex(kernel_cache::hash_file(info.dli_fname));
}
struct BuiltinsLibrary {
  void *handle = nullptr;
  ~BuiltinsLibrary() {
    if (handle)
      dlclose(handle);
  }
  std::string identity(int major, int minor) {
    // NVRTC's supported Linux builtins dependency is versioned by major and
    // minor. Resolve the same SONAME before compilation and retain the
    // object, so a loader search override is represented by its actual
    // bytes rather than only by the configure-time toolkit path.
    const std::string soname = "libnvrtc-builtins.so." + std::to_string(major) +
                               "." + std::to_string(minor);
    handle = dlopen(soname.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle)
      handle =
          dlopen(POISSON_SOLVER_CACHE_NVRTC_BUILTINS, RTLD_NOW | RTLD_LOCAL);
    // A configured absolute-path fallback must be the dependency SONAME
    // requested by this runtime NVRTC. An unknown/minor-mismatched compiler
    // must miss, not key its kernels using another compiler's builtins.
    void *selected =
        dlopen(soname.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
    const bool same_object = selected && selected == handle;
    if (selected)
      dlclose(selected);
    if (!same_object)
      throw std::runtime_error("NVRTC builtins SONAME is not bound");
    link_map *object = nullptr;
    if (!handle || dlinfo(handle, RTLD_DI_LINKMAP, &object) || !object ||
        !object->l_name)
      throw std::runtime_error("NVRTC builtins identity unavailable");
    return kernel_cache::hex(kernel_cache::hash_file(object->l_name));
  }
};
const std::optional<std::string> &library_identity() {
  static BuiltinsLibrary builtins;
  static const auto value = []() -> std::optional<std::string> {
    try {
      int major = 0, minor = 0;
      const std::string build = POISSON_SOLVER_CACHE_BUILD_KEY;
      if (build.size() != 64 ||
          build.find_first_not_of("0123456789abcdef") != std::string::npos)
        return {};
      if (nvrtcVersion(&major, &minor) != NVRTC_SUCCESS)
        return {};
      return build + ";nvrtc=" + std::to_string(major) + "." +
             std::to_string(minor) + ";nvrtc-image=" +
             library_digest(reinterpret_cast<const void *>(&nvrtcVersion)) +
             ";nvrtc-builtins=" + builtins.identity(major, minor) +
             ";driver-image=" +
             library_digest(reinterpret_cast<const void *>(&cuDeviceGet));
    } catch (...) {
      return {};
    }
  }();
  return value;
}
std::string device_identity() {
  CUdevice device;
  CUuuid uuid{};
  int driver = 0, runtime = 0;
  if (cuCtxGetDevice(&device) != CUDA_SUCCESS ||
      cuDeviceGetUuid_v2(&uuid, device) != CUDA_SUCCESS ||
      cuDriverGetVersion(&driver) != CUDA_SUCCESS ||
      cudaRuntimeGetVersion(&runtime) != cudaSuccess)
    throw std::runtime_error("kernel cache device identity unavailable");
  static constexpr CUdevice_attribute attributes[] = {
      CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
      CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
      CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
      CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
      CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
      CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,
      CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
      CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
      CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
      CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
      CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
      CU_DEVICE_ATTRIBUTE_WARP_SIZE,
      CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO};
  std::string result(uuid.bytes, sizeof uuid.bytes);
  result += ";driver=" + std::to_string(driver) +
            ";runtime=" + std::to_string(runtime);
  for (auto attribute : attributes) {
    int value = 0;
    if (cuDeviceGetAttribute(&value, attribute, device) != CUDA_SUCCESS)
      throw std::runtime_error("kernel cache attribute unavailable");
    result +=
        ';' + std::to_string(int(attribute)) + '=' + std::to_string(value);
  }
  return result;
}
} // namespace

CreationScope::CreationScope() noexcept : previous_(active_scope) {
  active_scope = this;
}
CreationScope::~CreationScope() {
  active_scope = previous_;
  // VkFFT error exits can omit nvrtcDestroyProgram. Retire cache-owned
  // compilation records at the enclosing plan-creation boundary regardless
  // of the exit path. No source kernels have been submitted at this point.
  auto &r = registry();
  for (;;) {
    nvrtcProgram program = nullptr;
    std::shared_ptr<State> state;
    {
      std::lock_guard lock(r.mutex);
      for (auto i = r.programs.begin(); i != r.programs.end(); ++i) {
        if (i->second->owner == this) {
          program = i->first;
          state = i->second;
          r.programs.erase(i);
          break;
        }
      }
    }
    if (!state)
      break;
    std::lock_guard lock(state->mutex);
    release_image(*state);
    nvrtcDestroyProgram(&program);
  }
}

nvrtcResult create(nvrtcProgram *out, const char *source, const char *name,
                   int headers, const char *const *contents,
                   const char *const *names) noexcept {
  const auto result =
      nvrtcCreateProgram(out, source, name, headers, contents, names);
  if (result != NVRTC_SUCCESS)
    return result;
  try {
    const char *directory = getenv("POISSON_SOLVER_KERNEL_CACHE_DIR");
    if (!active_scope || !directory || !*directory || !source || headers < 0 ||
        headers > 64)
      return result;
    kernel_cache::BinaryCache cache(bounded(directory, 4096));
    if (!cache.enabled())
      return result;
    std::vector<std::string> parts{
        "poisson-nvrtc-cubin-v1", bounded(source, max_source_bytes),
        name ? "name-present" : "name-null", name ? bounded(name, 4096) : ""};
    if (!self_contained(parts[1]))
      return result;
    std::size_t total = 0;
    for (int i = 0; i < headers; ++i) {
      if (!contents || !names)
        return result;
      auto text = bounded(contents[i], max_source_bytes);
      total += text.size();
      if (total > max_header_bytes)
        return result;
      if (!self_contained(text))
        return result;
      parts.push_back(bounded(names[i], 4096));
      parts.push_back(std::move(text));
    }
    auto state = std::make_shared<State>(
        directory, kernel_cache::hash_parts(parts), active_scope);
    auto &r = registry();
    std::lock_guard lock(r.mutex);
    if (r.programs.size() < max_live_programs)
      r.programs.emplace(*out, std::move(state));
  } catch (...) { /* Optional cache metadata failure preserves native NVRTC
                     semantics. */
  }
  return result;
}

nvrtcResult compile(nvrtcProgram program, int count,
                    const char *const *options) noexcept {
  std::shared_ptr<State> state;
  try {
    state = lookup(program);
  } catch (...) {
    return nvrtcCompileProgram(program, count, options);
  }
  if (!state)
    return nvrtcCompileProgram(program, count, options);
  std::lock_guard lock(state->mutex);
  release_image(*state);
  state->keyed = false;
  state->compiled = false;
  state->binary_bytes = 0;
  try {
    const auto &library = library_identity();
    if (!library || count < 0 || count > 64)
      throw std::runtime_error("uncacheable compiler recipe");
    std::vector<std::string> parts{*library, kernel_cache::hex(state->source),
                                   device_identity()};
    for (int i = 0; i < count; ++i) {
      if (!options)
        throw std::invalid_argument("absent options");
      auto option = bounded(options[i], 16384);
      if (!pinned_option(option))
        throw std::invalid_argument("compiler option outside pinned recipe");
      parts.push_back(std::move(option));
    }
    state->key = kernel_cache::hash_parts(parts);
    state->keyed = true;
    auto &r = registry();
    {
      std::lock_guard guard(r.mutex);
      if (r.image_reservations <=
          max_live_image_bytes - kernel_cache::max_binary_bytes) {
        r.image_reservations += kernel_cache::max_binary_bytes;
        state->reserved = true;
      }
    }
    if (state->reserved) {
      state->image =
          kernel_cache::BinaryCache(state->directory).load(state->key);
      if (!state->image.empty()) {
        state->compiled = true;
        return NVRTC_SUCCESS;
      }
      release_image(*state);
    }
  } catch (...) {
    release_image(*state);
    state->keyed = false;
  }
  const auto result = nvrtcCompileProgram(program, count, options);
  state->compiled = result == NVRTC_SUCCESS;
  return result;
}

nvrtcResult cubin_size(nvrtcProgram program, std::size_t *out) noexcept {
  try {
    auto state = lookup(program);
    if (!state)
      return nvrtcGetCUBINSize(program, out);
    std::lock_guard lock(state->mutex);
    if (!state->image.empty()) {
      if (!out)
        return NVRTC_ERROR_INVALID_INPUT;
      *out = state->image.size();
      return NVRTC_SUCCESS;
    }
    const auto result = nvrtcGetCUBINSize(program, out);
    if (result == NVRTC_SUCCESS && out)
      state->binary_bytes = *out;
    return result;
  } catch (...) {
    return NVRTC_ERROR_INTERNAL_ERROR;
  }
}

nvrtcResult cubin(nvrtcProgram program, char *out) noexcept {
  try {
    auto state = lookup(program);
    if (!state)
      return nvrtcGetCUBIN(program, out);
    std::lock_guard lock(state->mutex);
    if (!state->image.empty()) {
      if (!out)
        return NVRTC_ERROR_INVALID_INPUT;
      std::memcpy(out, state->image.data(), state->image.size());
      return NVRTC_SUCCESS;
    }
    const auto result = nvrtcGetCUBIN(program, out);
    if (result == NVRTC_SUCCESS && state->compiled && state->keyed && out &&
        state->binary_bytes &&
        state->binary_bytes <= kernel_cache::max_binary_bytes)
      kernel_cache::BinaryCache(state->directory)
          .store(state->key, {reinterpret_cast<const unsigned char *>(out),
                              state->binary_bytes});
    return result;
  } catch (...) {
    return NVRTC_ERROR_INTERNAL_ERROR;
  }
}

nvrtcResult destroy(nvrtcProgram *program) noexcept {
  if (!program)
    return nvrtcDestroyProgram(program);
  try {
    const auto original = *program;
    auto state = lookup(original);
    std::unique_lock<std::mutex> lock;
    if (state)
      lock = std::unique_lock<std::mutex>(state->mutex);
    const auto result = nvrtcDestroyProgram(program);
    if (result == NVRTC_SUCCESS && state) {
      release_image(*state);
      auto &r = registry();
      std::lock_guard guard(r.mutex);
      r.programs.erase(original);
    }
    return result;
  } catch (...) {
    return NVRTC_ERROR_INTERNAL_ERROR;
  }
}
} // namespace poisson_solver::detail::nvrtc_cache
