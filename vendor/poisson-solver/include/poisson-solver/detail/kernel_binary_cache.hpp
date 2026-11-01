#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace poisson_solver::detail::kernel_cache {
using Digest = std::array<unsigned char, 32>;
constexpr std::size_t max_binary_bytes = 16U << 20;
constexpr std::size_t max_disk_bytes = 64U << 20;

Digest hash(std::span<const unsigned char> bytes);
Digest hash_parts(const std::vector<std::string> &parts);
Digest hash_file(const std::string &filename);
std::string hex(const Digest &value);
bool valid_cubin(std::span<const unsigned char> bytes) noexcept;

// An explicit, caller-selected private directory. Missing/unsafe directories,
// corrupt artifacts and I/O failures are misses. No code is loaded by this
// class.
class BinaryCache {
public:
  explicit BinaryCache(const std::string &directory) noexcept;
  ~BinaryCache();
  BinaryCache(const BinaryCache &) = delete;
  BinaryCache &operator=(const BinaryCache &) = delete;
  bool enabled() const noexcept { return directory_ >= 0; }
  std::vector<unsigned char> load(const Digest &key) const noexcept;
  void store(const Digest &key,
             std::span<const unsigned char> bytes) const noexcept;

private:
  int directory_ = -1;
};
} // namespace poisson_solver::detail::kernel_cache
