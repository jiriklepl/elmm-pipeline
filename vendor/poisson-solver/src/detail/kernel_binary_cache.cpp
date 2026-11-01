#include "poisson-solver/detail/kernel_binary_cache.hpp"
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <fstream>
#include <memory>
#include <openssl/evp.h>
#include <stdexcept>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

namespace poisson_solver::detail::kernel_cache {
namespace {
constexpr std::array<unsigned char, 16> magic{'P', 'O', 'I', 'S', 'S', 'O',
                                              'N', '_', 'C', 'U', 'B', 'I',
                                              'N', '_', '1', 0};
constexpr std::size_t header_bytes = 16 + 32 + 8 + 32;
struct File {
  int fd;
  explicit File(int value = -1) : fd(value) {}
  ~File() {
    if (fd >= 0)
      close(fd);
  }
  File(const File &) = delete;
  File &operator=(const File &) = delete;
};
struct DirectoryDeleter {
  void operator()(DIR *value) const {
    if (value)
      closedir(value);
  }
};
bool private_file(int fd, struct stat &s) {
  return fd >= 0 && fstat(fd, &s) == 0 && S_ISREG(s.st_mode) &&
         s.st_uid == geteuid() && !(s.st_mode & 0077) && s.st_nlink == 1;
}
bool transfer(int fd, unsigned char *data, std::size_t bytes, bool writing) {
  while (bytes) {
    const ssize_t count =
        writing ? write(fd, data, bytes) : read(fd, data, bytes);
    if (count < 0 && errno == EINTR)
      continue;
    if (count <= 0)
      return false;
    data += count;
    bytes -= std::size_t(count);
  }
  return true;
}
std::uint64_t u64(std::span<const unsigned char> bytes, std::size_t offset) {
  std::uint64_t value = 0;
  for (unsigned i = 0; i < 8; ++i)
    value |= std::uint64_t(bytes[offset + i]) << (8 * i);
  return value;
}
std::uint32_t u32(std::span<const unsigned char> bytes, std::size_t offset) {
  return std::uint32_t(bytes[offset]) | std::uint32_t(bytes[offset + 1]) << 8 |
         std::uint32_t(bytes[offset + 2]) << 16 |
         std::uint32_t(bytes[offset + 3]) << 24;
}
std::uint16_t u16(std::span<const unsigned char> bytes, std::size_t offset) {
  return std::uint16_t(bytes[offset]) | std::uint16_t(bytes[offset + 1]) << 8;
}
bool extent(std::size_t total, std::uint64_t offset, std::uint64_t bytes) {
  return offset <= total && bytes <= total - offset;
}
bool managed_name(const std::string &name) {
  return name.size() == 70 && name.ends_with(".cubin") &&
         std::all_of(name.begin(), name.begin() + 64, [](char c) {
           return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
         });
}
} // namespace

Digest hash(std::span<const unsigned char> bytes) {
  Digest result{};
  unsigned length = 0;
  if (EVP_Digest(bytes.data(), bytes.size(), result.data(), &length,
                 EVP_sha256(), nullptr) != 1 ||
      length != result.size())
    throw std::runtime_error("kernel cache digest failed");
  return result;
}

Digest hash_parts(const std::vector<std::string> &parts) {
  // Length framing distinguishes ["ab","c"] from ["a","bc"], including NULs.
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(
      EVP_MD_CTX_new(), EVP_MD_CTX_free);
  if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1)
    throw std::runtime_error("kernel cache digest initialization failed");
  for (const auto &part : parts) {
    std::array<unsigned char, 8> size{};
    for (unsigned i = 0; i < 8; ++i)
      size[i] = (std::uint64_t(part.size()) >> (8 * i)) & 255;
    if (EVP_DigestUpdate(context.get(), size.data(), size.size()) != 1 ||
        EVP_DigestUpdate(context.get(), part.data(), part.size()) != 1)
      throw std::runtime_error("kernel cache digest update failed");
  }
  Digest result{};
  unsigned length = 0;
  if (EVP_DigestFinal_ex(context.get(), result.data(), &length) != 1 ||
      length != result.size())
    throw std::runtime_error("kernel cache digest finalization failed");
  return result;
}

Digest hash_file(const std::string &filename) {
  std::ifstream input(filename, std::ios::binary);
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(
      EVP_MD_CTX_new(), EVP_MD_CTX_free);
  if (!input || !context ||
      EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1)
    throw std::runtime_error("kernel cache library digest unavailable");
  std::array<char, 65536> bytes{};
  std::size_t total = 0;
  while (input) {
    input.read(bytes.data(), bytes.size());
    const auto count = input.gcount();
    total += std::size_t(count);
    if (total > 256U * 1024 * 1024 ||
        EVP_DigestUpdate(context.get(), bytes.data(), std::size_t(count)) != 1)
      throw std::runtime_error("kernel cache library digest failed");
  }
  if (!input.eof() || !total)
    throw std::runtime_error("kernel cache library read failed");
  Digest result{};
  unsigned length = 0;
  if (EVP_DigestFinal_ex(context.get(), result.data(), &length) != 1 ||
      length != result.size())
    throw std::runtime_error("kernel cache library digest finalization failed");
  return result;
}

std::string hex(const Digest &value) {
  static constexpr char alphabet[] = "0123456789abcdef";
  std::string result;
  result.reserve(64);
  for (auto byte : value) {
    result.push_back(alphabet[byte >> 4]);
    result.push_back(alphabet[byte & 15]);
  }
  return result;
}

bool valid_cubin(std::span<const unsigned char> bytes) noexcept {
  // Accept the bounded ELF64 little-endian CUDA container emitted by the
  // pinned modern NVRTC. Unsupported formats simply do not enter the cache.
  if (bytes.size() < 64 || bytes.size() > max_binary_bytes ||
      std::memcmp(bytes.data(), "\177ELF", 4) || bytes[4] != 2 ||
      bytes[5] != 1 || bytes[6] != 1 || u16(bytes, 18) != 190 ||
      u32(bytes, 20) != 1 || u16(bytes, 52) != 64 ||
      (u16(bytes, 16) != 1 && u16(bytes, 16) != 2))
    return false;
  const auto phoff = u64(bytes, 32), shoff = u64(bytes, 40);
  const auto phnum = u16(bytes, 56), shnum = u16(bytes, 60);
  if (!shnum || phnum == 65535 || u16(bytes, 62) >= shnum ||
      u16(bytes, 58) != 64 ||
      !extent(bytes.size(), shoff, std::uint64_t(shnum) * 64) ||
      (phnum && (u16(bytes, 54) != 56 ||
                 !extent(bytes.size(), phoff, std::uint64_t(phnum) * 56))))
    return false;
  for (std::uint64_t i = 0; i < shnum; ++i) {
    const auto p = std::size_t(shoff + i * 64);
    if ((u32(bytes, p + 4) != 8 &&
         !extent(bytes.size(), u64(bytes, p + 24), u64(bytes, p + 32))) ||
        u32(bytes, p + 40) >= shnum)
      return false;
  }
  for (std::uint64_t i = 0; i < phnum; ++i) {
    const auto p = std::size_t(phoff + i * 56);
    if (!extent(bytes.size(), u64(bytes, p + 8), u64(bytes, p + 32)))
      return false;
  }
  return true;
}

BinaryCache::BinaryCache(const std::string &path) noexcept {
  if (path.empty() || path.front() != '/' ||
      path.find('\0') != std::string::npos)
    return;
  // Parent directory creation belongs to the explicit caller/pipeline.
  directory_ =
      open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
  struct stat s{};
  if (directory_ >= 0 &&
      (fstat(directory_, &s) || s.st_uid != geteuid() || (s.st_mode & 0077))) {
    close(directory_);
    directory_ = -1;
  }
}
BinaryCache::~BinaryCache() {
  if (directory_ >= 0)
    close(directory_);
}

std::vector<unsigned char> BinaryCache::load(const Digest &key) const noexcept {
  try {
    if (!enabled())
      return {};
    const auto name = hex(key) + ".cubin";
    File file(openat(directory_, name.c_str(),
                     O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK));
    struct stat s{};
    if (!private_file(file.fd, s) || s.st_size < std::int64_t(header_bytes) ||
        s.st_size > std::int64_t(header_bytes + max_binary_bytes))
      return {};
    std::array<unsigned char, header_bytes> header{};
    if (!transfer(file.fd, header.data(), header.size(), false) ||
        !std::equal(magic.begin(), magic.end(), header.begin()) ||
        !std::equal(key.begin(), key.end(), header.begin() + 16))
      return {};
    const auto size = u64(header, 48);
    if (size != std::uint64_t(s.st_size - header_bytes))
      return {};
    std::vector<unsigned char> bytes(std::size_t(size), 0);
    if (!transfer(file.fd, bytes.data(), bytes.size(), false))
      return {};
    unsigned char extra = 0;
    if (read(file.fd, &extra, 1) != 0 || !valid_cubin(bytes))
      return {};
    const auto expected = hash(bytes);
    if (!std::equal(expected.begin(), expected.end(), header.begin() + 56))
      return {};
    return bytes;
  } catch (...) {
    return {};
  }
}

void BinaryCache::store(const Digest &key,
                        std::span<const unsigned char> bytes) const noexcept {
  std::string temporary;
  try {
    if (!enabled() || !valid_cubin(bytes))
      return;
    File lock(openat(directory_, ".lock",
                     O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK,
                     0600));
    struct stat lock_stat{};
    if (!private_file(lock.fd, lock_stat) || flock(lock.fd, LOCK_EX | LOCK_NB))
      return;
    const auto name = hex(key) + ".cubin";
    // A parallel miss may already have populated this key. Replacing it
    // atomically is safe and avoids a second full-image read/allocation.
    struct Row {
      std::string name;
      std::uint64_t bytes;
      timespec age;
      bool removable;
    };
    std::vector<Row> rows;
    std::uint64_t used = 0;
    // dup() would share and advance the retained directory offset across
    // stores. A fresh open file description starts every bounded scan.
    const int scan_fd = openat(directory_, ".",
                               O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    std::unique_ptr<DIR, DirectoryDeleter> entries(
        scan_fd < 0 ? nullptr : fdopendir(scan_fd));
    if (!entries && scan_fd >= 0)
      close(scan_fd);
    if (!entries)
      return;
    while (auto *entry = readdir(entries.get())) {
      const std::string filename = entry->d_name;
      if (filename == "." || filename == ".." || filename == ".lock")
        continue;
      struct stat s{};
      if (fstatat(directory_, filename.c_str(), &s, AT_SYMLINK_NOFOLLOW) ||
          s.st_size < 0)
        return;
      if (!S_ISREG(s.st_mode))
        return; // Foreign objects are never removed.
      if (std::uint64_t(s.st_size) > max_disk_bytes ||
          used > max_disk_bytes - std::uint64_t(s.st_size))
        return;
      used += std::uint64_t(s.st_size);
      rows.push_back(
          {filename, std::uint64_t(s.st_size), s.st_mtim,
           s.st_uid == geteuid() && !(s.st_mode & 0077) && s.st_nlink == 1 &&
               (managed_name(filename) || filename.starts_with(".tmp-"))});
      if (rows.size() > 4096)
        return;
    }
    const std::uint64_t required = header_bytes + bytes.size();
    std::sort(rows.begin(), rows.end(), [](const Row &a, const Row &b) {
      return a.age.tv_sec != b.age.tv_sec     ? a.age.tv_sec < b.age.tv_sec
             : a.age.tv_nsec != b.age.tv_nsec ? a.age.tv_nsec < b.age.tv_nsec
                                              : a.name < b.name;
    });
    // Remove stale same-key data and then oldest clean cache files. Loaded
    // images own their host bytes independently of any directory entry.
    for (const auto &row : rows) {
      if (row.removable && row.name == name) {
        if (unlinkat(directory_, row.name.c_str(), 0))
          return;
        used -= row.bytes;
      }
    }
    for (const auto &row : rows) {
      if (used <= max_disk_bytes - required)
        break;
      if (!row.removable || row.name == name)
        continue;
      if (unlinkat(directory_, row.name.c_str(), 0))
        return;
      used -= row.bytes;
    }
    if (used > max_disk_bytes - required)
      return;
    static std::atomic<unsigned long> serial{0};
    temporary = ".tmp-" + std::to_string(getpid()) + "-" +
                std::to_string(serial.fetch_add(1));
    File file(openat(directory_, temporary.c_str(),
                     O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
                     0600));
    if (file.fd < 0) {
      temporary.clear();
      return;
    }
    std::array<unsigned char, header_bytes> header{};
    std::copy(magic.begin(), magic.end(), header.begin());
    std::copy(key.begin(), key.end(), header.begin() + 16);
    for (unsigned i = 0; i < 8; ++i)
      header[48 + i] = (std::uint64_t(bytes.size()) >> (8 * i)) & 255;
    const auto checksum = hash(bytes);
    std::copy(checksum.begin(), checksum.end(), header.begin() + 56);
    if (transfer(file.fd, header.data(), header.size(), true) &&
        transfer(file.fd, const_cast<unsigned char *>(bytes.data()),
                 bytes.size(), true) &&
        fsync(file.fd) == 0 &&
        renameat(directory_, temporary.c_str(), directory_, name.c_str()) == 0)
      temporary.clear();
  } catch (...) { /* A cache cannot turn a valid normal compile into failure. */
  }
  if (!temporary.empty())
    unlinkat(directory_, temporary.c_str(), 0);
}
} // namespace poisson_solver::detail::kernel_cache
