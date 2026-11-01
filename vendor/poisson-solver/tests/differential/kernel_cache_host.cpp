#include "poisson-solver/detail/kernel_binary_cache.hpp"
#include "poisson-solver/detail/nvrtc_cache.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
namespace cache = poisson_solver::detail::kernel_cache;
namespace wrap = poisson_solver::detail::nvrtc_cache;
namespace fs = std::filesystem;

void set16(std::vector<unsigned char> &v, size_t p, unsigned x) {
  v[p] = x & 255;
  v[p + 1] = (x >> 8) & 255;
}
void set64(std::vector<unsigned char> &v, size_t p, uint64_t x) {
  for (unsigned i = 0; i < 8; ++i)
    v[p + i] = (x >> (8 * i)) & 255;
}
std::vector<unsigned char> binary(size_t n = 128) {
  std::vector<unsigned char> v(n, 0);
  std::memcpy(v.data(), "\177ELF", 4);
  v[4] = 2;
  v[5] = 1;
  v[6] = 1;
  set16(v, 16, 1);
  set16(v, 18, 190);
  v[20] = 1;
  set64(v, 40, 64);
  set16(v, 52, 64);
  set16(v, 58, 64);
  set16(v, 60, 1);
  return v;
}
struct _nvrtcProgram {
  std::string source;
  bool compiled = false;
};
std::atomic<unsigned> creates{0}, compiles{0}, destroys{0}, live{0};
extern "C" nvrtcResult nvrtcCreateProgram(nvrtcProgram *out, const char *source,
                                          const char *, int,
                                          const char *const *,
                                          const char *const *) {
  if (!out || !source)
    return NVRTC_ERROR_INVALID_INPUT;
  *out = new _nvrtcProgram{source};
  ++creates;
  ++live;
  return NVRTC_SUCCESS;
}
extern "C" nvrtcResult nvrtcCompileProgram(nvrtcProgram p, int,
                                           const char *const *) {
  if (!p)
    return NVRTC_ERROR_INVALID_INPUT;
  ++compiles;
  if (p->source == "FAIL")
    return NVRTC_ERROR_COMPILATION;
  p->compiled = true;
  return NVRTC_SUCCESS;
}
extern "C" nvrtcResult nvrtcGetCUBINSize(nvrtcProgram p, size_t *out) {
  if (!p || !p->compiled || !out)
    return NVRTC_ERROR_INVALID_INPUT;
  *out = 128;
  return NVRTC_SUCCESS;
}
extern "C" nvrtcResult nvrtcGetCUBIN(nvrtcProgram p, char *out) {
  if (!p || !p->compiled || !out)
    return NVRTC_ERROR_INVALID_INPUT;
  auto v = binary();
  std::memcpy(out, v.data(), v.size());
  return NVRTC_SUCCESS;
}
extern "C" nvrtcResult nvrtcDestroyProgram(nvrtcProgram *p) {
  if (!p || !*p)
    return NVRTC_ERROR_INVALID_INPUT;
  delete *p;
  *p = nullptr;
  ++destroys;
  --live;
  return NVRTC_SUCCESS;
}
int runtime_major = 13;
extern "C" nvrtcResult nvrtcVersion(int *a, int *b) {
  *a = runtime_major;
  *b = 4;
  return NVRTC_SUCCESS;
}
extern "C" CUresult CUDAAPI cuCtxGetDevice(CUdevice *d) {
  *d = 0;
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuDeviceGet(CUdevice *d, int) {
  *d = 0;
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuDeviceGetUuid_v2(CUuuid *u, CUdevice) {
  std::memset(u, 7, sizeof *u);
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuDriverGetVersion(int *v) {
  *v = 13050;
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuDeviceGetAttribute(int *v, CUdevice_attribute a,
                                                 CUdevice) {
  *v = int(a);
  return CUDA_SUCCESS;
}
extern "C" cudaError_t CUDARTAPI cudaRuntimeGetVersion(int *v) {
  *v = 13040;
  return cudaSuccess;
}

struct Temp {
  fs::path root;
  Temp() {
    char p[] = "/tmp/poisson-cache-test-XXXXXX";
    auto result = mkdtemp(p);
    assert(result);
    root = result;
  }
  ~Temp() { fs::remove_all(root); }
};
void output(const fs::path &p, std::span<const unsigned char> b) {
  std::ofstream f(p, std::ios::binary);
  f.write((const char *)b.data(), b.size());
  assert(f.good());
}
std::vector<unsigned char> input(const fs::path &p) {
  std::ifstream f(p, std::ios::binary);
  return {std::istreambuf_iterator<char>(f), {}};
}
void one(const std::string &source,
         const char *option = "--gpu-architecture=sm_86",
         const char *header = "header", bool abandon = false) {
  wrap::CreationScope scope;
  nvrtcProgram p = nullptr;
  const char *contents[] = {header};
  const char *names[] = {"header.h"};
  assert(wrap::create(&p, source.c_str(), "VkFFT.cu", 1, contents, names) ==
         NVRTC_SUCCESS);
  const char *options[] = {option};
  auto code = wrap::compile(p, 1, options);
  if (source == "FAIL") {
    assert(code == NVRTC_ERROR_COMPILATION);
    return;
  }
  assert(code == NVRTC_SUCCESS);
  size_t n = 0;
  assert(wrap::cubin_size(p, &n) == NVRTC_SUCCESS && n == 128);
  std::vector<unsigned char> v(n);
  assert(wrap::cubin(p, (char *)v.data()) == NVRTC_SUCCESS && v == binary());
  if (!abandon)
    assert(wrap::destroy(&p) == NVRTC_SUCCESS && p == nullptr);
}
size_t files(const fs::path &p) {
  size_t result = 0;
  for (auto &e : fs::directory_iterator(p))
    if (e.path().extension() == ".cubin")
      ++result;
  return result;
}

int main(int argc, char **argv) {
  if (argc == 2 && std::string(argv[1]) == "--unknown-runtime") {
    runtime_major = 999;
    Temp t;
    setenv("POISSON_SOLVER_KERNEL_CACHE_DIR", t.root.c_str(), 1);
    one("unknown runtime");
    one("unknown runtime");
    assert(compiles == 2 && files(t.root) == 0 && live == 0);
    assert(creates == destroys);
    std::cout << "PASS unknown runtime dependency bypass\n";
    return 0;
  }
  assert(argc == 1);
  unsigned cases = 0;
  {
    Temp t;
    setenv("POISSON_SOLVER_KERNEL_CACHE_DIR", t.root.c_str(), 1);
    one("held");
    const unsigned initial = compiles;
    wrap::CreationScope scope;
    std::vector<nvrtcProgram> held(5, nullptr);
    const char *contents[] = {"header"};
    const char *names[] = {"header.h"};
    const char *options[] = {"--gpu-architecture=sm_86"};
    for (auto &program : held) {
      assert(wrap::create(&program, "held", "VkFFT.cu", 1, contents, names) ==
             NVRTC_SUCCESS);
      assert(wrap::compile(program, 1, options) == NVRTC_SUCCESS);
      size_t size = 0;
      assert(wrap::cubin_size(program, &size) == NVRTC_SUCCESS && size == 128);
    }
    // Four loaded images reserve the entire budget; the fifth compiles.
    assert(compiles == initial + 1);
    assert(wrap::destroy(&held[0]) == NVRTC_SUCCESS);
    one("held");
    assert(compiles == initial + 1);
    for (auto &program : held)
      if (program)
        assert(wrap::destroy(&program) == NVRTC_SUCCESS);
    assert(live == 0);
    ++cases;
  }
  {
    auto good = binary();
    assert(cache::valid_cubin(good));
    for (size_t n = 0; n < 64; ++n)
      assert(!cache::valid_cubin(std::span(good).first(n)));
    auto bad = good;
    bad[5] = 2;
    assert(!cache::valid_cubin(bad));
    bad = good;
    set64(bad, 40, UINT64_MAX);
    assert(!cache::valid_cubin(bad));
    bad = good;
    set64(bad, 64 + 24, 120);
    set64(bad, 64 + 32, 20);
    assert(!cache::valid_cubin(bad));
    bad = good;
    set16(bad, 60, 65535);
    assert(!cache::valid_cubin(bad));
    assert(cache::hash_parts({"ab", "c"}) != cache::hash_parts({"a", "bc"}));
    ++cases;
  }
  {
    Temp t;
    cache::BinaryCache c(t.root);
    auto key = cache::hash_parts({"a"}), other = cache::hash_parts({"b"});
    auto good = binary();
    assert(c.enabled() && c.load(key).empty());
    c.store(key, good);
    assert(c.load(key) == good);
    ++cases;
    auto path = t.root / (cache::hex(key) + ".cubin"),
         wrong = t.root / (cache::hex(other) + ".cubin");
    fs::copy_file(path, wrong);
    assert(c.load(other).empty());
    fs::remove(wrong);
    ++cases;
    auto data = input(path);
    data.back() ^= 1;
    output(path, data);
    assert(c.load(key).empty());
    c.store(key, good);
    assert(c.load(key) == good);
    ++cases;
    fs::create_hard_link(path, wrong);
    assert(c.load(key).empty());
    fs::remove(wrong);
    ++cases;
    chmod(path.c_str(), 0666);
    assert(c.load(key).empty());
    chmod(path.c_str(), 0600);
    ++cases;
    fs::create_symlink(path, wrong);
    assert(c.load(other).empty());
    fs::remove(wrong);
    ++cases;
    auto link = t.root.string() + "-link";
    fs::create_directory_symlink(t.root, link);
    assert(!cache::BinaryCache(link).enabled());
    fs::remove(link);
    ++cases;
    chmod(t.root.c_str(), 0777);
    assert(!cache::BinaryCache(t.root).enabled());
    chmod(t.root.c_str(), 0700);
    ++cases;
  }
  {
    Temp t;
    cache::BinaryCache c(t.root);
    auto good = binary(13U << 20);
    for (int i = 0; i < 5; ++i)
      c.store(cache::hash_parts({std::to_string(i)}), good);
    uintmax_t total = 0;
    for (auto &e : fs::directory_iterator(t.root))
      total += e.file_size();
    assert(total <= cache::max_disk_bytes && files(t.root) == 4);
    ++cases;
    assert(c.load(cache::hash_parts({"0"})).empty());
    ++cases;
  }
  {
    Temp t;
    cache::BinaryCache c(t.root);
    int fd = open((t.root / "foreign").c_str(), O_CREAT | O_WRONLY, 0600);
    assert(fd >= 0);
    assert(ftruncate(fd, cache::max_disk_bytes) == 0);
    close(fd);
    c.store(cache::hash_parts({"full"}), binary());
    assert(files(t.root) == 0 && fs::exists(t.root / "foreign"));
    ++cases;
  }
  {
    Temp t;
    cache::BinaryCache c(t.root);
    chmod(t.root.c_str(), 0500);
    c.store(cache::hash_parts({"readonly"}), binary());
    assert(files(t.root) == 0);
    chmod(t.root.c_str(), 0700);
    ++cases;
  }
  {
    Temp t;
    setenv("POISSON_SOLVER_KERNEL_CACHE_DIR", t.root.c_str(), 1);
    auto initial = compiles.load();
    one("source");
    assert(compiles == initial + 1);
    one("source");
    assert(compiles == initial + 1 && files(t.root) == 1);
    ++cases;
    one("source changed");
    one("source", "--gpu-architecture=sm_89");
    one("source", "--gpu-architecture=sm_86", "changed header");
    assert(compiles == initial + 4 && files(t.root) == 4);
    ++cases;
    one("FAIL");
    assert(live == 0 && files(t.root) == 4);
    ++cases;
    one("source", "--gpu-architecture=sm_86", "header", true);
    assert(live == 0 && compiles == initial + 5);
    ++cases;
    std::vector<std::thread> threads;
    for (int i = 0; i < 12; ++i)
      threads.emplace_back([] { one("parallel"); });
    for (auto &thread : threads)
      thread.join();
    assert(live == 0);
    auto prior = compiles.load();
    one("parallel");
    assert(compiles == prior);
    ++cases;
    unsetenv("POISSON_SOLVER_KERNEL_CACHE_DIR");
    one("source");
    assert(compiles == prior + 1 && live == 0);
    ++cases;
  }
  {
    Temp t;
    setenv("POISSON_SOLVER_KERNEL_CACHE_DIR", t.root.c_str(), 1);
    auto initial = compiles.load();
    one("filesystem # include <other.h>");
    one("filesystem # include <other.h>");
    one("preprocessor #inc\\\nlude <other.h>");
    one("preprocessor #inc\\\nlude <other.h>");
    one("source", "--include-path=/tmp");
    one("source", "--include-path=/tmp");
    assert(compiles == initial + 6 && files(t.root) == 0 && live == 0);
    ++cases;
    unsetenv("POISSON_SOLVER_KERNEL_CACHE_DIR");
  }
  assert(creates == destroys && live == 0);
  std::cout << "PASS " << cases << " host cache groups; creates=" << creates
            << " compiles=" << compiles << " destroys=" << destroys << "\n";
}
