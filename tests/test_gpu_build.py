"""GPU build validation and opt-in native GNU/NVIDIA interoperability checks."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from elmm_pipeline.build import build
from elmm_pipeline.gpu_toolchain import (
    _architecture, compile_gpu_sources,
    discover_toolchain, probe_gpu, validate_gpu_sources,
)


class GPUConfigurationTests(unittest.TestCase):
    def discover_cuda(self, root, target="x86_64-linux-gnu", query_result=None, query_error=None, **kwargs):
        result = query_result or subprocess.CompletedProcess([], 0, target + "\n", "")
        with patch("elmm_pipeline.gpu_toolchain.probe_gpu", return_value={"devices": [], "error": "hidden"}), \
             patch("elmm_pipeline.gpu_toolchain._executable", side_effect=lambda explicit, *args: str(explicit)), \
             patch("elmm_pipeline.gpu_toolchain._version", return_value="mock compiler"), \
             patch("elmm_pipeline.gpu_toolchain.platform.machine", return_value="x86_64"), \
             patch("elmm_pipeline.gpu_toolchain.platform.system", return_value="Linux"), \
             patch("elmm_pipeline.gpu_toolchain.subprocess.run", return_value=result,
                   side_effect=query_error) as query:
            config = discover_toolchain(languages=("cuda",), nvcc=root / "bin/nvcc",
                                        cuda_host=root / "bin/host-c++", arch="90", **kwargs)
        return config, query

    def test_cuda_runtime_matches_selected_host_target_and_preserves_native_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "targets/x86_64-linux/lib"
            directory.mkdir(parents=True)
            (directory / "libcudart.so").touch()
            (root / "lib64").symlink_to(directory, target_is_directory=True)
            config, query = self.discover_cuda(root, "x86_64-pc-linux-gnu")
            self.assertEqual(config["cuda_library_dir"], str((root / "lib64").resolve()))
            self.assertEqual(config["cuda_runtime_discovery"]["target"], "x86_64-pc-linux-gnu")
            query.assert_called_once_with([str(root / "bin/host-c++"), "-dumpmachine"],
                                          capture_output=True, text=True, check=False, timeout=15)

    def test_cross_host_target_uses_its_runtime_instead_of_native_lib64(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            native = root / "lib64"
            native.mkdir()
            (native / "libcudart.so").touch()
            cross = root / "targets/aarch64-linux/lib"
            cross.mkdir(parents=True)
            (cross / "libcudart.so").touch()
            config, _ = self.discover_cuda(root, "aarch64-unknown-linux-gnu")
            self.assertEqual(config["cuda_library_dir"], str(cross))
            self.assertEqual(config["cuda_runtime_discovery"]["method"], "host_target")

    def test_redhat_native_target_uses_standard_cuda_target_and_lib64_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "targets/x86_64-linux/lib"
            directory.mkdir(parents=True)
            (directory / "libcudart.so").touch()
            (root / "lib64").symlink_to(directory, target_is_directory=True)
            config, _ = self.discover_cuda(root, "x86_64-redhat-linux")
            self.assertEqual(config["cuda_library_dir"], str(directory))
            self.assertEqual(config["cuda_runtime_discovery"]["target"], "x86_64-redhat-linux")
            self.assertEqual(config["cuda_runtime_discovery"]["method"], "host_target")

    def test_redhat_target_flat_runtime_is_native_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "lib64").mkdir()
            (root / "lib64/libcudart.so").touch()
            config, _ = self.discover_cuda(root, "x86_64-redhat-linux")
            self.assertEqual(config["cuda_runtime_discovery"]["method"], "native_flat_layout")
            with self.assertRaisesRegex(RuntimeError, "host target 'aarch64-redhat-linux'"):
                self.discover_cuda(root, "aarch64-redhat-linux")

    def test_redhat_native_target_rejects_foreign_target_lib64_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            foreign = root / "targets/aarch64-linux/lib"
            foreign.mkdir(parents=True)
            (foreign / "libcudart.so").touch()
            (root / "lib64").symlink_to(foreign, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "host target 'x86_64-redhat-linux'"):
                self.discover_cuda(root, "x86_64-redhat-linux")

    def test_explicit_cuda_runtime_override_requires_library_and_skips_target_query(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "custom runtime"
            directory.mkdir()
            (directory / "libcudart.so").touch()
            for kwargs in ({"cuda_runtime_dir": directory}, {"options": {"cuda_runtime_dir": str(directory)}},
                           {"options": {"cuda_library_dir": str(directory)}}):
                with self.subTest(kwargs=kwargs):
                    config, query = self.discover_cuda(root, target="unsupported", **kwargs)
                    self.assertEqual(config["cuda_library_dir"], str(directory))
                    self.assertEqual(config["cuda_runtime_discovery"]["method"], "explicit")
                    query.assert_not_called()
            # The compiler triple does not distinguish SBSA from other ARM platforms.
            sbsa = root / "targets/sbsa-linux/lib"
            sbsa.mkdir(parents=True)
            (sbsa / "libcudart.so").touch()
            config, query = self.discover_cuda(root, target="aarch64-linux-gnu", cuda_runtime_dir=sbsa)
            self.assertEqual(config["cuda_library_dir"], str(sbsa))
            query.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, "Explicit CUDA runtime directory"):
                self.discover_cuda(root, cuda_runtime_dir=root / "missing")

    def test_flat_runtime_is_native_only_and_foreign_target_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "lib64"
            directory.mkdir()
            (directory / "libcudart.so").touch()
            config, _ = self.discover_cuda(root)
            self.assertEqual(config["cuda_runtime_discovery"]["method"], "native_flat_layout")
            with self.assertRaisesRegex(RuntimeError, "host target 'aarch64-linux-gnu'"):
                self.discover_cuda(root, target="aarch64-linux-gnu")
            (directory / "libcudart.so").unlink()
            directory.rmdir()
            foreign = root / "targets/aarch64-linux/lib"
            foreign.mkdir(parents=True)
            (foreign / "libcudart.so").touch()
            directory.symlink_to(foreign, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "host target 'x86_64-linux-gnu'"):
                self.discover_cuda(root)

    def test_missing_and_invalid_host_targets_require_explicit_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "supply cuda_runtime_dir"):
                self.discover_cuda(root)
            with self.assertRaisesRegex(RuntimeError, "Invalid CUDA host compiler target"):
                self.discover_cuda(root, target="../../another-target")

    def test_host_target_query_failures_do_not_select_an_available_native_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "lib64").mkdir()
            (root / "lib64/libcudart.so").touch()
            for result in (subprocess.CompletedProcess([], 1, "x86_64-linux-gnu", "failed"),
                           subprocess.CompletedProcess([], 0, "", "")):
                with self.subTest(result=result), self.assertRaisesRegex(RuntimeError, "Invalid CUDA host compiler target"):
                    self.discover_cuda(root, query_result=result)
            for error in (subprocess.TimeoutExpired("host-c++", 15), OSError("unavailable")):
                with self.subTest(error=error), self.assertRaisesRegex(RuntimeError, "Could not query CUDA host compiler target"):
                    self.discover_cuda(root, query_error=error)

    def test_nonstandard_abi_requires_exact_target_directory_or_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            conventional = root / "targets/x86_64-linux/lib"
            conventional.mkdir(parents=True)
            (conventional / "libcudart.so").touch()
            # Neither a target directory nor a native flat layout proves ABI compatibility.
            (root / "lib64").mkdir()
            (root / "lib64/libcudart.so").touch()
            for target in ("x86_64-linux-gnux32", "x86_64-linux-musl",
                           "x86_64-redhat-linux-gnux32", "x86_64-redhat-linux-musl"):
                with self.subTest(target=target):
                    with self.assertRaisesRegex(RuntimeError, "supply cuda_runtime_dir"):
                        self.discover_cuda(root, target=target)
                    exact = root / "targets" / target / "lib"
                    exact.mkdir(parents=True)
                    (exact / "libcudart.so").touch()
                    config, _ = self.discover_cuda(root, target=target)
                    self.assertEqual(config["cuda_library_dir"], str(exact))

    def test_compute_capability_normalization_rejects_compiler_flags(self):
        for value in ("86", "8.6", "sm_86", "cc86"):
            self.assertEqual(_architecture(value), "86")
        self.assertEqual(_architecture("sm_100a"), "100a")
        for value in ("native", "86 -O0", "86;true", "", "0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _architecture(value)

    def test_failed_gpu_query_retains_diagnostic(self):
        result = subprocess.CompletedProcess([], 1, "", "Failed to initialize NVML: Insufficient Permissions")
        with patch("elmm_pipeline.gpu_toolchain.shutil.which", return_value="/usr/bin/nvidia-smi"), \
             patch("elmm_pipeline.gpu_toolchain.subprocess.run", return_value=result):
            device = probe_gpu()
        self.assertFalse(device["available"])
        self.assertIn("Insufficient Permissions", device["error"])

    def test_device_query_records_name_driver_and_architecture(self):
        result = subprocess.CompletedProcess([], 0, "NVIDIA GeForce RTX 3060 Ti, 580.1, 8.6\n", "")
        with patch("elmm_pipeline.gpu_toolchain.shutil.which", return_value="/usr/bin/nvidia-smi"), \
             patch("elmm_pipeline.gpu_toolchain.subprocess.run", return_value=result):
            device = probe_gpu()
        self.assertTrue(device["available"])
        self.assertEqual(device["devices"][0]["compute_capability"], "8.6")

    def test_hidden_device_requires_explicit_compile_architecture(self):
        with patch("elmm_pipeline.gpu_toolchain.probe_gpu", return_value={"available": False, "devices": [], "error": "denied"}):
            with self.assertRaisesRegex(RuntimeError, "explicit architecture"):
                discover_toolchain(languages=())
            config = discover_toolchain(languages=(), arch="86")
        self.assertEqual(config["architecture"], "86")

    def test_source_descriptors_reject_missing_wrong_or_duplicate_units(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "kernel.cu"
            source.touch()
            unit = {"source": source, "language": "cuda", "tool": "local"}
            self.assertEqual(validate_gpu_sources([unit])[0]["source"], str(source))
            for units in ([unit, unit], [{"source": source, "language": "openacc"}], [{"source": source}]):
                with self.subTest(units=units), self.assertRaises(ValueError):
                    validate_gpu_sources(units)


    def test_cuda_compilation_records_selected_host_architecture_and_defines(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "kernel.cu"
            source.write_text('extern "C" void kernel() {}\n')
            config = {"architecture": "86", "nvcc": "/cuda/nvcc", "cuda_host_cxx": "/usr/bin/g++-14", "cuda_library_dir": "/cuda/lib64"}

            def compile_object(command, **kwargs):
                Path(command[command.index("-o") + 1]).touch()
                return subprocess.CompletedProcess(command, 0)

            with patch("elmm_pipeline.gpu_toolchain.subprocess.run", side_effect=compile_object):
                manifest = compile_gpu_sources([{"source": source, "language": "cuda"}], root / "build", defines=["DPREC"], toolchain=config)
            command = manifest["commands"][0]
            self.assertIn("-arch=sm_86", command)
            self.assertIn("-DDPREC", command)
            self.assertEqual(command[command.index("-ccbin") + 1], "/usr/bin/g++-14")
            self.assertIn("-Xcompiler=-fPIC,-fopenmp", command)
            self.assertEqual(manifest["libraries"], ["cudart", "stdc++"])
            saved = json.loads((root / "build/gpu/gpu-build-command.json").read_text())
            self.assertEqual(saved["sources"][0]["sha256"], manifest["sources"][0]["sha256"])

    def test_gpu_objects_are_explicit_scons_link_dependencies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            elmm, poisfft = root / "ELMM", root / "PoisFFT"
            (elmm / "src").mkdir(parents=True)
            (poisfft / "src").mkdir(parents=True)
            (poisfft / "src/poisfft.f90").touch()
            (elmm / "src/SConstruct").write_text("t = env.Program()\n")
            source = root / "kernel.cu"
            source.touch()
            manifest = {"objects": [str(root / "kernel.o")], "library_dirs": ["/cuda/lib64"],
                        "libraries": ["cudart"], "rpaths": ["/cuda/lib64"], "link_flags": []}

            def scons(command, **kwargs):
                (root / "build/ELMM").touch()
                return subprocess.CompletedProcess(command, 0)

            with patch("elmm_pipeline.build.compile_gpu_sources", return_value=manifest) as compile_gpu, \
                 patch("elmm_pipeline.build.subprocess.run", side_effect=scons):
                build(elmm, poisfft, root / "build", Path(sys.executable), root / "lib", 2, 1, [],
                      gpu_sources=[{"source": source, "language": "cuda"}], gpu_options={"architecture": "86"})
            compile_gpu.assert_called_once()
            script = (elmm / "src/SConstruct.pipeline").read_text()
            self.assertIn("env.File(p)", script)
            self.assertIn("env.AppendUnique(RPATH=['/cuda/lib64'])", script)
            self.assertIn("env.AppendUnique(LIBS=['cudart'])", script)
            self.assertIn("pipeline_cuda_stdcxx", script)
            self.assertIn("env.File(pipeline_cuda_stdcxx)", script)
            self.assertNotIn("source=" + repr(str(source)), script)
            self.assertEqual(json.loads((root / "build/build-command.json").read_text())["gpu"], manifest)


@unittest.skipUnless(os.environ.get("ELMM_GPU_NATIVE_TESTS") == "1", "Set ELMM_GPU_NATIVE_TESTS=1 for native GPU execution")
class NativeGPUInteropTests(unittest.TestCase):
    """Exercise the actual public build API, GNU modules, GPU runtime and device."""

    def test_gnu_main_and_generated_wrapper_call_cuda(self):
        python = Path(__file__).resolve().parents[1] / ".cache/envs/build/bin/python"
        compiler = shutil.which("gfortran-15") or shutil.which("gfortran")
        if not python.is_file() or not compiler:
            self.skipTest("Run pipeline setup and install GNU Fortran first")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            elmm, poisfft = root / "ELMM", root / "PoisFFT"
            (elmm / "src").mkdir(parents=True)
            (poisfft / "src").mkdir(parents=True)
            (poisfft / "src/poisfft.f90").touch()
            (elmm / "src/SConstruct").write_text("""import os
env = Environment(ENV=os.environ, FORTRAN=os.environ['FC'], F90=os.environ['FC'], LINK=os.environ['FC'])
target_dir = ARGUMENTS['dir']
env.Append(FORTRANMODDIR=target_dir, F90PATH=[target_dir], F90FLAGS=['-fopenmp'], LIBS=['stdc++'], LINKFLAGS=['-fopenmp'])
objs = env.Object(target=target_dir + '/main', source='main.f90')
t = env.Program(target=target_dir + '/ELMM', source=objs)
Default(t)
""")
            (elmm / "src/main.f90").write_text("""program check
use host_wrapper
use omp_lib
implicit none
real(8)::a(16)
integer::threads,calls,i
a=1
threads=0
calls=0
!$omp parallel num_threads(4) reduction(+:threads)
threads=threads+1
!$omp single
call update(a)
calls=calls+1
!$omp end single
!$omp end parallel
if(threads/=4.or.calls/=1) stop 3
!$omp parallel num_threads(4)
!$omp single
call update(a)
!$omp end single
!$omp end parallel
!$omp parallel do num_threads(4)
do i=1,size(a)
a(i)=a(i)+1
end do
!$omp end parallel do
if(any(a/=6)) stop 1
print *, 'GPU execution passed'
end program
""")
            wrapper = root / "wrapper.f90"
            wrapper.write_text("""module host_wrapper
use iso_c_binding
interface
subroutine gpu_add(a,n) bind(c)
import c_double,c_int
integer(c_int),value::n
real(c_double)::a(n)
end subroutine
end interface
contains
subroutine update(a)
real(c_double),intent(inout)::a(:)
call gpu_add(a,size(a))
end subroutine
end module
""")
            source = root / "kernel.cu"
            source.write_text("""#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
static void check(cudaError_t code) { if (code != cudaSuccess) { std::fprintf(stderr, "%s\\n", cudaGetErrorString(code)); std::exit(1); } }
__global__ void add(double *a, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) a[i]+=2; }
extern "C" void gpu_add(double *a, int n) {
double *d; check(cudaMalloc(&d,n*sizeof(double)));
check(cudaMemcpy(d,a,n*sizeof(double),cudaMemcpyHostToDevice));
add<<<1,32>>>(d,n); check(cudaGetLastError()); check(cudaDeviceSynchronize());
check(cudaMemcpy(a,d,n*sizeof(double),cudaMemcpyDeviceToHost)); check(cudaFree(d));
std::fprintf(stderr,"native CUDA kernel completed\\n");
}
""")
            with patch.dict(os.environ, {"FC": compiler}):
                binary = build(elmm, poisfft, root / "build", python, root / "lib", 4, 1, [],
                               extra_sources=[wrapper], patched_sources=[elmm / "src/main.f90"],
                               gpu_sources=[{"source": source, "language": "cuda"}], gpu_options={"architecture": "86"})
                run = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertNotIn("Another OpenMP runtime", run.stderr)
            self.assertIn("GPU execution passed", run.stdout)
            self.assertIn("native CUDA kernel completed", run.stderr)
            self.assertTrue((root / "build/host_wrapper.mod").is_file())
            manifest = json.loads((root / "build/build-command.json").read_text())
            self.assertNotIn("nvomp", manifest["host_runtime"]["gnu_openmp"])

if __name__ == "__main__":
    unittest.main()
