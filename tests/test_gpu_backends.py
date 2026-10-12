"""Generic GPU contract tests; native runs require explicit GPU opt-in.

ELMM_LOCAL_PYTHON selects the compiler interpreter.
ELMM_COMPILER_ROOT selects the local compiler. ELMM_GPU_TEST=1 enables native
GPU compilation/execution and checks the runtime's transfer/kernel evidence.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from elmm_pipeline.gpu_backend import transform

FLOAT64 = """module unrelated
implicit none
contains
subroutine flux(n, field, result)
integer, intent(in) :: n
real(kind=8), contiguous, intent(in) :: field(:)
real(kind=8), contiguous, intent(inout) :: result(:)
integer :: cell
real(kind=8) :: alpha, beta
alpha = 2.0_8
do cell=1,n
result(cell) = field(cell)*alpha
end do
beta = 3.0_8
do cell=2,n-1
result(cell) = result(cell)+beta
end do
end subroutine
end module
"""

FLOAT32 = """module unrelated
implicit none
contains
subroutine flux(nx, ny, field, result)
integer, intent(in) :: nx, ny
real(kind=4), contiguous, intent(in) :: field(:,:)
real(kind=4), contiguous, intent(inout) :: result(:,:)
integer :: column, row
real(kind=4) :: work
do row=0,ny
do column=-2,nx
work=field(column+3,row+2)+field(column+4,row+1)
result(column+4,row+2)=work*0.5_4
end do
end do
do row=1,ny
do column=-1,nx
result(column+4,row+2)=result(column+4,row+2)+2.0_4
end do
end do
end subroutine
end module
"""

INTEGER = """module unrelated
implicit none
contains
subroutine flux(nx,ny,nz,field,result)
integer, intent(in) :: nx,ny,nz
integer, contiguous, intent(in) :: field(:,:,:)
integer, contiguous, intent(inout) :: result(:,:,:)
integer :: x,y,z
do z=1,nz
do y=1,ny
do x=1,nx
result(x,y,z)=field(x,y,z)*2+3
end do
end do
end do
end subroutine
end module
"""

DRIVER64 = """program compare
use unrelated, only: flux
implicit none
real(kind=8):: x(103),y(103),expected(103)
integer :: i, rep
x=[(real(i,8),i=1,103)]
do rep=1,2
y=-19.0_8
expected=y
expected(1:100)=x(1:100)*2
expected(2:99)=expected(2:99)+3
call flux(100,x,y)
if(any(y/=expected)) error stop 'mismatch'
x=x+1
end do
print *, 'PASS'
end program
"""
DRIVER32 = """program compare
use unrelated, only: flux
implicit none
real(kind=4):: x(-3:11,-1:7),y(-3:11,-1:7),expected(-3:11,-1:7)
integer :: i,j,rep
do rep=1,2
do j=-1,7
do i=-3,11
x(i,j)=real(i+j*10+rep,4)
end do
end do
y=-19.0_4
expected=y
do j=0,6
do i=-2,10
expected(i,j)=(x(i-1,j)+x(i,j-1))*0.5_4
end do
end do
expected(-1:10,1:6)=expected(-1:10,1:6)+2.0_4
call flux(10,6,x,y)
if(any(y/=expected)) error stop 'float32 mismatch or halo changed'
end do
print *, 'PASS'
end program
"""
DRIVERINTEGER = """program compare
use unrelated, only: flux
implicit none
integer :: x(11,7,5),y(11,7,5),expected(11,7,5),i,j,k,rep
do rep=1,2
do k=1,5
do j=1,7
do i=1,11
x(i,j,k)=i+j*10+k*100+rep
end do
end do
end do
y=-19
expected=y
expected(1:10,1:6,1:4)=x(1:10,1:6,1:4)*2+3
call flux(10,6,4,x,y)
if(any(y/=expected)) error stop 'integer mismatch or halo changed'
end do
print *, 'PASS'
end program
"""


class GPUBackendTests(unittest.TestCase):
    def tools(self):
        result = [(name, Path(os.environ[f"ELMM_{name.upper()}_PYTHON"])) for name in ("local",)
                  if os.environ.get(f"ELMM_{name.upper()}_PYTHON")]
        result = [(name, python) for name, python in result if name != "local" or os.environ.get("ELMM_COMPILER_ROOT")]
        if not result:
            self.skipTest("Set ELMM_LOCAL_PYTHON and ELMM_COMPILER_ROOT")
        return result

    def generate(self, directory, tool, python, source, entry="flux"):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "input.f90"
        path.write_text(source)
        result = transform(tool, path, python, directory / "generated", entry=entry,
                           compiler_root=Path(os.environ["ELMM_COMPILER_ROOT"]))
        self.assertEqual(path.read_text(), source)
        self.assertEqual(result["extra_sources"], [result["generated_path"]])
        self.assertEqual(result["gpu"]["data_regions"], 1)
        return result

    def compile_run(self, directory, metadata, driver):
        from elmm_pipeline.gpu_toolchain import compile_gpu_sources
        options = {key: os.environ[env] for key, env in
                   (("nvcc", "ELMM_NVCC"),
                    ("cuda_host_cxx", "ELMM_CUDA_HOST_CXX"), ("architecture", "ELMM_GPU_ARCH")) if os.environ.get(env)}
        try:
            native = compile_gpu_sources(metadata["gpu_sources"], directory / "build", options=options)
        except RuntimeError as exc:
            self.fail(str(exc) + "\n" + (directory / "build/gpu/gpu-build.log").read_text())
        (directory / "driver.f90").write_text(driver)
        command = ["gfortran", "-O2", "-fopenmp", "-ffree-line-length-none", "-fcheck=all",
                   *metadata["extra_sources"], str(directory / "driver.f90"), *native["objects"],
                   *["-L" + path for path in native["library_dirs"]],
                   *["-Wl,-rpath," + path for path in native["rpaths"]],
                   *native["link_flags"], *["-l" + lib for lib in native["libraries"]],
                   "-o", str(directory / "check")]
        compiled = subprocess.run(command, cwd=directory, capture_output=True, text=True, check=False)
        self.assertEqual(compiled.returncode, 0, compiled.stderr)
        executed = subprocess.run([str(directory / "check")], cwd=directory,
                                  env=dict(os.environ, FORT_RUNTIME_TRACE="1"),
                                  capture_output=True, text=True, check=False)
        (directory / "execution.log").write_text(executed.stdout + executed.stderr)
        self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)
        self.assertIn("PASS", executed.stdout)
        return executed

    def test_unrelated_sources_generate_real_gpu_regions(self):
        for tool, python in self.tools():
            for name, source in (("double", FLOAT64), ("single", FLOAT32), ("integer", INTEGER)):
                with self.subTest(tool=tool, case=name), tempfile.TemporaryDirectory() as temporary:
                    result = self.generate(Path(temporary), tool, python, source)
                    generated = Path(result["gpu_sources"][0]["source"]).read_text().lower()
                    self.assertIn("__global__", generated)
                    self.assertNotRegex(generated, r"\bcpp_start_hot\s*\(")
                    self.assertRegex(generated, r"static\s+__global__\s+void\s+kernel_region_\d+_device\s*\(")


    def test_optional_markers_and_equal_loop_bounds_use_real_fusion(self):
        source = FLOAT64.replace("do cell=2,n-1", "do cell=1,n")
        source = source.replace("alpha = 2.0_8", "!$f2c kernel\nalpha = 2.0_8")
        for tool, python in self.tools():
            with self.subTest(tool=tool), tempfile.TemporaryDirectory() as temporary:
                result = self.generate(Path(temporary), tool, python, source)
                self.assertEqual(result["gpu"]["kernel_regions"], 1)


    @unittest.skipUnless(os.environ.get("ELMM_GPU_TEST") == "1", "Set ELMM_GPU_TEST=1 for actual NVIDIA execution")
    def test_real_gpu_numeric_kinds_ranks_bounds_halos_and_fresh_inputs(self):
        for tool, python in self.tools():
            for name, source, driver in (("double", FLOAT64, DRIVER64), ("single", FLOAT32, DRIVER32),
                                         ("integer", INTEGER, DRIVERINTEGER)):
                with self.subTest(tool=tool, case=name), tempfile.TemporaryDirectory() as temporary:
                    directory = Path(temporary)
                    metadata = self.generate(directory, tool, python, source)
                    result = self.compile_run(directory, metadata, driver)
                    self.assertEqual(result.stderr.count("FORT_RUNTIME upload bytes="), 4)
                    self.assertEqual(result.stderr.count("FORT_RUNTIME download bytes="), 2)
                    self.assertEqual(result.stderr.count("FORT_RUNTIME kernel"), 2 * metadata["gpu"]["kernel_regions"])

    @unittest.skipUnless(os.environ.get("ELMM_GPU_TEST") == "1", "Set ELMM_GPU_TEST=1 for actual NVIDIA execution")
    def test_output_only_partial_and_empty_regions_preserve_halos(self):
        source = FLOAT64.replace("intent(inout)", "intent(out)")
        driver = DRIVER64.replace("integer :: i, rep", "real(kind=8) :: empty_in(0), empty_out(0)\ninteger :: i, rep")
        driver = driver.replace("print *, 'PASS'", "y=-19.0_8\ncall flux(0,x,y)\nif(any(y/=-19.0_8)) error stop 'empty region changed halos'\ncall flux(0,empty_in,empty_out)\nprint *, 'PASS'")
        for tool, python in self.tools():
            with self.subTest(tool=tool), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                metadata = self.generate(directory, tool, python, source)
                self.compile_run(directory, metadata, driver)

    @unittest.skipUnless(os.environ.get("ELMM_GPU_TEST") == "1", "Set ELMM_GPU_TEST=1 for multi-entry CUDA linkage")
    def test_thirteen_cuda_entries_link_without_export_collisions(self):
        tools = dict(self.tools())
        if "local" not in tools:
            self.skipTest("Local compiler interpreter and root required")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            units, interfaces, uses, calls = [], [], [], []
            source = FLOAT64[:FLOAT64.index("beta = 3")] + "end subroutine\nend module\n"
            for number in range(13):
                name = f"flux_{number}"
                module = f"unrelated_{number}"
                result = self.generate(directory / name, "local", tools["local"],
                                       source.replace("flux", name).replace("unrelated", module), entry=name)
                units.extend(result["gpu_sources"])
                interfaces.extend(result["extra_sources"])
                uses.append(f"use {module}, only: {name}")
                calls.append(f"call {name}(3,x,y)\nif(any(y/=x*2)) error stop 'namespace mismatch'")
            driver = "program compare\n" + "\n".join(uses) + "\nreal(kind=8)::x(3),y(3)\nx=3\n" + "\n".join(calls) + "\nprint *, 'PASS'\nend program\n"
            self.compile_run(directory, {"gpu_sources": units, "extra_sources": interfaces}, driver)


if __name__ == "__main__":
    unittest.main()
