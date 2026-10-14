"""Failures here guard against accidental dependence on another checkout."""

from pathlib import Path
import ast
import re
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from elmm_pipeline.build import build, _find_library, preprocessing_defines, PREPROCESSOR_SWITCHES
from elmm_pipeline.build_receipts import require_build_receipts


class BuildTests(unittest.TestCase):
    def stage_stub(self, root):
        elmm = root / "ELMM"
        poisfft = root / "PoisFFT"
        (elmm / "src").mkdir(parents=True)
        (poisfft / "src").mkdir(parents=True)
        (poisfft / "src" / "poisfft.f90").touch()
        return elmm, poisfft

    def test_poisfft_location_is_explicitly_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            elmm = root / "ELMM"
            (elmm / "src").mkdir(parents=True)
            (elmm / "src" / "SConstruct").write_text("t = env.Program()\n")
            with self.assertRaisesRegex(ValueError, "stage ELMM and PoisFFT as siblings"):
                build(elmm, root / "elsewhere", root / "build", Path("python"),
                      root / "lib", 1, 1, [])

    def test_precision_cannot_be_changed_through_extra_arguments(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for source in (root / "ELMM" / "src", root / "PoisFFT" / "src"):
                source.mkdir(parents=True)
            (root / "ELMM" / "src" / "SConstruct").write_text("t = env.Program()\n")
            (root / "PoisFFT" / "src" / "poisfft.f90").touch()
            with self.assertRaisesRegex(ValueError, "Unsupported build argument"):
                build(root / "ELMM", root / "PoisFFT", root / "build", Path("python"),
                      root / "lib", 1, 1, ["prec=sp"])

    def test_shared_preprocessor_configuration_tracks_every_upstream_switch(self):
        self.assertEqual(preprocessing_defines([]), ["DPREC"])
        for option, macro in PREPROCESSOR_SWITCHES.items():
            with self.subTest(option=option):
                self.assertEqual(preprocessing_defines([option + "=yes"]), ["DPREC", macro])
                self.assertEqual(preprocessing_defines([option + "=no"]), ["DPREC"])
        values = preprocessing_defines([key + "=yes" for key in PREPROCESSOR_SWITCHES])
        self.assertEqual(set(values), {"DPREC", *PREPROCESSOR_SWITCHES.values()})

    def test_macro_map_matches_pinned_native_scons_source(self):
        upstream = Path(__file__).resolve().parents[1] / ".cache/repos/elmm/src/SConstruct"
        if not upstream.is_file():
            self.skipTest("Run pipeline setup to verify against pinned upstream build logic")
        tree = ast.parse(upstream.read_text())
        aliases = {node.targets[0].id: node.value for node in tree.body
                   if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)}
        actual = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            condition = aliases.get(node.test.id, node.test) if isinstance(node.test, ast.Name) else node.test
            if not (isinstance(condition, ast.Compare) and len(condition.comparators) == 1
                    and isinstance(condition.ops[0], ast.Eq)
                    and isinstance(condition.comparators[0], ast.Constant)
                    and condition.comparators[0].value == "yes"):
                continue
            variable = condition.left
            if isinstance(variable, ast.Call) and isinstance(variable.func, ast.Name) and variable.func.id == "str" and len(variable.args) == 1:
                variable = variable.args[0]
            if not isinstance(variable, ast.Name):
                continue
            for statement in node.body:
                call = statement.value if isinstance(statement, ast.Expr) else None
                if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute) or call.func.attr != "Append":
                    continue
                for keyword in call.keywords:
                    if keyword.arg == "F90FLAGS" and isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
                        macros = re.findall(r"-D([A-Za-z_]\w*)", keyword.value.value)
                        if macros and variable.id != "mpi":
                            actual[variable.id] = macros
        self.assertTrue(actual, "Did not find upstream preprocessor switches")
        self.assertEqual(set(actual), set(PREPROCESSOR_SWITCHES))
        for option, macros in actual.items():
            with self.subTest(option=option):
                self.assertEqual(preprocessing_defines([option + "=yes"]), ["DPREC", *macros])

    def test_analysis_incompatible_build_options_fail_before_subprocess(self):
        options = ["prec=sp", "mpi=yes", "openmp=no", "comp=intel", "cray=yes", "ompp=yes", "dir=elsewhere",
                   "extra=-DMY_SCALE=3", "extra=-fdefault-real-8", "extra=-fdefault-integer-8",
                   "aggressive=yes", "sanitize=yes", "F90FLAGS=-DMY_SCALE=3", "customConfig=YES",
                   "unknown=yes", "no_separator"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            elmm, poisfft = self.stage_stub(root)
            (elmm / "src" / "SConstruct").write_text("t = env.Program()\n")
            for option in options:
                with self.subTest(option=option), patch("elmm_pipeline.build.subprocess.run") as process:
                    with self.assertRaisesRegex(ValueError, "Unsupported build argument"):
                        build(elmm, poisfft, root / "build", Path(sys.executable), root / "lib", 1, 1, [option])
                    process.assert_not_called()

    def test_safe_named_options_and_custom_hook_paths_preserved(self):
        options = ["customSB=yes", "SB=/tmp/custom hook.f90", "customPB=yes", "PB=/tmp/boundary.f90",
                   "lapack_blas=openblas", "fast=no", "lto=yes", "prof=yes", "diag=yes",
                   "cgal=no", "aggressive=no", "sanitize=no", "extra="]
        self.assertEqual(preprocessing_defines(options), ["DPREC", "CUSTOMSB", "CUSTOMPB"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            preprocessing_defines(["customSB=yes", "customSB=no"])

    def test_runtime_only_library_from_system_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary) / "libfftw3.so.3.7.11"
            runtime.touch()
            with patch("elmm_pipeline.build.ctypes.util.find_library", return_value="libfftw3.so.3"):
                found = _find_library("fftw3", {"libfftw3.so.3": runtime}, "gfortran")
            self.assertEqual(found, runtime)

    def test_patched_source_validation_precedes_build(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            elmm, poisfft = self.stage_stub(root)
            (elmm / "src" / "SConstruct").write_text("t = env.Program()\n")
            with self.assertRaisesRegex(ValueError, "patched Fortran source"):
                build(elmm, poisfft, root / "build", Path(sys.executable), root / "lib", 1, 1, [],
                      patched_sources=[elmm / "src" / "missing.f90"])

    def test_changed_sources_receive_interfaces_without_filename_assumptions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            elmm, poisfft = self.stage_stub(root)
            source = elmm / "src" / "update_state.f90"
            source.touch()
            interface = root / "outline.f90"
            interface.touch()
            (elmm / "src" / "SConstruct").write_text("t = env.Program()\n")

            def fake_build(command, **kwargs):
                (root / "build" / "ELMM").touch()
                return subprocess.CompletedProcess(command, 0)

            with patch("elmm_pipeline.build.subprocess.run", side_effect=fake_build):
                build(elmm, poisfft, root / "build", Path(sys.executable), root / "lib", 1, 1, [],
                      extra_sources=[interface], patched_sources=[source])
            script = (elmm / "src" / "SConstruct.pipeline").read_text()
            self.assertNotIn("advection", script)
            self.assertIn("pipeline_native_object.sources", script)
            self.assertIn("os.path.realpath(s.get_abspath())", script)
            self.assertIn(str(source), script)
            command = json.loads((root / "build" / "build-command.json").read_text())
            self.assertEqual(command["patched_sources"], [str(source)])
            self.assertEqual(command["preprocessing_defines"], ["DPREC"])

    def test_parallel_native_build_resolves_multiple_generated_modules(self):
        # Prefer the invoking interpreter; the explicitly pipeline-owned build
        # environment is also valid when testing the portable stdlib test suite.
        if importlib.util.find_spec("SCons"):
            python = Path(sys.executable)
        else:
            python = Path(os.environ.get("ELMM_SCONS_PYTHON", str(
                Path(__file__).resolve().parents[1] / ".cache" / "envs" / "build" / "bin" / "python")))
            if not python.is_file():
                self.skipTest("SCons is unavailable; run pipeline setup before this native build check")
            check = subprocess.run([str(python), "-c", "import SCons"], capture_output=True, check=False)
            if check.returncode:
                self.skipTest("The private interpreter does not contain SCons")
        if not all(shutil.which(name) for name in ("gfortran", "gcc", "g++")):
            self.skipTest("Native GNU Fortran/C/C++ compilers are unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            elmm, poisfft = self.stage_stub(root)
            (elmm / "src" / "SConstruct").write_text("""import os
env = Environment(ENV=os.environ, FORTRAN='gfortran', F90='gfortran', LINK='gfortran')
target_dir = ARGUMENTS['dir']
env.Append(FORTRANMODDIR=target_dir, F90PATH=[target_dir], LIBS=['stdc++'])
objs = env.Object(target=target_dir + '/renamed_native', source='state_update.f90')
objs += env.Object(target=target_dir + '/main', source='main.f90')
objs = [o for o in objs if not str(o).endswith(('.mod', '.smod'))]
t = env.Program(target=target_dir + '/ELMM', source=objs)
Default(t)
""")
            source = elmm / "src" / "state_update.f90"
            source.write_text("""module native_state
use second_outline, only: update_values
contains
subroutine advance(a)
real(8),intent(inout)::a(:)
call update_values(a)
end subroutine
end module
""")
            (elmm / "src" / "main.f90").write_text("""program check
use native_state, only: advance
implicit none
real(8)::a(5)
a=1
call advance(a)
if(any(a/=3)) stop 1
print *, 'passed'
end program
""")
            first = root / "first.f90"
            first.write_text("""module first_outline
use iso_c_binding
interface
subroutine update_c(a,n) bind(c)
import c_double,c_int
real(c_double)::a(*)
integer(c_int),value::n
end subroutine
end interface
end module
""")
            second = root / "second.f90"
            second.write_text("""module second_outline
use first_outline, only: update_c
contains
subroutine update_values(a)
real(8),intent(inout)::a(:)
call update_c(a,size(a))
end subroutine
end module
""")
            cpp = root / "implementation.cpp"
            cpp.write_text('extern "C" void update_c(double *a, int n) { for (int i=0;i<n;++i) a[i]+=2; }\n')
            with patch.dict(os.environ, {"FC": "gfortran", "CC": "gcc", "CXX": "g++"}):
                try:
                    binary = build(elmm, poisfft, root / "build", python, root / "lib", 4, 1, [],
                                   extra_sources=[second, cpp, first], patched_sources=[source])
                except RuntimeError:
                    self.fail((root / "build" / "build.log").read_text())
            run = subprocess.run([str(binary)], text=True, capture_output=True, check=False)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn("passed", run.stdout)
            self.assertTrue((root / "build" / "first_outline.mod").is_file())
            self.assertTrue((root / "build" / "second_outline.mod").is_file())
            self.assertFalse(list((root / "build" / "extra").glob("*.mod")))
            receipt = require_build_receipts(root / "build" / "build-receipts.json")
            self.assertEqual(len(receipt["units"]), 5)
            self.assertEqual({unit["source"] for unit in receipt["units"]},
                             {str(path) for path in (source, elmm / "src" / "main.f90", first, second, cpp)})
            self.assertFalse(receipt["placement_authority"])
            # A fresh receipt directory prevents reusing old commands for a
            # no-op incremental build; ordinary build behavior stays intact.
            with patch.dict(os.environ, {"FC": "gfortran", "CC": "gcc", "CXX": "g++"}):
                build(elmm, poisfft, root / "build", python, root / "lib", 4, 1, [],
                      extra_sources=[second, cpp, first], patched_sources=[source])
            with self.assertRaisesRegex(ValueError, "unavailable"):
                require_build_receipts(root / "build" / "build-receipts.json")


if __name__ == "__main__":
    unittest.main()
