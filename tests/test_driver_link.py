"""Diagnostic relinking preserves the fresh build's complete command semantics."""
import json
import shlex
import sys
import tempfile
import unittest
from pathlib import Path

from elmm_pipeline.campaigns.driver_link import derive_driver_link


class DriverLinkTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="driver link ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cwd = self.root / "application source"
        self.build = self.root / "build output"
        self.cwd.mkdir()
        self.build.mkdir()
        (self.build / "P").mkdir()
        self.compiler = Path(sys.executable).resolve()
        self.binary = self.build / "ELMM"
        self.original_source = self.cwd / "original main.f90"
        self.original_object = self.build / "main.o"
        self.library_main = self.build / "P" / "main.o"
        self.source = self.root / "complete fields.f90"
        self.object = self.root / "diagnostic main.o"
        self.output = self.root / "diagnostic executable"
        for path in (self.binary, self.original_source, self.original_object, self.library_main, self.source):
            path.write_text("retained " + path.name)
        self.compile = [str(self.compiler), "-O3", "-fopenmp", "-cpp", "-DDPREC", "-fbacktrace", "-g",
                        "-I", "module path", "-J" + str(self.build), "-o", str(self.original_object),
                        "-c", str(self.original_source)]
        self.link = [str(self.compiler), "-o", str(self.binary), "-O3", "-fopenmp",
                     str(self.library_main), str(self.original_object), "-Wl,--disable-new-dtags",
                     "-Llibrary path", "-lfftw3_omp", "-lfftw3", "-lpthread"]
        self.receipt = self.build / "build-command.json"
        self.log = self.build / "build.log"
        self.receipt.write_text(json.dumps({"cwd": str(self.cwd), "argv": ["python", "-m", "SCons", "-j", "2"]}))
        self.write_log()

    def write_log(self, compile=None, link=None, extra=()):
        commands = [self.compile if compile is None else compile, self.link if link is None else link, *extra]
        self.log.write_text("SCons: building targets ...\n" + "\n".join(shlex.join(row) for row in commands) + "\n")

    def plan(self, **kwargs):
        return derive_driver_link(self.binary, fortran=self.compiler, source=self.source,
                                  object_path=self.object, output=self.output, **kwargs)

    def test_native_commands_change_only_main_source_object_and_output(self):
        plan = self.plan()
        expected_compile = list(self.compile)
        expected_compile[-1] = str(self.source)
        expected_compile[expected_compile.index("-o") + 1] = str(self.object)
        expected_link = list(self.link)
        expected_link[expected_link.index(str(self.original_object))] = str(self.object)
        expected_link[expected_link.index("-o") + 1] = str(self.output)
        self.assertEqual(plan["compile_argv"], expected_compile)
        self.assertEqual(plan["link_argv"], expected_link)
        self.assertEqual(plan["cwd"], str(self.cwd))
        self.assertEqual(plan["provenance"]["original_compile_argv"], self.compile)
        self.assertEqual(plan["provenance"]["original_link_argv"], self.link)
        self.assertIn(str(self.library_main), plan["link_argv"])
        self.assertFalse(self.object.exists())
        self.assertFalse(self.output.exists())

    def test_gpu_generated_objects_and_library_order_are_preserved(self):
        self.link[5:5] = ["extra/worker.o", "gpu/runtime.o", "gpu/kernel.o"]
        self.link += ["/cuda path/libcudart.so", "-Wl,--push-state,--no-as-needed", "-lstdc++", "-Wl,--pop-state"]
        self.write_log()
        plan = self.plan()
        changes = {self.link.index(str(self.original_object)): str(self.object),
                   self.link.index("-o") + 1: str(self.output)}
        self.assertEqual(plan["link_argv"], [changes.get(index, arg) for index, arg in enumerate(self.link)])

    def test_relative_original_paths_keep_recorded_cwd_and_module_arguments(self):
        self.compile[-1] = "original main.f90"
        self.compile[self.compile.index("-o") + 1] = "../build output/main.o"
        self.link[self.link.index(str(self.original_object))] = "../build output/main.o"
        self.link[self.link.index(str(self.library_main))] = "../build output/P/main.o"
        self.link[self.link.index("-o") + 1] = "../build output/ELMM"
        self.write_log()
        plan = self.plan()
        self.assertEqual(plan["cwd"], str(self.cwd))
        self.assertIn("../build output/P/main.o", plan["link_argv"])
        self.assertIn("module path", plan["compile_argv"])
        self.assertEqual(plan["provenance"]["original_main_source"], str(self.original_source))

    def test_identical_repeated_log_commands_are_one_unique_plan(self):
        self.write_log(extra=(self.compile, self.link))
        self.assertEqual(self.plan()["provenance"]["original_link_argv"], self.link)

    def test_missing_and_distinct_ambiguous_commands_reject(self):
        for role in ("compile", "link"):
            with self.subTest(role=role, failure="missing"):
                commands = [self.link] if role == "compile" else [self.compile]
                self.log.write_text("\n".join(shlex.join(row) for row in commands))
                with self.assertRaisesRegex(ValueError, "unique application main"):
                    self.plan()
            with self.subTest(role=role, failure="ambiguous"):
                other = [*(self.compile if role == "compile" else self.link), "-g"]
                self.write_log(extra=(other,))
                with self.assertRaisesRegex(ValueError, "unique application main"):
                    self.plan()

    def test_wrong_compiler_and_wrong_compile_mode_reject(self):
        fake = self.root / "different compiler"
        fake.write_text("not executed")
        other = [str(fake), *self.compile[1:]]
        self.write_log(compile=other)
        with self.assertRaisesRegex(ValueError, "supplied Fortran compiler"):
            self.plan()
        self.write_log(compile=[arg for arg in self.compile if arg != "-c"])
        with self.assertRaisesRegex(ValueError, "mode is malformed"):
            self.plan()

    def test_malformed_shell_response_and_duplicate_main_commands_reject(self):
        variants = [([*self.compile, "&&", "echo", "unexpected"], self.link, "direct argv"),
                    ([*self.compile, "@flags"], self.link, "direct argv"),
                    ([*self.compile, "-o"], self.link, "malformed output"),
                    (self.compile, [*self.link, str(self.original_object)], "exactly one application main")]
        for compile, link, reason in variants:
            with self.subTest(reason=reason):
                self.write_log(compile=compile, link=link)
                with self.assertRaisesRegex(ValueError, reason):
                    self.plan()
        self.log.write_text(shlex.join(self.compile) + " '\n" + shlex.join(self.link))
        with self.assertRaisesRegex(ValueError, "malformed command"):
            self.plan()
        self.log.write_text(shlex.join(self.compile) + " 2>errors.log\n" + shlex.join(self.link))
        with self.assertRaisesRegex(ValueError, "direct argv"):
            self.plan()

    def test_missing_or_multiple_original_sources_reject(self):
        self.write_log(compile=self.compile[:-1])
        with self.assertRaisesRegex(ValueError, "unique original Fortran"):
            self.plan()
        self.write_log(compile=[*self.compile, str(self.source)])
        with self.assertRaisesRegex(ValueError, "unique original Fortran"):
            self.plan()

    def test_existing_or_shared_diagnostic_outputs_reject(self):
        for path in (self.object, self.output):
            with self.subTest(path=path.name):
                path.touch()
                with self.assertRaisesRegex(ValueError, "already exists"):
                    self.plan()
                path.unlink()
        with self.assertRaisesRegex(ValueError, "must be distinct"):
            derive_driver_link(self.binary, fortran=self.compiler, source=self.source,
                               object_path=self.output, output=self.output)
        self.object.symlink_to(self.root / "missing target")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.plan()

    def test_receipt_cwd_and_argv_are_explicit(self):
        self.receipt.write_text(json.dumps({"cwd": "relative", "argv": ["scons"]}))
        with self.assertRaisesRegex(ValueError, "explicit cwd and argv"):
            self.plan()


if __name__ == "__main__":
    unittest.main()
