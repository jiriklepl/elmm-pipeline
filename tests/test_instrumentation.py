import shutil
import subprocess
import tempfile
from pathlib import Path
import unittest

from elmm_pipeline.instrumentation import execution_coverage, region_call


class InstrumentationTests(unittest.TestCase):
    def test_markers_are_optional_and_names_are_arbitrary(self):
        region = {"id": "f_123", "call": "call unrelated_tile(x)"}
        self.assertEqual(region_call(region, coverage=False), "call unrelated_tile(x)\n")
        self.assertIn("ELMM_PIPELINE_REGION f_123", region_call(region))
        self.assertIn("threadprivate", region_call(region))

    def test_coverage_reports_partial_and_unexercised_regions(self):
        regions = [{"id": "r1"}, {"id": "r2"}]
        result = execution_coverage("ELMM_PIPELINE_REGION r2\nELMM_PIPELINE_REGION r2\n", regions)
        self.assertEqual(result["executed_regions"], ["r2"])
        self.assertEqual(result["unobserved_regions"], ["r1"])
        with self.assertRaises(ValueError):
            execution_coverage("ELMM_PIPELINE_REGION unknown\n", regions)
        self.assertFalse(execution_coverage("", regions, enabled=False)["enabled"])

    @unittest.skipUnless(shutil.which("gfortran"), "requires GNU Fortran")
    def test_thread_local_markers_compile_inside_native_parallel_loop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fragment = region_call({"id": "arbitrary_42", "call": "call advance(i)"})
            source = ("program test\ninteger::i\n!$omp parallel do num_threads(2)\n"
                      "do i=1,10\n" + fragment + "end do\n!$omp end parallel do\ncontains\n"
                      "subroutine advance(index)\ninteger,intent(in)::index\n"
                      "if(index<1) stop 1\nend subroutine\nend program\n")
            (root / "test.f90").write_text(source)
            subprocess.run(["gfortran", "-fopenmp", str(root / "test.f90"), "-o", str(root / "test")], check=True, capture_output=True)
            output = subprocess.check_output([str(root / "test")], text=True)
            self.assertEqual(output.count("ELMM_PIPELINE_REGION arbitrary_42"), 2)
