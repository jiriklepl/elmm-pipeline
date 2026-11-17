import unittest

from elmm_pipeline.instrumentation import execution_coverage


class InstrumentationTests(unittest.TestCase):

    def test_coverage_reports_partial_and_unexercised_regions(self):
        regions = [{"id": "r1"}, {"id": "r2"}]
        result = execution_coverage("ELMM_PIPELINE_REGION r2\nELMM_PIPELINE_REGION r2\n", regions)
        self.assertEqual(result["executed_regions"], ["r2"])
        self.assertEqual(result["unobserved_regions"], ["r1"])
        with self.assertRaises(ValueError):
            execution_coverage("ELMM_PIPELINE_REGION unknown\n", regions)
        self.assertFalse(execution_coverage("", regions, enabled=False)["enabled"])
