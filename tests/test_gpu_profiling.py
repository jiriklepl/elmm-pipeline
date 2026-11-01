"""Keep device activities distinct from host callbacks and profiler overhead."""

from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from elmm_pipeline.gpu_profiling import parse_nsys_sqlite, profile_command


class GPUProfilingTests(unittest.TestCase):
    def fixture(self, path, *, kernels=True, copies=True):
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")
            connection.execute("INSERT INTO StringIds VALUES (1,'outlined_kernel')")
            connection.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, "end" INTEGER, demangledName INTEGER)')
            connection.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER, "end" INTEGER, bytes INTEGER, copyKind INTEGER)')
            connection.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_OPENACC_LAUNCH (start INTEGER, "end" INTEGER)')
            connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_OPENACC_LAUNCH VALUES (1,9000000000)")
            connection.execute("CREATE TABLE ENUM_CUDA_MEMCPY_OPER (id INTEGER, name TEXT, label TEXT)")
            connection.executemany("INSERT INTO ENUM_CUDA_MEMCPY_OPER VALUES (?,?,?)", [
                (11, "CUDA_MEMCPY_KIND_HTOD", "Host-to-Device"), (12, "CUDA_MEMCPY_KIND_DTOH", "Device-to-Host"),
            ])
            if kernels:
                connection.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,1)", [(1000, 5000), (2000, 8000)])
            if copies:
                connection.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?,?,?,?)", [(0, 500, 80, 11), (8000, 10000, 40, 12)])

    def test_activity_sums_bytes_names_and_span_exclude_host_callbacks(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gpu.sqlite"
            self.fixture(path)
            metrics = parse_nsys_sqlite(path)
        self.assertTrue(metrics["available"])
        self.assertEqual(metrics["kernel_count"], 2)
        self.assertAlmostEqual(metrics["kernel_seconds"], 0.000010)
        self.assertAlmostEqual(metrics["transfer_seconds"], 0.0000025)
        self.assertAlmostEqual(metrics["gpu_span_seconds"], 0.000010)
        self.assertEqual(metrics["h2d_bytes"], 80)
        self.assertEqual(metrics["d2h_bytes"], 40)
        self.assertEqual(metrics["transfer_bytes"], 120)
        self.assertEqual(metrics["transfer_count"], 2)
        self.assertEqual(metrics["kernels"], [{"name": "outlined_kernel", "count": 2, "seconds": 0.000010}])

    def test_empty_trace_is_unavailable_not_zero_cost(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gpu.sqlite"
            self.fixture(path, kernels=False, copies=False)
            metrics = parse_nsys_sqlite(path)
        self.assertFalse(metrics["available"])
        for name in ("kernel_seconds", "transfer_seconds", "h2d_bytes", "d2h_bytes"):
            self.assertIsNone(metrics[name])

    def test_compute_only_trace_preserves_missing_transfer_measurement(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gpu.sqlite"
            self.fixture(path, copies=False)
            metrics = parse_nsys_sqlite(path)
        self.assertTrue(metrics["available"])
        self.assertTrue(metrics["kernel_timing_available"])
        self.assertFalse(metrics["transfer_timing_available"])
        self.assertIsNone(metrics["transfer_seconds"])

    def test_missing_or_invalid_export_is_unavailable(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gpu.sqlite"
            self.assertFalse(parse_nsys_sqlite(path)["available"])
            path.write_text("not a SQLite export")
            metrics = parse_nsys_sqlite(path)
        self.assertFalse(metrics["available"])
        self.assertIn("Could not read", metrics["reason"])

    def test_old_mixed_trace_reports_graph_aggregates_without_inventing_kernels(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gpu.sqlite"
            self.fixture(path)
            with sqlite3.connect(path) as connection:
                connection.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE (start INTEGER, "end" INTEGER)')
                connection.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES (?,?)',
                                       [(-1000, 20000), (0, 5000)])
            metrics = parse_nsys_sqlite(path)
        self.assertTrue(metrics["available"])
        self.assertEqual(metrics["status"], "partial")
        self.assertFalse(metrics["kernel_inventory_complete"])
        self.assertEqual(metrics["kernel_count"], 2)
        self.assertAlmostEqual(metrics["kernel_seconds"], 0.000010)
        self.assertEqual(metrics["graph_aggregate_count"], 2)
        self.assertAlmostEqual(metrics["graph_aggregate_seconds"], 0.000026)
        self.assertAlmostEqual(metrics["gpu_span_seconds"], 0.000021)
        self.assertTrue(metrics["graph_timing_available"])
        self.assertIsNone(metrics["graph_node_kernel_count"])

    def test_graph_only_capture_has_no_individual_kernel_count_or_duration(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "graph.sqlite"
            with sqlite3.connect(path) as connection:
                connection.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE (start INTEGER, "end" INTEGER)')
                connection.execute('INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES (1000, 8000)')
            metrics = parse_nsys_sqlite(path)
        self.assertEqual(metrics["status"], "partial")
        self.assertTrue(metrics["graph_timing_available"])
        self.assertFalse(metrics["kernel_timing_available"])
        self.assertIsNone(metrics["kernel_count"])
        self.assertIsNone(metrics["kernel_seconds"])
        self.assertIsNone(metrics["transfer_seconds"])
        self.assertEqual(metrics["graph_aggregate_count"], 1)
        self.assertAlmostEqual(metrics["gpu_span_seconds"], 0.000007)

    def test_node_trace_excludes_null_and_zero_ids_without_certifying_full_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nodes.sqlite"
            self.fixture(path, copies=False)
            with sqlite3.connect(path) as connection:
                connection.execute('ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD COLUMN graphNodeId INTEGER')
                connection.execute('UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET graphNodeId=0 WHERE start=1000')
                connection.execute('UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET graphNodeId=8 WHERE start=2000')
                connection.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (9000, 10000, 1, NULL)')
            metrics = parse_nsys_sqlite(path)
        self.assertEqual(metrics["status"], "ok")
        self.assertEqual(metrics["kernel_count"], 3)
        self.assertEqual(metrics["graph_node_kernel_count"], 1)
        self.assertAlmostEqual(metrics["kernel_seconds"], 0.000011)
        self.assertIsNone(metrics["graph_aggregate_count"])
        self.assertIsNone(metrics["graph_aggregate_seconds"])
        self.assertIsNone(metrics["kernel_inventory_complete"])

    def test_direct_zero_ids_have_no_graph_nodes_in_older_concurrent_kernel_table(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "direct.sqlite"
            self.fixture(path, copies=False)
            with sqlite3.connect(path) as connection:
                connection.execute('ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD COLUMN graphNodeId INTEGER')
                connection.execute('UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET graphNodeId=0')
                connection.execute('ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL RENAME TO CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL')
            metrics = parse_nsys_sqlite(path)
        self.assertEqual(metrics["kernel_count"], 2)
        self.assertEqual(metrics["graph_node_kernel_count"], 0)
        self.assertAlmostEqual(metrics["kernel_seconds"], 0.000010)
        self.assertIsNone(metrics["kernel_inventory_complete"])

    def test_aggregate_and_node_rows_remain_separate_even_when_they_overlap(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mixed.sqlite"
            self.fixture(path)
            with sqlite3.connect(path) as connection:
                connection.execute('ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD COLUMN graphNodeId INTEGER')
                connection.execute('UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET graphNodeId=3')
                connection.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE (start INTEGER, "end" INTEGER)')
                connection.execute('INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES (0,10000)')
            metrics = parse_nsys_sqlite(path)
        self.assertEqual(metrics["kernel_count"], 2)
        self.assertEqual(metrics["graph_node_kernel_count"], 2)
        self.assertAlmostEqual(metrics["kernel_seconds"], 0.000010)
        self.assertAlmostEqual(metrics["graph_aggregate_seconds"], 0.000010)
        self.assertFalse(metrics["kernel_inventory_complete"])

    def test_empty_graph_table_does_not_invent_aggregate_activity(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "empty_graph.sqlite"
            self.fixture(path, copies=False)
            with sqlite3.connect(path) as connection:
                connection.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE (start INTEGER, "end" INTEGER)')
            metrics = parse_nsys_sqlite(path)
        self.assertEqual(metrics["status"], "ok")
        self.assertIsNone(metrics["graph_aggregate_count"])
        self.assertFalse(metrics["graph_timing_available"])
        self.assertIsNone(metrics["kernel_inventory_complete"])

    def test_missing_profiler_does_not_run_application(self):
        with tempfile.TemporaryDirectory() as temporary, \
             patch("elmm_pipeline.gpu_profiling.shutil.which", return_value=None), \
             patch("elmm_pipeline.gpu_profiling.subprocess.Popen") as process:
            log, wall, metrics = profile_command(["/bin/true"], Path(temporary), {}, Path(temporary), nsys="absent-nsys")
        process.assert_not_called()
        self.assertFalse(metrics["application_launched"])
        self.assertFalse(metrics["available"])
        self.assertEqual(wall, 0)
        self.assertIn("not found", log)

    def test_profile_uses_safe_cpu_flags_and_records_output_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            class Process:
                returncode = 0
                pid = 123

                def __init__(process, argv, **kwargs):
                    prefix = Path(next(argument.split("=", 1)[1] for argument in argv if argument.startswith("--output=")))
                    self.fixture(prefix.with_suffix(".sqlite"))
                    prefix.with_suffix(".nsys-rep").touch()
                    kwargs["stdout"].write("application result\n")
                    self.assertEqual(argv.count("--cuda-graph-trace=node"), 1)
                    self.assertIn("--sample=none", argv)
                    self.assertIn("--cpuctxsw=none", argv)
                    self.assertEqual(argv[-2:], ["/bin/application", "--option"])

                def wait(process, timeout=None):
                    return 0

            with patch("elmm_pipeline.gpu_profiling.shutil.which", return_value="/usr/bin/nsys"), \
                 patch("elmm_pipeline.gpu_profiling.subprocess.Popen", Process):
                log, wall, metrics = profile_command(["/bin/application", "--option"], root, {"FORT_RUNTIME_TRACE": "1"}, root)
            self.assertIn("application result", log)
            self.assertGreater(wall, 0)
            self.assertTrue(metrics["available"])
            self.assertTrue(Path(metrics["report_path"]).is_file())
            self.assertTrue(Path(metrics["sqlite_path"]).is_file())
            saved = json.loads((root / "gpu-profile-metrics.json").read_text())
            self.assertEqual(saved["kernel_count"], 2)


if __name__ == "__main__":
    unittest.main()
