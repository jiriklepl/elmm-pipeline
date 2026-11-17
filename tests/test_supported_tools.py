"""The public pipeline accepts only native and local compiler implementations."""

import argparse
from unittest.mock import patch

import pytest

from elmm_pipeline.cli import TOOLS, tool_list
from elmm_pipeline.gpu_toolchain import discover_toolchain


def test_tool_selection_preserves_native_reference():
    assert TOOLS == ("baseline", "local")
    assert tool_list("local,baseline,local") == ["baseline", "local"]
    assert tool_list("baseline") == ["baseline"]
    with pytest.raises(argparse.ArgumentTypeError):
        tool_list("baseline,unsupported")


def test_default_gpu_toolchain_discovers_only_cuda(tmp_path):
    toolkit = tmp_path / "cuda"
    runtime = toolkit / "lib"
    with patch("elmm_pipeline.gpu_toolchain.probe_gpu", return_value={"devices": [], "error": "hidden"}), \
            patch("elmm_pipeline.gpu_toolchain._executable", side_effect=lambda explicit, *args: str(explicit)) as executable, \
            patch("elmm_pipeline.gpu_toolchain._version", return_value="mock compiler"), \
            patch("elmm_pipeline.gpu_toolchain._cuda_runtime_directory", return_value=(runtime, {"method": "mock"})):
        result = discover_toolchain(nvcc=toolkit / "bin/nvcc", cuda_host="/mock/c++", arch="90")
    assert executable.call_count == 2
    assert set(result["versions"]) == {"nvcc", "cuda_host_cxx"}
    assert result["cuda_library_dir"] == str(runtime)
    with patch("elmm_pipeline.gpu_toolchain.probe_gpu") as probe:
        with pytest.raises(ValueError, match="Unsupported GPU languages"):
            discover_toolchain(languages=("unsupported",))
        probe.assert_not_called()
