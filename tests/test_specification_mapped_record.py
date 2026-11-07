"""Precise bound publication must retain all original CALL and live-handle checks."""
from copy import deepcopy

import pytest
from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans
from test_conditional_return_record import seal

from compiler.tests.test_conditional_return import compile_fixture
from compiler.tests.test_specification_entry_mapped import MAPPED_SOURCE


@pytest.fixture(scope="module")
def original(tmp_path_factory):
    _, report, _, _ = compile_fixture(tmp_path_factory.mktemp("mapped-entry-reader"), MAPPED_SOURCE)
    return report


@pytest.mark.parametrize("fault", ["missing", "extra", "resource", "canonical", "callee", "caller", "boolean",
                                  "handle_check", "fallback", "observe", "wrong_handle"])
def test_resealed_metadata_cannot_change_its_projection_or_runtime_checks(original, fault):
    report = deepcopy(original)
    returned = report["scopes"][0]["checked_execution_plan"]["storage_epoch"]["return_contract"]
    route, = [r for member in returned["members"] for r in member["routes"] if "specification_entry" in r]
    phase = route["specification_entry"]
    binding, = phase["bindings"]
    if fault == "missing":
        phase["bindings"] = []
    elif fault == "extra":
        phase["bindings"].append(deepcopy(binding))
    elif fault in {"resource", "canonical", "callee", "caller", "boolean"}:
        key = {"resource": "resource", "canonical": "canonical_resource", "callee": "callee_slot",
               "caller": "caller_slot", "boolean": "caller_slot"}[fault]
        binding[key] = True if fault == "boolean" else 1 if fault in {"callee", "caller"} else "unrelated::metadata"
    else:
        edit, = [e for e in report["source_edits"] if [e["first_line"], e["last_line"]] == phase["source_span"]]
        before, after = {
            "handle_check": ("if (fort_specification_valid == 0)", "if (.false.)"),
            "fallback": ("fort_scope_observe_all_v1(", "fort_scope_wait("),
            "observe": ("fort_scope_observe_v1(", "fort_scope_wait("),
            "wrong_handle": (f'fort_specification_buffers(1) = fort_buffer_{binding["caller_slot"]}',
                             "fort_specification_buffers(1) = 0"),
        }[fault]
        assert before in edit["replacement"]
        edit["replacement"] = edit["replacement"].replace(before, after)
    seal(report)
    with pytest.raises(ValueError, match="Compiler checked execution plan"):
        validate_checked_execution_plans(report)
