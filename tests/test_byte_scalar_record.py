"""The manifest transports byte storage without inventing scalar definitions."""
from copy import deepcopy

import pytest
from elmm_pipeline.source_task_record import validate_source_task
from test_source_task_record import _issued, check, reseal

from compiler.tests.test_byte_scalar_transport import byte_fixture


@pytest.fixture
def issued(tmp_path):
    builder, _nodes = byte_fixture(tmp_path)
    args = _issued(tmp_path, envelopes=True, source=(tmp_path / "source.f90").read_text(), facts=builder.facts)
    check(args)
    record, member, procedure, inputs, resources, owner, manifest = args
    validate_source_task(record, member, procedure, inputs, resources, owner=owner,
                         manifest=manifest, capture_facts=builder.facts)
    return args


def test_byte_storage_source_and_provider_round_trip(issued):
    record = issued[0]
    assert record["version"] == 2
    assert record["execution_binding"]["provider"]["metadata"]["version"] == 9
    assert record["definitions_authorized"] is False


@pytest.mark.parametrize("mutation", ["source_version", "provider_version", "definition", "host_value", "resources",
                                     "whole_fact", "target", "escapes", "integer"])
def test_rehashed_byte_record_cannot_widen_its_contract(issued, mutation):
    args = deepcopy(issued)
    record = args[0]
    provider = record["execution_binding"]["provider"]
    meta = provider["metadata"]
    capture = next(c for c in record["scalar_captures"] if c["resource"] == "argument::scale")
    if mutation == "source_version":
        record["version"] = 1
    elif mutation == "provider_version":
        meta["version"] = 7
    elif mutation == "definition":
        meta["scalar_transport"]["definitions_authorized"] = True
    elif mutation == "host_value":
        meta["scalar_transport"]["host"] = "read real value eagerly"
    elif mutation == "resources":
        meta["scalar_transport"]["resources"] = []
    elif mutation == "whole_fact":
        capture["authority"]["fact"]["initialized"] = "whole"
    elif mutation == "target":
        capture["descriptor"]["attributes"].append("target")
    elif mutation == "escapes":
        capture["authority"]["fact"]["escapes"] = True
    else:
        capture["descriptor"]["type"] = "integer"
    reseal(record)
    with pytest.raises(ValueError, match="byte scalar|source device task"):
        check(args)


def test_byte_scalar_capture_round_trips_complete_checked_scope(tmp_path):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans

    builder, _nodes = byte_fixture(tmp_path)
    _outputs, manifest = builder.run()
    scope, = manifest["scopes"]
    task, = scope["source_device_tasks"]
    assert task["version"] == 2
    assert task["execution_binding"]["provider"]["metadata"]["version"] == 9
    validate_checked_execution_plans(manifest, capture_facts=builder.facts)


def test_byte_source_cannot_enter_shared_authority_reader_for_queued_or_priced_work(issued):
    from elmm_pipeline.source_task_record import _validate_source_authority

    record, member, procedure, inputs, resources, owner, manifest = issued
    with pytest.raises(ValueError, match="source proof version differs"):
        _validate_source_authority(record, member, procedure, inputs, resources, owner=owner, manifest=manifest)
