"""Check the transport of unchanged original CPU may-access transactions."""
from .checked_execution_plan import _require, _sha256, _text


def validate_original_host_accesses(procedure, unit, manifest):
    runtime = (manifest.get("runtime") or {}).get("task_acquisition", {})
    for operation in unit["operations"].get("native_operations", ()):
        contract = operation.get("original_host_access")
        complete_call = operation.get("kind") == "complete original native CALL"
        if contract is None:
            _require(not (runtime.get("abi_version") == 2 and complete_call),
                     "original host CALL is missing its acquisition contract")
            continue
        _require(isinstance(contract, dict) and set(contract) == {
            "version", "begin", "plan_kind", "effects", "definitions", "proof"},
            "original host acquisition schema differs")
        _require(type(contract["version"]) is int and contract["version"] == 1
                 and type(contract["plan_kind"]) is int and contract["plan_kind"] == 6
                 and contract["begin"] == "fort_scope_original_host_begin_v1"
                 and contract["effects"] == "may-read and may-write"
                 and contract["definitions"] == "must-overwrites only",
                 "original host acquisition semantics differ")
        _require(runtime.get("abi_version") == 2 and runtime.get("original_host_begin") == contract["begin"]
                 and runtime.get("original_host_plan_kind") == contract["plan_kind"],
                 "original host acquisition runtime ABI differs")
        _require(complete_call and operation.get("preserves_original_source") is True
                 and operation.get("completion", {}).get("available") is True
                 and operation.get("completion", {}).get("internal_cuts_authorized") is False
                 and operation.get("sections", {}).get("available") is False,
                 "original host acquisition requires one complete unchanged CALL")
        proof = contract["proof"]
        false_fields = {"alias_authorized", "capture_authorized", "definitions_authorized", "motion_authorized",
                        "internal_cuts_authorized"}
        _require(isinstance(proof, dict) and set(proof) == false_fields | {
            "schema_version", "identity", "procedure", "reachability_identity", "storage_identity", "resources", "execution"},
            "original host CALL proof schema differs")
        _require(type(proof["schema_version"]) is int and proof["schema_version"] == 1
                 and all(proof[key] is False for key in false_fields)
                 and all(_sha256(proof[key]) for key in ("identity", "reachability_identity", "storage_identity"))
                 and proof["identity"] == operation.get("demand_identity")
                 and proof["procedure"] == procedure["procedure"]
                 and proof["execution"] == "unchanged original CALL after coherent access and physical alias registration",
                 "original host CALL proof authority differs")
        selected = operation.get("selected_original_nodes")
        _require(isinstance(selected, list) and len(selected) == 1 and _text(selected[0])
                 and any(selected[0] == node or selected[0].startswith((node + "/", node + ":"))
                         for node in unit["selected_node_ids"])
                 and operation.get("structured_summary_identity") == procedure["structured_summary_identity"],
                 "original host CALL source selection differs")
        _require(isinstance(proof["resources"], list), "original host CALL resources differ")
        effects, roots = {}, set()
        for resource in proof["resources"]:
            _require(isinstance(resource, dict) and set(resource) == {"resource", "role", "actions"},
                     "original host CALL resource schema differs")
            root, role, actions = resource["resource"], resource["role"], resource["actions"]
            _require(_text(root) and root not in roots and role in {"managed", "constant", "reservation"}
                     and isinstance(actions, list) and all(_text(action) for action in actions)
                     and actions == sorted(set(actions))
                     and set(actions) <= {"read", "write", "undefine", "association", "descriptor"},
                     "original host CALL resource effects differ")
            roots.add(root)
            if role == "managed":
                _require("undefine" not in actions, "original host CALL cannot grant definition-kill authority")
                payload = sorted(set(actions) & {"read", "write"})
                if payload:
                    effects[root] = payload
            elif role == "constant":
                _require(set(actions) <= {"read", "association", "descriptor"}
                         and root in operation.get("source_constants", {}),
                         "original host CALL constant authority differs")
            elif set(actions) - {"association"}:
                _require(root in operation.get("host_metadata", {}),
                         "original host CALL storage reservation is missing")
        _require(effects == operation.get("resource_effects"),
                 "original host CALL managed effects differ from its complete proof")

