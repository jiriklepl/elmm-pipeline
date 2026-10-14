"""Primary-case targets use native medians and match control runs by round ID."""
from __future__ import annotations

from copy import deepcopy

import pytest

from elmm_pipeline.campaigns.identity import digest, read, write
from elmm_pipeline.campaigns.prepare import validate
from elmm_pipeline.campaigns.run import borderline, gates, summaries


def campaign():
    return {"schema_version": 1, "title": "Automatic residency", "threads": 4, "repetitions": 7,
            "repositories": {"compiler": {}, "pipeline": {}}, "baseline_modes": ["native", "sgs-direct"],
            "candidate_mode": "auto-resident", "modes": [
                {"name": "native", "label": "Native"}, {"name": "sgs-direct", "label": "SGS direct"},
                {"name": "auto-resident", "label": "Automatic resident"}],
            "cases": [{"name": "channel-simple", "performance_target": {
                "schema_version": 1, "metric": "seconds", "limit": 9, "comparison": "lt",
                "control_mode": "sgs-direct", "require_control_improvement": False}},
                {"name": "channel", "performance_target": {"schema_version": 1,
                 "metric": "ratio_to_native", "limit": .744, "comparison": "le", "control_mode": "sgs-direct"}}]}


def measured(state, *, native=12, control=9.6, candidate=8.8):
    values = {"native": native, "sgs-direct": control, "auto-resident": candidate}
    runs = [{"mode": mode["name"], "round": round_, "warmup": round_ == 0, "traced": False,
             "wall_seconds": (values[mode["name"]][round_ - 1] if isinstance(values[mode["name"]], list)
                              and round_ > 0 else 100 if round_ == 0 else values[mode["name"]]),
             "peak_host_rss_bytes": 1024}
            for round_ in range(8) for mode in state["modes"]]
    return runs, summaries(runs, state["modes"])


def test_primary_targets_validate_and_have_distinct_boundaries():
    state = campaign()
    validate(state)
    runs, summary = measured(state, candidate=9)
    simple = gates(state, summary, state["cases"][0], runs)
    assert simple["target_seconds"] == 9 and not simple["accepted"]
    runs, summary = measured(state, native=12.5, candidate=9.3)
    channel = gates(state, summary, state["cases"][1], runs)
    assert channel["target_seconds"] == pytest.approx(9.3)
    assert channel["baseline_mode"] == "native"  # The faster SGS control is not the ratio denominator.
    assert channel["accepted"] and channel["comfortable"]["met"]
    assert "thirty_percent_lower" not in channel


def test_acceptance_and_comfort_have_independent_control_requirements():
    state = campaign()
    runs, summary = measured(state, control=8.5)
    simple = gates(state, summary, state["cases"][0], runs)
    assert simple["target_met"] and simple["accepted"]
    assert not simple["faster_than_control"] and not simple["comfortable"]["met"]
    channel = gates(state, summary, state["cases"][1], runs)
    assert channel["target_met"] and not channel["accepted"]


def test_comfort_matches_rounds_despite_rotated_execution_and_excludes_warmup():
    state = campaign()
    runs, summary = measured(state, control=[8.7, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6])
    runs.reverse()  # Positional pairing is invalid after rotated execution.
    result = gates(state, summary, state["cases"][0], runs)
    assert result["accepted"] and result["comfortable"]["all_samples_meet_target"]
    assert not result["comfortable"]["beats_control_each_round"]
    assert not result["comfortable"]["met"]
    for row in runs:
        if row["mode"] == "sgs-direct" and row["round"] == 1:
            row["wall_seconds"] = 9.9
    assert gates(state, summary, state["cases"][0], runs)["comfortable"]["met"]


def test_one_slow_candidate_prevents_comfort_even_when_median_passes():
    state = campaign()
    runs, summary = measured(state, candidate=[8.5, 8.6, 8.7, 8.8, 8.9, 8.9, 9.0])
    result = gates(state, summary, state["cases"][0], runs)
    assert result["accepted"] and not result["comfortable"]["all_samples_meet_target"]
    assert not result["comfortable"]["met"]


def test_incomplete_rounds_cannot_establish_comfort():
    state = campaign()
    runs, summary = measured(state)
    runs = [row for row in runs if row["round"] != 7 or row["mode"] != "sgs-direct"]
    result = gates(state, summary, state["cases"][0], runs)
    assert not result["comfortable"]["paired_rounds_complete"] and not result["comfortable"]["met"]


@pytest.mark.parametrize("update", [{"round": 2}, {"traced": True}, {"wall_seconds": float("nan")}])
def test_invalid_round_evidence_rejects(update):
    state = campaign()
    runs, summary = measured(state)
    next(row for row in runs if row["round"] == 1 and row["mode"] == "auto-resident").update(update)
    with pytest.raises(ValueError, match="qualification requires"):
        gates(state, summary, state["cases"][0], runs)


@pytest.mark.parametrize("change", [
    {"schema_version": True}, {"schema_version": 2}, {"metric": "ratio_to_frozen"}, {"comparison": "lte"},
    {"limit": float("nan")}, {"limit": 0}, {"limit": True}, {"control_mode": "missing"},
    {"control_mode": "native"}, {"control_mode": "auto-resident"}, {"require_control_improvement": "false"},
    {"typo": 1}, {"metric": []}, {"comparison": {}}, {"control_mode": []}])
def test_invalid_target_contract_rejects_before_preparation(change):
    state = campaign()
    state["cases"][0]["performance_target"].update(change)
    with pytest.raises(ValueError, match="performance_target"):
        validate(state)


def test_per_case_target_requires_full_seven_round_protocol():
    state = campaign()
    state["repetitions"] = 3
    with pytest.raises(ValueError, match="seven measured"):
        validate(state)


def test_borderline_uses_configured_seconds_or_native_ratio():
    state = campaign()
    _, summary = measured(state, candidate=8.99)
    assert borderline(state, summary, state["cases"][0])
    _, summary = measured(state, candidate=7)
    assert not borderline(state, summary, state["cases"][0])


def plot_fixture(tmp_path, monkeypatch):
    from elmm_pipeline.campaigns import report
    state = campaign()
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    monkeypatch.setattr(report, "check", lambda root: state)
    write(tmp_path / "campaign.json", state)
    campaign_hash = digest(tmp_path / "campaign.json")
    cases = []
    for case in state["cases"]:
        runs, summary = measured(state)
        cases.append({"case": case["name"], "runs": runs, "summary": summary,
                      "gates": gates(state, summary, case, runs)})
    timing = {"status": "passed", "campaign_sha256": campaign_hash, "cases": cases}
    write(tmp_path / "timings/report.json", timing)
    write(tmp_path / "verification/report.json", {"status": "passed", "campaign_sha256": campaign_hash,
          "cases": [{"case": case["name"], "runs": [{"mode": mode["name"], "activity": {"kernel_launches": 1}}
                    for mode in state["modes"]]} for case in state["cases"]]})
    return report, state, timing


def test_target_charts_render_both_units_metadata_labels_and_samples(tmp_path, monkeypatch):
    report, state, _ = plot_fixture(tmp_path, monkeypatch)
    import matplotlib.pyplot as plt
    axes = []
    original_close = plt.close

    def close(figure=None):
        if hasattr(figure, "axes"):
            axes.extend(figure.axes)
        original_close(figure)

    monkeypatch.setattr(plt, "close", close)
    report.plot(tmp_path)
    assert len(axes) == 4
    assert all([label.get_text() for label in ax.get_xticklabels()] == [m["label"] for m in state["modes"]]
               for ax in axes)
    assert [ax.get_legend().get_texts()[0].get_text() for ax in axes] == [
        "Target: < 9 s", "Target: ≤ 74.4% of native", "Target: < 9 s", "Target: ≤ 74.4% of native"]
    assert all(sum(len(collection.get_offsets()) for collection in ax.collections
                   if collection.__class__.__name__ == "PathCollection") == 21 for ax in axes)
    assert axes[0].get_ylabel() == "Complete wall time (seconds)"
    assert axes[2].get_ylabel() == "Complete wall time / native median"
    for kind in ("absolute", "relative"):
        for suffix in ("png", "svg", "pdf"):
            assert (tmp_path / f"figures/complete-wall-{kind}.{suffix}").is_file()
    receipt = read(tmp_path / "figures/source.json")
    assert receipt["relative_baseline"] == "native"
    assert receipt["performance_targets"]["channel-simple"] == state["cases"][0]["performance_target"]
    with pytest.raises(FileExistsError):
        report.plot(tmp_path)


@pytest.mark.parametrize("mutation", ["campaign", "target", "samples"])
def test_target_plots_reject_stale_or_mismatched_evidence_before_output(tmp_path, monkeypatch, mutation):
    report, _, timing = plot_fixture(tmp_path, monkeypatch)
    timing = deepcopy(timing)
    if mutation == "campaign":
        timing["campaign_sha256"] = "stale"
    elif mutation == "target":
        timing["cases"][0]["gates"]["target_seconds"] = 10
    else:
        timing["cases"][0]["summary"]["auto-resident"]["samples"][0] = 1
    write(tmp_path / "timings/report.json", timing)
    with pytest.raises(ValueError, match="plots require|plot target|plot summary"):
        report.plot(tmp_path)
    assert not (tmp_path / "figures").exists()


@pytest.mark.parametrize("tamper_chart", [False, True])
def test_seal_records_unmet_performance_and_rejects_modified_chart(tmp_path, monkeypatch, tamper_chart):
    report, state, timing = plot_fixture(tmp_path, monkeypatch)
    state.update(previous_snapshot="../previous", repositories={
        "compiler": {"commit": "a" * 40}, "pipeline": {"commit": "b" * 40}})
    write(tmp_path / "campaign.json", state)
    campaign_hash = digest(tmp_path / "campaign.json")
    timing["campaign_sha256"] = campaign_hash
    for entry, case in zip(timing["cases"], state["cases"], strict=True):
        runs, summary = measured(state, candidate=9.2)
        entry.update(runs=runs, summary=summary, gates=gates(state, summary, case, runs))
    write(tmp_path / "timings/report.json", timing)
    verification = read(tmp_path / "verification/report.json")
    verification["campaign_sha256"] = campaign_hash
    write(tmp_path / "verification/report.json", verification)
    write(tmp_path / "halos/report.json", {"status": "passed"})
    for phase in ("verification", "halos", "timings"):
        write(tmp_path / f"guards/{phase}.json", {"status": "finished", "exit_code": 0,
              "headroom_bytes": 3 * 2**30, "minimum_available_bytes": 4 * 2**30,
              "cleanup_complete": True, "remaining_descendants": []})
    report.plot(tmp_path)
    if tamper_chart:
        (tmp_path / "figures/complete-wall-absolute.png").write_bytes(b"modified")
        with pytest.raises(ValueError, match="frozen artifact changed"):
            report.seal(tmp_path, tmp_path / "snapshot")
        assert not (tmp_path / "snapshot").exists()
    else:
        output = report.seal(tmp_path, tmp_path / "snapshot")
        rendered = (output / "report.md").read_text()
        assert "Automatic resident (s)" in rendered and "< 9 s | False | False" in rendered
        assert "≤ 74.4% of native | False | False" in rendered and "30% target" not in rendered
        receipt = read(output / "snapshot.json")
        assert receipt["status"] == "sealed" and not receipt["policy_promoted"]
