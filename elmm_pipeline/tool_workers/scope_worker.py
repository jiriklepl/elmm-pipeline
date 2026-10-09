"""Capture extraction in the explicitly selected parser Python environment."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if __name__ == "__main__":
    from elmm_pipeline.scope_extraction import (
        export_analysis_sources,
        export_numerical_sources,
    )
    from elmm_pipeline.structural import Project

    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    output = Path(request["output_dir"])
    sources = [Path(path) for path in request["source_files"]]
    project = Project(Path(request["project_root"]), request["defines"])
    numerical = export_numerical_sources(project, request["entry"], output / "numerical", source_files=sources)
    analysis = export_analysis_sources(project, output / "analysis", source_files=sources)
    report = {"analysis_sources": str(analysis), "numerical_sources": numerical["package"],
              "numerical_extraction": numerical}
    (output / "scope-extraction.json").write_text(json.dumps(report, indent=2) + "\n")
