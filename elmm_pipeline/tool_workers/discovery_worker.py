"""Parser-environment entry point; the host pipeline needs no fparser import."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from elmm_pipeline.structural import discover_sources


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    manifest = discover_sources(**request)
    destination = Path(request["output_dir"]) / "discovery.json"
    destination.write_text(json.dumps(manifest, indent=2) + "\n")
