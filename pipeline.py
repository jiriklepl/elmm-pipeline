#!/usr/bin/env python3
"""Build the GPU-resident ELMM and its native controls. Run from any working directory."""
import argparse
import sys

from elmm_pipeline.resident_build import add_arguments, execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    add_arguments(commands.add_parser(
        "resident-build", help="Build the resident channel GPU path and native controls from source"))
    args = parser.parse_args()
    try:
        execute(args)
    except (ValueError, RuntimeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
