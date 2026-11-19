#!/usr/bin/env python3
"""Build the GPU-resident ELMM and its native controls. Run from any working directory."""
import argparse
import os
import sys


def main():
    # The adapter imports the compiler, so the checkout must be known before the adapter loads.
    early = argparse.ArgumentParser(add_help=False)
    early.add_argument("--compiler")
    selected = early.parse_known_args()[0].compiler
    if selected is not None:
        os.environ["ELMM_COMPILER_ROOT"] = os.path.abspath(selected)
    from elmm_pipeline.resident_build import add_arguments, execute

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
