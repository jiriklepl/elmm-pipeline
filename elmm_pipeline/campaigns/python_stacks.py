"""Run authenticated Python source with periodic main-thread stack samples."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
import threading
import time
from pathlib import Path
from types import ModuleType

MAX_FRAMES = 100
MAX_LOG_BYTES = 16 * 2**20


def positive_interval(value):
    try:
        interval = float(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("interval must be a finite positive number") from error
    if not math.isfinite(interval) or interval <= 0:
        raise argparse.ArgumentTypeError("interval must be a finite positive number")
    return interval


def _write(stream, record):
    line = json.dumps(record, sort_keys=True) + "\n"
    if stream.tell() + len(line.encode("utf-8")) > MAX_LOG_BYTES:
        raise RuntimeError("Python stack log byte limit exceeded")
    stream.write(line)
    stream.flush()


def _sample(stream, target, interval, stop, failures):
    try:
        while not stop.wait(interval):
            frame = None
            try:
                frame = sys._current_frames().get(target)
                if frame is None:
                    raise RuntimeError("target Python thread has no current frame")
                frames = []
                while frame is not None and len(frames) < MAX_FRAMES:
                    frames.append({"filename": frame.f_code.co_filename,
                                   "line": frame.f_lineno,
                                   "function": frame.f_code.co_name})
                    frame = frame.f_back
                truncated = frame is not None
            finally:
                # Do not retain the sampled stack (and its locals) between ticks.
                frame = None
            _write(stream, {"schema_version": 1, "event": "python_stack",
                            "time_unix": time.time(), "thread_id": target,
                            "frames": frames, "truncated": truncated})
    except BaseException as error:  # noqa: BLE001 - report worker failures without replaying the script.
        failures.append(error)
        stop.set()
        print(f"Python stack sampler failed: {type(error).__name__}: {error}",
              file=sys.stderr, flush=True)


def run_script(script, script_sha256, output, *, interval=120, arguments=()):
    """Execute the original script once; sampler errors never replay its work."""
    interval = positive_interval(interval)
    if not isinstance(script_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", script_sha256):
        raise ValueError("script SHA-256 must contain 64 lowercase hexadecimal digits")
    script_argument = os.fspath(script)
    original = Path(script).resolve(strict=True)
    if not original.is_file():
        raise ValueError("script must be an existing file")
    source = original.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest != script_sha256:
        raise ValueError("script SHA-256 does not match; no script executed")
    code = compile(source, script_argument, "exec", dont_inherit=True)

    stop, failures = threading.Event(), []
    argv, path = sys.argv, sys.path
    missing = object()
    previous_main = sys.modules.get("__main__", missing)
    module = ModuleType("__main__")
    module.__dict__.update(__file__=script_argument, __cached__=None,
                           __package__="", __spec__=None, __loader__=None)
    failed = False
    # Explicit close preserves the original exception if a buffered flush fails.
    stream = Path(output).open("x", encoding="utf-8")  # noqa: SIM115
    try:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("stack output must be a regular file")
        target = threading.get_ident()
        _write(stream, {"schema_version": 1, "event": "sampler_started",
                        "script": str(original), "script_sha256": digest,
                        "thread_id": target, "interval_seconds": interval,
                        "max_frames": MAX_FRAMES, "max_log_bytes": MAX_LOG_BYTES})
        sampler = threading.Thread(target=_sample, name="python-stack-sampler",
                                   args=(stream, target, interval, stop, failures))
        sys.argv = [script_argument, *arguments]
        sys.path = [str(original.parent), *path[1:]]
        sys.modules["__main__"] = module
        try:
            sampler.start()
            exec(code, module.__dict__)  # noqa: S102 - execute the explicitly supplied, verified script.
        except BaseException as error:
            failed = not isinstance(error, SystemExit) or error.code not in (None, 0)
            raise
        finally:
            try:
                stop.set()
                if sampler.ident is not None:
                    sampler.join()
            except BaseException as error:
                if not failed:
                    raise
                print(f"Python stack sampler shutdown failed: {error}", file=sys.stderr, flush=True)
            finally:
                sys.argv, sys.path = argv, path
                if previous_main is missing:
                    sys.modules.pop("__main__", None)
                else:
                    sys.modules["__main__"] = previous_main
            if failures and not failed:
                raise RuntimeError("Python stack sampler failed") from failures[0]
    finally:
        try:
            stream.close()
        except BaseException as error:
            if not failed:
                raise
            print(f"Python stack log close failed: {error}", file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", required=True, help="original Python source script")
    parser.add_argument("--script-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=positive_interval, default=120)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    run_script(args.script, args.script_sha256, args.output,
               interval=args.interval, arguments=arguments)


if __name__ == "__main__":
    main()
