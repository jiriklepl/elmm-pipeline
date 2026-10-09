"""Guard one benchmark command and all attributable Linux descendants."""
from __future__ import annotations
import argparse
import ctypes
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def available():
    return int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                    if line.startswith("MemAvailable:"))) * 1024


def processes():
    result = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rpartition(")")[2].split()
            result[int(path.parent.name)] = (int(fields[1]), int(fields[19]), fields[0])
        except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError, ValueError):
            pass
    return result


class Descendants:
    def __init__(self, child):
        self.child = child
        self.identities = {}
        self.signals = []
        self.reaped = []
        self.refresh()

    def refresh(self):
        table = processes()
        pending = {pid for pid, (parent, start, _) in table.items()
                   if parent == os.getpid() or (pid == self.child.pid and self.child.poll() is None)}
        pending.update(pid for pid, start in self.identities.items()
                       if pid in table and table[pid][1] == start)
        while pending:
            pid = pending.pop()
            if pid not in table or pid in self.identities and self.identities[pid] != table[pid][1]:
                continue
            fresh = pid not in self.identities
            self.identities[pid] = table[pid][1]
            if fresh:
                pending.update(target for target, (parent, _, _) in table.items() if parent == pid)
            else:
                for target, (parent, start, _) in table.items():
                    if parent == pid and self.identities.get(target) != start:
                        pending.add(target)
        return {pid: start for pid, start in self.identities.items()
                if pid in table and table[pid][1] == start and table[pid][2] != "Z"}

    def send(self, pid, start, signum):
        table = processes()
        if pid not in table or table[pid][1] != start or table[pid][2] == "Z":
            return
        fd = None
        try:
            # A pidfd prevents accidental signaling after PID reuse. Validate
            # start time again after opening it to bind the intended process.
            fd = os.pidfd_open(pid)
            current = processes().get(pid)
            if current is None or current[1] != start:
                return
            signal.pidfd_send_signal(fd, signum)
            self.signals.append({"pid": pid, "start_time_ticks": start, "signal": signum})
        except ProcessLookupError:
            pass
        finally:
            if fd is not None:
                os.close(fd)

    def reap(self):
        self.child.poll()
        for pid in self.identities:
            if pid == self.child.pid:
                continue
            try:
                result, status = os.waitpid(pid, os.WNOHANG)
                if result:
                    self.reaped.append({"pid": pid, "status": status})
            except ChildProcessError:
                pass

    def stop(self):
        signaled = set()
        for signum, timeout in ((signal.SIGTERM, 5), (signal.SIGKILL, 5)):
            deadline = time.monotonic() + timeout
            while True:
                live = self.refresh()
                for pid, start in live.items():
                    key = (pid, start, signum)
                    if key not in signaled:
                        self.send(pid, start, signal.SIGCONT)
                        self.send(pid, start, signum)
                        signaled.add(key)
                self.reap()
                if not self.refresh():
                    return []
                if time.monotonic() >= deadline:
                    break
                time.sleep(.05)
        return [{"pid": pid, "start_time_ticks": start} for pid, start in self.refresh().items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        parser.error("missing command")
    if args.record.exists() or args.log.exists():
        raise RuntimeError("existing attempt evidence must remain immutable")
    # Serialize maintained heavy jobs even when independent campaigns are
    # launched from different checkouts. The descriptor lives through cleanup.
    lock = open(f"/tmp/elmm-campaign-{os.getuid()}.lock", "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another guarded campaign job is active; no process started")
    threshold = 3 * 2**30
    minimum = available()
    if minimum < threshold:
        raise SystemExit("less than 3 GiB available; no process started")
    # Orphaned grandchildren, including setsid children, stay attributable.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "cannot enable Linux child subreaper")
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise RuntimeError("Linux pidfd support is required before starting a guarded job")
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    record = {"argv": command, "cwd": os.getcwd(), "headroom_bytes": threshold,
              "started_unix": time.time(), "profiling_policy": "selected by caller; no overrides",
              "termination_policy": {"schema_version": 2, "linux_child_subreaper": True,
                  "identity": "PID and /proc start time, signaled through validated pidfd",
                  "new_sessions_tracked": True, "term_grace_seconds": 5, "kill_grace_seconds": 5}}
    started = time.monotonic()
    interrupted = 0
    def request_stop(signum, _frame):
        nonlocal interrupted
        interrupted = signum
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, request_stop)
    stopped = False
    with args.log.open("w") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        descendants = Descendants(child)
        record.update(child_pid=child.pid, process_group=child.pid, status="running")
        args.record.write_text(json.dumps(record, indent=2) + "\n")
        try:
            while child.poll() is None:
                descendants.refresh()
                minimum = min(minimum, available())
                if minimum < threshold or interrupted:
                    stopped = minimum < threshold
                    break
                time.sleep(1)
        finally:
            # Cleanup also runs if the command returns with detached children.
            remaining = descendants.stop()
            descendants.reap()
            record.update(exit_code=child.poll(), minimum_available_bytes=minimum,
                          elapsed_seconds=time.monotonic() - started, stopped_for_memory=stopped,
                          interrupted_signal=interrupted, status="finished",
                          observed_descendants=[{"pid": pid, "start_time_ticks": start}
                                                for pid, start in sorted(descendants.identities.items())],
                          descendant_signals=descendants.signals, adopted_reaped=descendants.reaped,
                          cleanup_complete=not remaining, remaining_descendants=remaining)
            args.record.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    raise SystemExit(child.returncode or int(stopped) or (128 + interrupted if interrupted else 0) or int(bool(remaining)))


if __name__ == "__main__":
    main()
