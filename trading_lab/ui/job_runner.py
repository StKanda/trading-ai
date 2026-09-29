from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from .jobs import job_paths


def _write(path: Path, payload: dict):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--job-id", required=True)
    p.add_argument("--cwd", required=True)
    p.add_argument("rest", nargs=argparse.REMAINDER)
    args = p.parse_args(argv)
    command = list(args.rest)
    if command and command[0] == "--":
        command = command[1:]
    root, status_path, log_path = job_paths(args.job_id)
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "job_id": args.job_id,
        "state": "RUNNING",
        "command": command,
        "cwd": args.cwd,
        "started_at": time.time(),
        "pid": os.getpid(),
        "returncode": None,
    }
    _write(status_path, payload)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write("$ " + " ".join(command) + "\n\n")
        try:
            proc = subprocess.Popen(command, cwd=args.cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1)
            payload["pid"] = proc.pid
            _write(status_path, payload)
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
            rc = proc.wait()
            payload["returncode"] = rc
            payload["finished_at"] = time.time()
            payload["state"] = "COMPLETED" if rc == 0 else "FAILED"
        except Exception as exc:
            log.write(f"\nJOB ERROR: {exc}\n")
            payload["returncode"] = -1
            payload["finished_at"] = time.time()
            payload["state"] = "FAILED"
            payload["error"] = str(exc)
        _write(status_path, payload)


if __name__ == "__main__":
    main()
