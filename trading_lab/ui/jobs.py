from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

RUNTIME = Path("runtime") / "ui_jobs"


def _safe_job_id(job_id: str) -> str:
    return "".join(c for c in str(job_id) if c.isalnum() or c in "-_")[:80] or "job"


def job_paths(job_id: str):
    job_id = _safe_job_id(job_id)
    root = RUNTIME / job_id
    return root, root / "status.json", root / "job.log"


def read_job(job_id: str) -> dict:
    root, status, log = job_paths(job_id)
    data = {"job_id": _safe_job_id(job_id), "state": "IDLE", "returncode": None}
    if status.exists():
        try:
            data.update(json.loads(status.read_text(encoding="utf-8")))
        except Exception:
            data["state"] = "UNKNOWN"
    if log.exists():
        try:
            data["log_tail"] = "\n".join(log.read_text(errors="replace").splitlines()[-160:])
        except Exception:
            data["log_tail"] = ""
    else:
        data["log_tail"] = ""
    return data


def launch_job(job_id: str, command: Iterable[str], cwd: str | Path = ".") -> dict:
    job_id = _safe_job_id(job_id)
    root, status, log = job_paths(job_id)
    root.mkdir(parents=True, exist_ok=True)
    current = read_job(job_id)
    if current.get("state") == "RUNNING":
        raise RuntimeError(f"Le job {job_id} est déjà en cours.")
    payload = {
        "job_id": job_id,
        "state": "QUEUED",
        "command": list(map(str, command)),
        "cwd": str(Path(cwd).resolve()),
        "created_at": time.time(),
        "pid": None,
        "returncode": None,
    }
    status.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log.write_text("", encoding="utf-8")
    runner = [sys.executable, "-m", "trading_lab.ui.job_runner", "--job-id", job_id, "--cwd", str(Path(cwd).resolve()), "--", *payload["command"]]
    proc = subprocess.Popen(
        runner,
        cwd=str(Path(cwd).resolve()),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    payload["runner_pid"] = proc.pid
    status.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def request_stop(job_id: str) -> bool:
    data = read_job(job_id)
    pid = data.get("pid") or data.get("runner_pid")
    if not pid or data.get("state") not in {"QUEUED", "RUNNING"}:
        return False
    try:
        os.killpg(int(pid), signal.SIGTERM)
    except Exception:
        try:
            os.kill(int(pid), signal.SIGTERM)
        except Exception:
            return False
    return True
