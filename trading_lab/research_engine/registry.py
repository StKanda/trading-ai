from __future__ import annotations
import hashlib
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
  experiment_id TEXT PRIMARY KEY,
  hypothesis_id TEXT NOT NULL,
  created_at TEXT NOT NULL,
  status TEXT NOT NULL,
  dataset_hash TEXT NOT NULL,
  code_version TEXT NOT NULL,
  model TEXT NOT NULL,
  params_json TEXT NOT NULL,
  metrics_json TEXT,
  robustness_json TEXT,
  decision TEXT,
  reason TEXT,
  runtime_seconds REAL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_experiments_signature
ON experiments(dataset_hash, model, params_json);
CREATE TABLE IF NOT EXISTS promotions (
  promoted_at TEXT NOT NULL,
  old_champion TEXT,
  new_champion TEXT NOT NULL,
  experiment_id TEXT,
  reason TEXT NOT NULL
);
"""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def experiment_signature(dataset_hash, model, params):
    payload = canonical({"dataset_hash": dataset_hash, "model": model, "params": params})
    return hashlib.sha256(payload.encode()).hexdigest()


class ExperimentRegistry:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as con:
            con.executescript(SCHEMA)

    def exists(self, dataset_hash, model, params):
        signature = experiment_signature(dataset_hash, model, params)
        with sqlite3.connect(self.path) as con:
            row = con.execute("SELECT 1 FROM experiments WHERE experiment_id=?", (signature,)).fetchone()
        return row is not None

    def start(self, hypothesis_id, dataset_hash, code_version, model, params):
        experiment_id = experiment_signature(dataset_hash, model, params)
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.path) as con:
            try:
                con.execute("INSERT INTO experiments(experiment_id,hypothesis_id,created_at,status,dataset_hash,code_version,model,params_json) VALUES(?,?,?,?,?,?,?,?)",
                            (experiment_id, hypothesis_id, now, "RUNNING", dataset_hash, code_version, model, canonical(params)))
            except sqlite3.IntegrityError:
                return experiment_id, False
        return experiment_id, True

    def finish(self, experiment_id, status="COMPLETED", metrics=None, robustness=None,
               decision=None, reason=None, runtime_seconds=0.0):
        with sqlite3.connect(self.path) as con:
            con.execute("UPDATE experiments SET status=?, metrics_json=?, robustness_json=?, decision=?, reason=?, runtime_seconds=? WHERE experiment_id=?",
                        (status, canonical(metrics or {}), canonical(robustness or {}), decision,
                         reason, float(runtime_seconds), experiment_id))

    def count(self):
        with sqlite3.connect(self.path) as con:
            return int(con.execute("SELECT COUNT(*) FROM experiments").fetchone()[0])

    def list_recent(self, limit=100):
        with sqlite3.connect(self.path) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute("SELECT * FROM experiments ORDER BY created_at DESC LIMIT ?", (int(limit),)).fetchall()
        return [dict(r) for r in rows]

    def promote(self, old_champion, new_champion, experiment_id, reason):
        with sqlite3.connect(self.path) as con:
            con.execute("INSERT INTO promotions VALUES(?,?,?,?,?)", (
                datetime.now(timezone.utc).isoformat(), old_champion, new_champion, experiment_id, reason))
