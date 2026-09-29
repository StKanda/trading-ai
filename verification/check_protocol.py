"""Vérifie les livrables synthétiques et le rejet des changements de protocole."""
from pathlib import Path
import copy
import json
import sys
import tempfile
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trading_lab.config import AppConfig
from trading_lab.data.market import Snapshot
from trading_lab.data.providers import CSVProvider, DemoProvider, fingerprint
from trading_lab.research.walkforward import evaluate_final, code_hashes, canonical_hash, versions
from trading_lab.service import LabService
from trading_lab.web.server import clean


def run():
    folder = ROOT / "verification"
    lock_path = folder / "research_verified/protocol_lock.json"
    lock = json.loads(lock_path.read_text())
    snapshot = CSVProvider(folder / "data_demo").fetch("SPY")
    assert lock["integrity"] == canonical_hash({k: v for k, v in lock.items() if k != "integrity"})
    assert lock["code_hashes"] == code_hashes()
    assert lock["versions"] == versions(), "Relancer recherche et final avec les versions de ce poste."
    assert lock["pretest_fingerprint"] == fingerprint(snapshot, lock["test_start"])
    checked = {}
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        altered = copy.deepcopy(lock)
        altered["config"]["research"]["threshold"] = 0.01
        altered_path = temp / "changed.json"
        altered_path.write_text(json.dumps(altered))
        try:
            evaluate_final(snapshot, altered_path, temp / "should_not_exist_1")
        except ValueError as exc:
            assert "Verrou modifié" in str(exc)
            checked["modified_lock_rejected"] = True
        else:
            raise AssertionError("Le verrou altéré a été accepté.")
        revised = Snapshot(snapshot.symbol, snapshot.prices.copy(), snapshot.schedule.copy(), snapshot.metadata.copy())
        revised.prices.iloc[0, :4] *= 1.001
        try:
            evaluate_final(revised, lock_path, temp / "should_not_exist_2")
        except ValueError as exc:
            assert "pré-test ont changé" in str(exc)
            checked["revised_pretest_rejected"] = True
        else:
            raise AssertionError("La révision des données passées a été acceptée.")
        truncated = Snapshot(snapshot.symbol, snapshot.prices.iloc[:-78].copy(), snapshot.schedule.copy(), snapshot.metadata.copy())
        try:
            evaluate_final(truncated, lock_path, temp / "should_not_exist_3")
        except ValueError as exc:
            assert "Séances finales manquantes" in str(exc)
            checked["missing_final_day_rejected"] = True
        else:
            raise AssertionError("Le test final tronqué a été accepté.")
    for phase in ("research_verified", "final_verified"):
        audit = pd.read_csv(folder / phase / "split_audit.csv")
        assert (audit.gap_bars >= audit.required_gap_bars).all()
        assert (pd.to_datetime(audit.train_label_end_max, utc=True) < pd.to_datetime(audit.prediction_boundary, utc=True)).all()
        metrics = pd.read_csv(folder / phase / "metrics.csv")
        assert len(metrics) == 14
        predictions = pd.read_csv(folder / phase / "predictions.csv")
        assert len(predictions.columns) == 13
        for trades in (folder / phase).glob("trades_*.csv"):
            assert "exit_reason" in pd.read_csv(trades)
        checked[phase] = dict(models=12, benchmarks=2, folds=len(audit), prediction_rows=len(predictions))
    official = pd.read_csv(folder / "final_verified/metrics.csv")
    assert official.loc[official.official, "model"].tolist() == [lock["official_model"]]
    estimate = LabService(DemoProvider(), AppConfig(), "demo").model_estimate("SPY", 5, 270)
    assert not estimate["calibrated"] and estimate["synthetic"]
    assert 0 <= estimate["score"] <= 1
    (folder / "ui_model_example.json").write_text(json.dumps(clean(estimate), indent=2, ensure_ascii=False))
    checked.update(synthetic_only=True, official_model_unchanged=lock["official_model"],
                   code_hashes_valid=True, ui_model_past_only=True, versions=versions(),
                   browser_visual_test="not_available_no_browser_engine", yahoo_live_test="not_run")
    (folder / "verification_summary.json").write_text(json.dumps(checked, indent=2, ensure_ascii=False))
    print(json.dumps(checked, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run()
