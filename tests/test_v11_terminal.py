import tempfile
import pandas as pd
import unittest
from pathlib import Path

from trading_lab.data.market import Snapshot

from trading_lab.terminal_pipeline import _parse_dl_model, _closed_session_snapshot, doctor


class TerminalV11Tests(unittest.TestCase):
    def test_parse_dl_model(self):
        self.assertEqual(_parse_dl_model("MLP_12"), ("mlp", 12))
        self.assertEqual(_parse_dl_model("tcn_24"), ("tcn", 24))
        with self.assertRaises(ValueError):
            _parse_dl_model("LSTM_24")

    def test_doctor(self):
        report = doctor()
        self.assertIn("python", report)
        self.assertIn("torch", report)
        self.assertIn("cuda", report)

    def test_partial_trailing_session_is_excluded(self):
        schedule = pd.DataFrame({
            "MarketOpen": [pd.Timestamp("2026-09-25 13:30", tz="UTC"), pd.Timestamp("2026-09-28 13:30", tz="UTC")],
            "MarketClose": [pd.Timestamp("2026-09-25 20:00", tz="UTC"), pd.Timestamp("2026-09-28 20:00", tz="UTC")],
        }, index=pd.Index(["2026-09-25", "2026-09-28"], name="Session"))
        idx1 = pd.date_range(schedule.iloc[0].MarketOpen, schedule.iloc[0].MarketClose, freq="5min", inclusive="left")
        idx2 = pd.date_range(schedule.iloc[1].MarketOpen, pd.Timestamp("2026-09-28 16:00", tz="UTC"), freq="5min", inclusive="left")
        idx = idx1.append(idx2)
        prices = pd.DataFrame({"Open":100.0,"High":101.0,"Low":99.0,"Close":100.5,"Volume":1000.0}, index=idx)
        snap = Snapshot("NVDA", prices, schedule, {"bar_minutes":5,"fetched_at":"2026-09-28T16:08:00+00:00"})
        closed = _closed_session_snapshot(snap)
        self.assertEqual(list(closed.schedule.index), ["2026-09-25"])
        self.assertEqual(closed.prices.index[-1], pd.Timestamp("2026-09-25 19:55", tz="UTC"))

    def test_cli_wrapper_exists(self):
        self.assertTrue(Path("trading-ai").exists())
        self.assertTrue(Path("V11_TERMINAL_README.md").exists())


if __name__ == "__main__":
    unittest.main()
