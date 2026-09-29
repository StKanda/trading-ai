import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from trading_lab.data.providers import DemoProvider
from trading_lab.data.market import completed_bars
from trading_lab.features.engine import FeatureEngine
from trading_lab.ui.charting import build_market_figure, discover_local_symbols, read_runtime_state


class DashboardV101Tests(unittest.TestCase):
    def test_symbol_discovery_excludes_demo_validation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "real").mkdir()
            (root / "real" / "SPY_5m.csv").write_text("x")
            (root / "real" / "SPY_5m.meta.json").write_text("{}")
            (root / "verification" / "data_demo").mkdir(parents=True)
            (root / "verification" / "data_demo" / "QQQ_5m.csv").write_text("x")
            (root / "verification" / "data_demo" / "QQQ_5m.meta.json").write_text("{}")
            self.assertEqual(discover_local_symbols(root), ["SPY"])

    def test_runtime_state_is_symbol_scoped(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "SPY").mkdir()
            (root / "SPY" / "runtime_state.json").write_text(json.dumps({"symbol": "SPY", "decision": "NO TRADE"}))
            (root / "runtime_state.json").write_text(json.dumps({"symbol": "QQQ", "decision": "LONG CANDIDATE"}))
            state = read_runtime_state("SPY", root)
            self.assertEqual(state["symbol"], "SPY")
            self.assertEqual(state["decision"], "NO TRADE")

    def test_decision_chart_contains_candles_volume_and_score(self):
        snap = DemoProvider().fetch("SPY")
        bars = completed_bars(snap, 5, asof="2026-09-25T20:00:00+00:00")
        feats = FeatureEngine().build(bars)
        fig = build_market_figure(bars, feats, "SPY", "RULE_MOMENTUM", 0.55, sessions=1)
        names = {getattr(trace, "name", None) for trace in fig.data}
        self.assertIn("SPY", names)
        self.assertIn("VWAP", names)
        self.assertIn("Score RULE_MOMENTUM", names)
        self.assertTrue(any(getattr(trace, "type", None) == "bar" for trace in fig.data))


if __name__ == "__main__":
    unittest.main()
