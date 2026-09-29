import tempfile
import unittest
from pathlib import Path

from trading_lab.data.providers import DemoProvider
from trading_lab.data.market import completed_bars
from trading_lab.features.engine import FeatureEngine
from trading_lab.ui.charting import build_market_figure, discover_trade_models


class DashboardV103Tests(unittest.TestCase):
    def test_trade_model_discovery(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "trades_RF_A.csv").write_text("entry_time,entry_price\n", encoding="utf-8")
            (root / "trades_RULE_MOMENTUM.csv").write_text("entry_time,entry_price\n", encoding="utf-8")
            (root / "metrics.csv").write_text("model\nRF_A\n", encoding="utf-8")
            self.assertEqual(discover_trade_models(root), ["RF_A", "RULE_MOMENTUM"])

    def test_probable_level_zones_add_shapes(self):
        snap = DemoProvider().fetch("SPY")
        bars = completed_bars(snap, 5, asof="2026-09-25T20:00:00+00:00")
        feats = FeatureEngine().build(bars)
        fig = build_market_figure(bars, feats, "SPY", "RULE_MOMENTUM", 0.55,
                                  sessions=1, show_levels=True, show_probable_zones=True)
        # hrects become layout shapes; at least one support/resistance zone should exist on demo data.
        self.assertGreaterEqual(len(fig.layout.shapes or []), 1)

    def test_scroll_zoom_disabled_in_dashboard(self):
        text = Path("app.py").read_text(encoding="utf-8")
        self.assertIn('"scrollZoom": False', text)
        self.assertIn('▶ Lancer le backtest', text)
        self.assertIn("▶ Lancer l'apprentissage DL", text)


if __name__ == "__main__":
    unittest.main()
