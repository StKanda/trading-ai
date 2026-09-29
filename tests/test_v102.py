import unittest

from trading_lab.data.market import completed_bars
from trading_lab.data.providers import DemoProvider
from trading_lab.features.engine import FeatureEngine
from trading_lab.ui.charting import build_market_figure


class DashboardV102Tests(unittest.TestCase):
    def test_30_minute_chart_is_supported(self):
        snap = DemoProvider().fetch("SPY")
        base = completed_bars(snap, 5, asof="2026-09-25T20:00:00+00:00")
        bars = completed_bars(snap, 30, asof="2026-09-25T20:00:00+00:00")
        self.assertEqual(bars.attrs["timeframe"], 30)
        feats = FeatureEngine().build(bars)
        fig = build_market_figure(
            bars, feats, "SPY", "RULE_MOMENTUM", 0.55,
            sessions=1, base_bars_5m=base,
        )
        self.assertIn("30 min", fig.layout.annotations[0].text)
        names = {getattr(trace, "name", None) for trace in fig.data}
        self.assertIn("OR20 High", names)
        self.assertIn("OR20 Low", names)
        self.assertIn("Score descriptif RULE_MOMENTUM", names)

    def test_completed_bars_still_rejects_unsupported_interval(self):
        snap = DemoProvider().fetch("SPY")
        with self.assertRaises(ValueError):
            completed_bars(snap, 7, asof="2026-09-25T20:00:00+00:00")


if __name__ == "__main__":
    unittest.main()
