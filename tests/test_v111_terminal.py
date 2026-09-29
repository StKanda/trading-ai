import unittest
import numpy as np
import pandas as pd
from trading_lab.config import ResearchConfig, RiskConfig
from trading_lab.deep_learning.utility import build_utility_labels, build_utility_sequence_bundle


class V111UtilityTests(unittest.TestCase):
    def _bars(self, n=50):
        start = pd.Timestamp("2026-09-01 13:30", tz="UTC")
        starts = pd.date_range(start, periods=n, freq="5min")
        ends = starts + pd.Timedelta(minutes=5)
        close = np.linspace(100, 102, n)
        bars = pd.DataFrame({
            "Start": starts, "Open": close, "High": close + 0.6, "Low": close - 0.3,
            "Close": close + 0.1, "Volume": 100000.0,
            "Session": "2026-09-01", "SessionOpen": start,
            "SessionClose": pd.Timestamp("2026-09-01 20:00", tz="UTC"),
        }, index=ends)
        bars.attrs["timeframe"] = 5
        return bars

    def test_utility_targets_have_expected_columns(self):
        bars = self._bars()
        features = pd.DataFrame({"ATR14Pct": 0.002}, index=bars.index)
        labels = build_utility_labels(bars, features, ResearchConfig(), RiskConfig())
        for col in ("TargetBeforeStop_15", "TargetBeforeStop_30", "TargetBeforeStop_60",
                    "MFE_30", "MAE_30", "NetReturn_15", "NetReturn_30", "NetReturn_60"):
            self.assertIn(col, labels.columns)
        valid = labels.TargetBeforeStop_30.dropna()
        self.assertGreater(len(valid), 10)
        self.assertTrue(valid.between(0, 1).all())
        self.assertTrue((labels.MFE_30.dropna() >= labels.MAE_30.dropna()).all())

    def test_stop_priority_same_bar(self):
        bars = self._bars()
        features = pd.DataFrame({"ATR14Pct": 0.001}, index=bars.index)
        # Pour le premier signal exploitable, la bougie d'entrée touche volontairement
        # un low très bas et un high très haut : stop doit gagner la priorité.
        bars.iloc[1, bars.columns.get_loc("Low")] = 90.0
        bars.iloc[1, bars.columns.get_loc("High")] = 110.0
        labels = build_utility_labels(bars, features, ResearchConfig(), RiskConfig())
        self.assertEqual(float(labels.iloc[0].TargetBeforeStop_30), 0.0)

    def test_sequence_scalers_fit_train_only(self):
        bars = self._bars(70)
        features = pd.DataFrame({"ATR14Pct": 0.002, "X": np.arange(70, dtype=float)}, index=bars.index)
        labels = build_utility_labels(bars, features, ResearchConfig(), RiskConfig())
        # Les features du valid sont artificiellement très grandes; le scaler train ne doit pas les absorber.
        features.loc[features.index[50:], "X"] += 10000
        train = np.zeros(70, dtype=bool); train[15:45] = True
        valid = np.zeros(70, dtype=bool); valid[50:58] = True
        bundle = build_utility_sequence_bundle(features, labels, train, valid, context_bars=12,
                                               columns=["ATR14Pct", "X"])
        self.assertLess(bundle.scaler.mean_[1], 1000)
        self.assertEqual(bundle.context_bars, 12)


if __name__ == "__main__":
    unittest.main()
