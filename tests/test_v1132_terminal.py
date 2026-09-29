import unittest
import pandas as pd

from trading_lab.terminal_pipeline import _raw_rank_metrics, _recenter_threshold


class V1132DiagnosticsTests(unittest.TestCase):
    def test_raw_rank_metrics_survive_no_trade_gate(self):
        y = pd.Series([0, 0, 1, 1], index=pd.RangeIndex(4))
        score = pd.Series([0.1, 0.2, 0.8, 0.9], index=pd.RangeIndex(4))
        row = _raw_rank_metrics(y, score, "MLP_12")
        self.assertEqual(row["rows"], 4)
        self.assertAlmostEqual(row["rank_auc"], 1.0)
        self.assertGreater(row["average_precision"], 0.99)

    def test_raw_rank_metrics_calls_score_not_probability(self):
        y = pd.Series([0, 1, 0, 1])
        score = pd.Series([0.3, 0.7, 0.4, 0.6])
        row = _raw_rank_metrics(y, score, "TCN_12")
        self.assertIn("pas une probabilité calibrée", row["note"])

    def test_recenter_semantics_unchanged(self):
        raw = pd.Series([0.4, 0.5, 0.6])
        adjusted = _recenter_threshold(raw, 0.5, 0.55)
        self.assertAlmostEqual(float(adjusted.iloc[1]), 0.55, places=12)


if __name__ == '__main__':
    unittest.main()
