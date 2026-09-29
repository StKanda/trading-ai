import unittest
import numpy as np
import pandas as pd

from trading_lab.config import ResearchConfig
from trading_lab.terminal_pipeline import _recenter_threshold


class V1131ThresholdAuditTests(unittest.TestCase):
    def test_selected_raw_threshold_maps_exactly_to_backtester_threshold(self):
        raw = pd.Series([0.20, 0.50, 0.80])
        adjusted = _recenter_threshold(raw, selected_threshold=0.50, output_threshold=0.55)
        self.assertAlmostEqual(float(adjusted.iloc[1]), 0.55, places=12)
        self.assertTrue(bool(adjusted.iloc[0] < 0.55))
        self.assertTrue(bool(adjusted.iloc[2] > 0.55))

    def test_no_trade_selection_forces_zero_scores(self):
        raw = pd.Series([0.1, 0.5, 0.9])
        adjusted = _recenter_threshold(raw, selected_threshold=None, output_threshold=0.55)
        np.testing.assert_array_equal(adjusted.to_numpy(), np.zeros(3))

    def test_robust_validation_defaults(self):
        cfg = ResearchConfig()
        self.assertEqual(cfg.min_validation_trades, 5)
        self.assertEqual(cfg.min_validation_sessions_with_trades, 2)
        self.assertAlmostEqual(cfg.max_validation_top1_abs_share, 0.60)


if __name__ == '__main__':
    unittest.main()
