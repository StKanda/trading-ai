import unittest
import pandas as pd
from trading_lab.data.providers import CSVProvider
from trading_lab.data.market import completed_bars
from trading_lab.features.engine import FeatureEngine
from trading_lab.terminal_pipeline import _candidate_setup_mask


class V113CandidateSetupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        snap = CSVProvider('verification/data_demo').fetch('SPY')
        cls.bars = completed_bars(snap, 5, asof=snap.schedule.MarketClose.max(), require_complete=True)

    def test_candidate_filter_is_selective_and_boolean(self):
        f = FeatureEngine().build(self.bars).values
        mask = _candidate_setup_mask(self.bars, f)
        self.assertEqual(mask.dtype, bool)
        self.assertGreater(int(mask.sum()), 0)
        self.assertLess(int(mask.sum()), len(mask))
        self.assertLess(float(mask.mean()), 0.60)

    def test_candidate_filter_is_prefix_invariant(self):
        cut = min(1800, len(self.bars) - 100)
        full_f = FeatureEngine().build(self.bars).values
        full = _candidate_setup_mask(self.bars, full_f).iloc[:cut]
        prefix_bars = self.bars.iloc[:cut].copy()
        prefix_bars.attrs.update(self.bars.attrs)
        prefix_f = FeatureEngine().build(prefix_bars).values
        prefix = _candidate_setup_mask(prefix_bars, prefix_f)
        pd.testing.assert_series_equal(full, prefix)


if __name__ == '__main__':
    unittest.main()
