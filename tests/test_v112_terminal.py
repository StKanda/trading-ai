import unittest
import numpy as np
import pandas as pd
from trading_lab.terminal_pipeline import _choose_validation_threshold, _recenter_threshold


class V112SelectionTests(unittest.TestCase):
    def test_threshold_selected_only_on_positive_validation_utility(self):
        idx = pd.date_range('2026-01-01', periods=10, freq='5min', tz='UTC')
        score = pd.Series([.2,.3,.4,.45,.5,.55,.6,.65,.7,.8], index=idx)
        labels = pd.DataFrame({
            'TargetBeforeStop_60':[0,0,0,0,0,1,1,1,1,1],
            'NetReturn_60':[-.01,-.01,-.005,-.003,-.001,.002,.003,.004,.005,.006],
        }, index=idx)
        thr, diag = _choose_validation_threshold(score, labels)
        self.assertIsNotNone(thr)
        self.assertGreater(diag['validation_mean_net60'], 0)
        self.assertGreaterEqual(diag['validation_selected'], 3)

    def test_negative_validation_utility_means_no_trade(self):
        idx = pd.date_range('2026-01-01', periods=10, freq='5min', tz='UTC')
        score = pd.Series(np.linspace(.3,.9,10), index=idx)
        labels = pd.DataFrame({'TargetBeforeStop_60':[0,1]*5, 'NetReturn_60':[-.002]*10}, index=idx)
        thr, diag = _choose_validation_threshold(score, labels)
        self.assertIsNone(thr)
        self.assertEqual(diag['reason'], 'no_positive_validation_utility')

    def test_recenter_maps_selected_threshold_to_execution_threshold(self):
        s = pd.Series([.4,.6,.8])
        out = _recenter_threshold(s, .6, .55)
        self.assertAlmostEqual(float(out.iloc[1]), .55, places=7)
        self.assertLess(out.iloc[0], .55)
        self.assertGreater(out.iloc[2], .55)


if __name__ == '__main__':
    unittest.main()
