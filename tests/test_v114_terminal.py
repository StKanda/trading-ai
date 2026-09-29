import unittest
import numpy as np
import pandas as pd

from trading_lab.data.providers import DemoProvider
from trading_lab.data.market import completed_bars
from trading_lab.features.engine import FeatureEngine
from trading_lab.config import AppConfig
from trading_lab.terminal_pipeline import _candidate_setup_mask, _economic_score
from trading_lab.deep_learning.economic import build_economic_labels


class V114EconomicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        snap = DemoProvider().fetch("SPY")
        cls.bars = completed_bars(snap, 5, snap.schedule.MarketClose.max(), True)
        cls.features = FeatureEngine().build(cls.bars).values
        cls.cfg = AppConfig()
        cls.candidates = _candidate_setup_mask(cls.bars, cls.features)

    def test_economic_labels_match_candidate_domain_and_have_direct_pnl(self):
        labels = build_economic_labels(self.bars, self.features, self.cfg.research, self.cfg.risk, self.candidates)
        eligible = labels.RealizedPnL.notna()
        self.assertGreater(int(eligible.sum()), 0)
        self.assertTrue(bool((eligible <= self.candidates).all()))
        self.assertTrue(set(labels.loc[eligible, "ProfitableAfterCosts"].unique()).issubset({0.0, 1.0}))
        self.assertTrue(np.isfinite(labels.loc[eligible, "RealizedReturn"]).all())

    def test_economic_score_is_monotone_and_zero_value_maps_to_half(self):
        frame = pd.DataFrame({"RealizedPnL": [-10.0, 0.0, 10.0]})
        score = _economic_score(frame, pnl_scale=10.0)
        self.assertLess(score.iloc[0], 0.5)
        self.assertAlmostEqual(float(score.iloc[1]), 0.5, places=12)
        self.assertGreater(score.iloc[2], 0.5)
        self.assertTrue(bool(score.is_monotonic_increasing))
