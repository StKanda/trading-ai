import unittest
import numpy as np
import pandas as pd
from trading_lab.data.providers import DemoProvider
from trading_lab.data.market import completed_bars
from trading_lab.features.engine import FeatureEngine, GROUPS
from trading_lab.research.rules import rule_predictions, RULE_IDS
from trading_lab.research.walkforward import _choose_official


class V9RulesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        snap = DemoProvider().fetch('SPY')
        cls.bars = completed_bars(snap, 5, snap.schedule.MarketClose.max(), True)
        cls.features = FeatureEngine().build(cls.bars).values

    def test_rules_are_binary_and_named(self):
        ix = self.bars.index[self.features[GROUPS['D']].notna().all(axis=1)][-200:]
        p = rule_predictions(self.bars, self.features, ix)
        self.assertEqual(list(p.columns), RULE_IDS)
        self.assertTrue(set(np.unique(p.to_numpy())).issubset({0.0, 1.0}))

    def test_rule_prefix_invariance(self):
        ix = self.bars.index[self.features[GROUPS['D']].notna().all(axis=1)]
        cut = ix[-100]
        before = ix[ix <= cut]
        p1 = rule_predictions(self.bars, self.features, before)
        bars2 = self.bars.copy(); feat2 = self.features.copy()
        bars2.loc[bars2.index > cut, ['Open','High','Low','Close']] *= 10
        feat2.loc[feat2.index > cut, :] = 999
        p2 = rule_predictions(bars2, feat2, before)
        pd.testing.assert_frame_equal(p1, p2)

    def test_selection_can_choose_cash(self):
        models = ['RF_A','RF_D','GB_D','LR_A','LR_D'] + RULE_IDS
        table = pd.DataFrame({'model': models, 'trades': [20]*len(models),
                              'total_return': [-.01]*len(models), 'max_drawdown': [-.02]*len(models)})
        stress = pd.DataFrame([{'model': m, 'extra_slippage_bps': x, 'total_return': -.02}
                               for m in models for x in (2.0,5.0)])
        official, _ = _choose_official(table, stress, 30)
        self.assertEqual(official, 'CASH')


if __name__ == '__main__':
    unittest.main()
