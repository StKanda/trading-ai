import json
import tempfile
import unittest
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from trading_lab.config import RiskConfig
from trading_lab.risk import detailed_position_size
from trading_lab.data.freshness import FreshnessPolicy, evaluate_freshness
from trading_lab.data.market import Snapshot
from trading_lab.research_engine.registry import ExperimentRegistry
from trading_lab.research_engine.engine import ResearchBudget, run_bounded_cycle
from trading_lab.validation_engine.robustness import pnl_concentration, neighborhood_stability, robustness_gate
from trading_lab.deep_learning.dataset import build_sequence_bundle


class RiskSizingTests(unittest.TestCase):
    def test_detailed_sizing_respects_risk_and_leverage(self):
        cfg=RiskConfig(initial_capital=100000,max_leverage=1,risk_per_trade=.0025,daily_loss_limit=.01,
                       fee_bps=0,slippage_bps=0,borrowing_bps_per_day=0,stop_atr=1,min_stop_pct=.001,
                       reward_risk=2,max_volume_fraction=.05)
        r=detailed_position_size(100000,500,499,100000,cfg,risk_buffer_pct=0)
        self.assertLessEqual(r.planned_risk,250+1e-9)
        self.assertLessEqual(r.notional,100000+1e-9)
        self.assertEqual(r.quantity,200)  # leverage cap, risk would allow 250

    def test_daily_limit_can_block(self):
        cfg=RiskConfig(initial_capital=10000,risk_per_trade=.0025,daily_loss_limit=.01)
        r=detailed_position_size(9900,500,499,100000,cfg,opening_equity=10000,realized_daily_pnl=-100)
        self.assertEqual(r.quantity,0)
        self.assertEqual(r.status,"NO_TRADE")


class FreshnessTests(unittest.TestCase):
    def snapshot(self,last_start="2026-09-25T19:55:00Z"):
        idx=pd.date_range("2026-09-25T13:30:00Z",last_start,freq="5min")
        p=pd.DataFrame({"Open":100,"High":101,"Low":99,"Close":100,"Volume":1000},index=idx)
        sched=pd.DataFrame({"MarketOpen":["2026-09-25T13:30:00Z"],"MarketClose":["2026-09-25T20:00:00Z"]},index=["2026-09-25"])
        return Snapshot("SPY",p,sched,{"bar_minutes":5,"fetched_at":"2026-09-25T20:01:00Z"})

    def test_fresh_when_last_complete_present(self):
        r=evaluate_freshness(self.snapshot(),"2026-09-25T20:00:30Z",5,FreshnessPolicy())
        self.assertEqual(r.status,"FRESH")
        self.assertTrue(r.usable_for_signal)

    def test_stale_when_many_bars_behind(self):
        r=evaluate_freshness(self.snapshot("2026-09-25T19:25:00Z"),"2026-09-25T20:00:00Z",5,
                             FreshnessPolicy(5,10,20))
        self.assertEqual(r.status,"STALE")
        self.assertFalse(r.usable_for_signal)


class RegistryTests(unittest.TestCase):
    def test_duplicate_is_detected(self):
        with tempfile.TemporaryDirectory() as td:
            reg=ExperimentRegistry(Path(td)/"e.db")
            a,created=reg.start("H1","DATA","10","RF",{"x":1})
            self.assertTrue(created)
            b,created2=reg.start("H2","DATA","10","RF",{"x":1})
            self.assertEqual(a,b); self.assertFalse(created2)

    def test_cycle_stops_after_no_improvement(self):
        with tempfile.TemporaryDirectory() as td:
            reg=ExperimentRegistry(Path(td)/"e.db")
            hyps=[{"hypothesis_id":f"H{i}","model":"RF","params":{"x":i}} for i in range(10)]
            report=run_bounded_cycle(reg,"DATA","10",hyps,lambda h: ({"trades":1},{"passes":False}),
                                     ResearchBudget(10,100,1,3,5))
            self.assertEqual(report["experiments_this_cycle"],3)
            self.assertEqual(report["stop_reason"],"no_robust_improvement_limit")


class RobustnessTests(unittest.TestCase):
    def test_concentration_flags_single_winner(self):
        t=pd.DataFrame({"pnl":[100,-10,-10,-10]})
        self.assertTrue(pnl_concentration(t)["highly_concentrated"])

    def test_plateau_vs_peak(self):
        self.assertTrue(neighborhood_stability({.56:4.1,.57:4.4,.58:4.5,.59:4.2},.58)["stable"])
        self.assertFalse(neighborhood_stability({.56:-1,.57:-2,.58:14,.59:-3},.58)["stable"])

    def test_gate_requires_trades_and_expectancy(self):
        g=robustness_gate({"trades":5,"expectancy":-1,"max_drawdown":-.05},min_trades=20)
        self.assertFalse(g["passes"])


class DLDataTests(unittest.TestCase):
    def test_scaler_fits_train_only(self):
        n=80; idx=pd.date_range("2026-01-01",periods=n,freq="5min",tz="UTC")
        x=pd.DataFrame({"a":np.r_[np.arange(60),np.full(20,10000)],"b":np.arange(n)},index=idx)
        labels=pd.DataFrame({"Target":np.arange(n)%2},index=idx)
        train=np.zeros(n,dtype=bool); train[:55]=True
        valid=np.zeros(n,dtype=bool); valid[60:]=True
        bundle=build_sequence_bundle(x,labels,train,valid,context_bars=5)
        self.assertLess(bundle.scaler.mean_[0],100)  # validation extreme not used by scaler
        self.assertGreater(len(bundle.x_valid),5)


class OptionalDependencyTests(unittest.TestCase):
    def test_core_imports_do_not_require_torch(self):
        code = r'''
import builtins
orig = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        err = ModuleNotFoundError("No module named torch")
        err.name = "torch"
        raise err
    return orig(name, *args, **kwargs)
builtins.__import__ = blocked
from trading_lab.deep_learning.dataset import build_sequence_bundle
import trading_lab.v10_commands
import trading_lab.cli
print("OK")
'''
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("OK", proc.stdout)


if __name__ == "__main__":
    unittest.main()
