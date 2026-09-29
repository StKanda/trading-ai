"""Tests de risques réels de fuite/exécution ; unittest, aucune dépendance pytest."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.request
import urllib.error
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from trading_lab.config import AppConfig, RiskConfig, ResearchConfig
from trading_lab.data.market import Snapshot, completed_bars, validate_prices, TZ
from trading_lab.data.providers import DemoProvider, CSVProvider, write_snapshot, fingerprint
from trading_lab.features.engine import FeatureEngine, build_labels, GROUPS
from trading_lab.features.legacy_v7 import accept_event
from trading_lab.risk import DailyRiskGuard
from trading_lab.research.backtest import simulate, reference_curve
from trading_lab.research.models import eligible_train, fit_predict
from trading_lab.research.metrics import performance
from trading_lab.research.walkforward import slice_snapshot, canonical_hash
from trading_lab.service import LabService
from trading_lab.web.server import create_server


def fixture(days=("2026-09-21",), close_time="16:00"):
    sessions = pd.DataFrame({
        "MarketOpen": [pd.Timestamp(f"{d} 09:30", tz=TZ).tz_convert("UTC") for d in days],
        "MarketClose": [pd.Timestamp(f"{d} {close_time}", tz=TZ).tz_convert("UTC") for d in days]}, index=days)
    ix = pd.DatetimeIndex([t for row in sessions.itertuples() for t in
                           pd.date_range(row.MarketOpen, row.MarketClose, freq="5min", inclusive="left")])
    prices = pd.DataFrame(dict(Open=100., High=100.1, Low=99.9, Close=100., Volume=100000.), index=ix)
    meta = dict(symbol="SPY", source="test", synthetic=True, bar_minutes=5, timestamp_convention="bar_start", currency="USD", adjustment="raw",
                fetched_at=sessions.MarketClose.iloc[-1].isoformat())
    return Snapshot("SPY", prices, sessions, meta)


class DataTests(unittest.TestCase):
    def test_10min_alignment_and_incomplete_current(self):
        snap = fixture()
        op = snap.schedule.MarketOpen.iloc[0]
        bars = completed_bars(snap, 10, op + pd.Timedelta(minutes=19))
        self.assertEqual(len(bars), 1)
        self.assertEqual(bars.index[0], op + pd.Timedelta(minutes=10))
        self.assertEqual(bars.Volume.iloc[0], 200000)

    def test_15min_never_crosses_session(self):
        snap = fixture(("2026-09-21", "2026-09-22"))
        bars = completed_bars(snap, 15, snap.schedule.MarketClose.iloc[-1], True)
        self.assertEqual(bars.groupby("Session").size().tolist(), [26, 26])
        self.assertTrue((bars.Start >= bars.SessionOpen).all())
        self.assertTrue((bars.index <= bars.SessionClose).all())

    def test_early_close_and_dst(self):
        snap = fixture(("2026-03-06", "2026-03-09"), close_time="13:00")
        bars = completed_bars(snap, 10, snap.schedule.MarketClose.iloc[-1], True)
        self.assertEqual(bars.groupby("Session").size().tolist(), [21, 21])
        self.assertEqual(snap.schedule.MarketOpen.iloc[0].hour, 14)
        self.assertEqual(snap.schedule.MarketOpen.iloc[1].hour, 13)

    def test_missing_bar_not_forward_filled(self):
        snap = fixture()
        missing = snap.prices.index[1]
        snap.prices = snap.prices.drop(missing)
        bars = completed_bars(snap, 10, snap.schedule.MarketClose.iloc[-1])
        self.assertEqual(len(bars), 38)
        self.assertEqual(bars.attrs["missing_bars"], 1)
        with self.assertRaisesRegex(ValueError, "incomplète"):
            completed_bars(snap, 10, snap.schedule.MarketClose.iloc[-1], True)

    def test_missing_entire_day_rejected(self):
        snap = fixture(("2026-09-21", "2026-09-22", "2026-09-23"))
        snap.prices = snap.prices.loc[snap.prices.index.tz_convert(TZ).strftime("%Y-%m-%d") != "2026-09-22"]
        with self.assertRaisesRegex(ValueError, "incomplète"):
            completed_bars(snap, 5, snap.schedule.MarketClose.iloc[-1], True)

    def test_invalid_prices_and_naive_time_rejected(self):
        snap = fixture()
        invalid = snap.prices.copy(); invalid.iloc[0, invalid.columns.get_loc("High")] = 50
        with self.assertRaises(ValueError): validate_prices(invalid)
        naive = snap.prices.copy(); naive.index = naive.index.tz_localize(None)
        with self.assertRaises(ValueError): validate_prices(naive)

    def test_snapshot_roundtrip(self):
        snap = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            write_snapshot(snap, tmp)
            loaded = CSVProvider(tmp).fetch("SPY")
            self.assertEqual(fingerprint(snap), fingerprint(loaded))

    def test_capture_time_does_not_complete_last_partial_bar(self):
        snap = fixture()
        snap.metadata["fetched_at"] = (snap.schedule.MarketClose.iloc[0] - pd.Timedelta(minutes=3)).isoformat()
        bars = completed_bars(snap, 5, snap.schedule.MarketClose.iloc[0])
        self.assertEqual(len(bars), 77)
        with self.assertRaisesRegex(ValueError, "incomplète"):
            completed_bars(snap, 5, snap.schedule.MarketClose.iloc[0], True)


class CausalityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snap = slice_snapshot(DemoProvider().fetch("SPY"), "2026-08-22")
        cls.bars = completed_bars(cls.snap, 5, cls.snap.schedule.MarketClose.iloc[-1], True)
        cls.features = FeatureEngine().build(cls.bars)
        cls.cfg = replace(ResearchConfig(), rf_trees=10, gb_iterations=10, min_train_rows=100)
        cls.labels = build_labels(cls.bars, cls.cfg, RiskConfig())

    def test_features_prefix_invariance(self):
        cut = len(self.bars) - 27
        before = FeatureEngine().build(self.bars.iloc[:cut])
        assert_frame_equal(before.values, self.features.values.iloc[:cut])
        confirmed = self.features.pivots.loc[self.features.pivots.confirm_date <= self.bars.index[cut-1]].reset_index(drop=True)
        assert_frame_equal(before.pivots.reset_index(drop=True), confirmed)
        events = self.features.events.loc[self.features.events.breakout_date <= self.bars.index[cut-1]].reset_index(drop=True)
        assert_frame_equal(before.events.reset_index(drop=True), events)

    def test_future_shock_changes_no_past_feature(self):
        cut = len(self.bars) - 30
        shocked = self.bars.copy()
        shocked.loc[shocked.index[cut:], ["Open", "High", "Low", "Close"]] *= 1.8
        shocked.loc[shocked.index[cut:], "Volume"] *= 10
        alt = FeatureEngine().build(shocked)
        assert_frame_equal(alt.values.iloc[:cut], self.features.values.iloc[:cut])

    def test_dedup_first_signal_is_never_replaced(self):
        accepted = {}
        self.assertTrue(accept_event("W", 10, accepted, 6))
        self.assertFalse(accept_event("W", 15, accepted, 6))
        self.assertEqual(accepted["W"], 10)
        self.assertTrue(accept_event("W", 17, accepted, 6))

    def test_pivots_only_known_two_bars_later(self):
        p = self.features.pivots
        self.assertGreater(len(p), 0)
        self.assertTrue(((p.confirm_date - p.pivot_date) == pd.Timedelta(minutes=10)).all())

    def test_labels_no_overnight_and_unknown_not_zero(self):
        for day, bars in self.bars.groupby("Session"):
            self.assertTrue(self.labels.loc[bars.index[-7:], "Target"].isna().all())
        known = self.labels.dropna()
        self.assertTrue((known.LabelEnd.dt.tz_convert(TZ).dt.strftime("%Y-%m-%d").to_numpy()
                         == self.bars.loc[known.index, "Session"].to_numpy()).all())

    def test_label_uses_next_open_and_30min_exit(self):
        snap = fixture()
        bars = completed_bars(snap, 5, snap.schedule.MarketClose.iloc[-1], True)
        bars.iloc[1, bars.columns.get_loc("Open")] = 110
        bars.iloc[7, bars.columns.get_loc("Open")] = 121
        risk = replace(RiskConfig(), fee_bps=0, slippage_bps=0)
        labels = build_labels(bars, self.cfg, risk)
        self.assertAlmostEqual(labels.FutureNetReturn.iloc[0], .1)
        self.assertEqual(labels.LabelEnd.iloc[0], bars.Start.iloc[7])

    def test_volume_reference_excludes_same_day_future(self):
        b = self.bars
        i = len(b)-20
        slot = (b.Start.iloc[i] - b.SessionOpen.iloc[i]).total_seconds()
        previous = b.loc[(b.Session < b.Session.iloc[i]) &
                         ((b.Start - b.SessionOpen).dt.total_seconds() == slot)].Volume.tail(20).mean()
        self.assertAlmostEqual(self.features.values.RelativeVolumeSlot.iloc[i], b.Volume.iloc[i]/previous)

    def test_purge_and_predictions_future_invariant(self):
        boundary = self.bars.SessionOpen.iloc[-1]
        mask, audit = eligible_train(self.bars, self.features.values, self.labels, boundary, 5, self.cfg)
        self.assertLess(self.labels.loc[mask, "LabelEnd"].max(), boundary)
        self.assertGreaterEqual(audit["gap_bars"], 8)
        first = self.bars.index[self.bars.Session == self.bars.Session.iloc[-1]][:5]
        p1 = fit_predict("RF_A", self.features.values, self.labels, mask, first, self.cfg)
        f2 = self.features.values.copy(); f2.loc[f2.index > first[-1], GROUPS["A"]] = 999
        p2 = fit_predict("RF_A", f2, self.labels, mask, first, self.cfg)
        assert_series_equal(p1, p2)

    def test_pretest_slice_excludes_all_final_prices(self):
        cutoff = "2026-08-17"
        past = slice_snapshot(self.snap, cutoff)
        self.assertLess(past.prices.index[-1].tz_convert(TZ).strftime("%Y-%m-%d"), cutoff)
        changed = Snapshot(self.snap.symbol, self.snap.prices.copy(), self.snap.schedule, self.snap.metadata)
        mask = changed.prices.index.tz_convert(TZ).strftime("%Y-%m-%d") >= cutoff
        changed.prices.loc[mask, ["Open", "High", "Low", "Close"]] *= 2
        self.assertEqual(fingerprint(self.snap, cutoff), fingerprint(changed, cutoff))


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        snap = fixture()
        self.bars = completed_bars(snap, 5, snap.schedule.MarketClose.iloc[-1], True)
        self.risk = replace(RiskConfig(), fee_bps=0, slippage_bps=0, borrowing_bps_per_day=0,
                            stop_atr=2, min_stop_pct=.01, risk_per_trade=.005, daily_loss_limit=.05)
        self.cfg = ResearchConfig()
        self.features = pd.DataFrame({"ATR14Pct": .005}, index=self.bars.index)
        self.p = pd.Series(np.nan, index=self.bars.index); self.p.iloc[0] = 1

    def test_next_open_not_signal_close(self):
        self.bars.iloc[1, self.bars.columns.get_loc("Open")] = 110
        self.bars.iloc[1, self.bars.columns.get_loc("High")] = 111
        sim = simulate(self.bars, self.p, self.features, self.risk, self.cfg)
        self.assertEqual(sim.trades.entry_price.iloc[0], 110)
        self.assertEqual(sim.trades.entry_time.iloc[0], self.bars.Start.iloc[1])
        self.assertEqual(sim.trades.signal_time.iloc[0], self.bars.index[0])

    def test_both_barriers_stop_first(self):
        self.bars.iloc[1, self.bars.columns.get_loc("High")] = 103
        self.bars.iloc[1, self.bars.columns.get_loc("Low")] = 98
        sim = simulate(self.bars, self.p, self.features, self.risk, self.cfg)
        self.assertEqual(sim.trades.exit_reason.iloc[0], "stop")
        self.assertEqual(sim.trades.exit_price.iloc[0], 99)
        self.assertLess(sim.trades.pnl.iloc[0], 0)

    def test_gap_exceeds_stop_budget(self):
        self.bars.iloc[2, self.bars.columns.get_loc("Open")] = 80
        self.bars.iloc[2, self.bars.columns.get_loc("Low")] = 79
        sim = simulate(self.bars, self.p, self.features, self.risk, self.cfg)
        self.assertEqual(sim.trades.exit_price.iloc[0], 80)
        self.assertLess(sim.trades.pnl.iloc[0], -self.risk.initial_capital*self.risk.risk_per_trade)
        self.assertEqual(sim.ledger.Halted.iloc[2], 1)

    def test_daily_loss_latches_until_next_session(self):
        guard = DailyRiskGuard(self.risk)
        guard.begin("2026-09-21", 10000)
        self.assertTrue(guard.observe(9499))
        self.assertTrue(guard.observe(10500))
        guard.begin("2026-09-21", 10500)
        self.assertTrue(guard.halted)
        guard.begin("2026-09-22", 10500)
        self.assertFalse(guard.halted)

    def test_daily_stop_blocks_later_signals(self):
        risk = replace(self.risk, risk_per_trade=.01, daily_loss_limit=.01)
        self.p[:] = 1
        self.bars.iloc[1, self.bars.columns.get_loc("Low")] = 98
        sim = simulate(self.bars, self.p, self.features, risk, self.cfg)
        self.assertEqual(len(sim.trades), 1)
        self.assertTrue((sim.ledger.Halted.iloc[1:] == 1).all())

    def test_close_before_end_and_no_overnight(self):
        self.p[:] = 1
        sim = simulate(self.bars, self.p, self.features, self.risk, self.cfg)
        self.assertEqual(sim.ledger.Position.iloc[-1], 0)
        self.assertTrue((pd.to_datetime(sim.trades.exit_time, utc=True) <= self.bars.Start.iloc[-1]).all())
        self.assertTrue((sim.trades.entry_time.dt.date == sim.trades.exit_time.dt.date).all())

    def test_position_size_uses_previous_volume(self):
        self.bars.iloc[0, self.bars.columns.get_loc("Volume")] = 0
        sim = simulate(self.bars, self.p, self.features, self.risk, self.cfg)
        self.assertTrue(sim.trades.empty)

    def test_leverage_is_cap_not_target(self):
        risk = replace(self.risk, max_leverage=3)
        sim = simulate(self.bars, self.p, self.features, risk, self.cfg)
        self.assertLessEqual(sim.trades.notional_leverage.max(), 3)
        self.assertLess(sim.trades.notional_leverage.iloc[0], 1)

    def test_roundtrip_costs_on_flat_prices(self):
        risk = replace(self.risk, fee_bps=10, slippage_bps=5)
        sim = reference_curve(self.bars, risk)
        expected = 10000*(1-.0005)*(1-.001)/((1+.0005)*(1+.001))
        self.assertAlmostEqual(sim.ledger.Capital.iloc[-1], expected)

    def test_sharpe_uses_days_and_initial_drawdown(self):
        sim = reference_curve(self.bars, replace(self.risk, fee_bps=10))
        m = performance(sim, 10000)
        self.assertEqual(m["sessions"], 1)
        self.assertTrue(np.isnan(m["sharpe"]))
        self.assertLess(m["max_drawdown"], 0)

    def test_no_trade_without_prediction(self):
        self.p[:] = np.nan
        sim = simulate(self.bars, self.p, self.features, self.risk, self.cfg)
        self.assertEqual(sim.ledger.Capital.iloc[-1], 10000)
        self.assertEqual(sim.ledger.PositionChanges.sum(), 0)


class InterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = LabService(DemoProvider(), AppConfig(symbols=("SPY", "AAPL")), "demo")
        cls.server = create_server(cls.service, 0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close(); cls.thread.join(timeout=5)

    def get(self, path):
        with urllib.request.urlopen(self.url+path, timeout=120) as response:
            return json.load(response)

    def test_http_graphic_all_timeframes(self):
        for minutes, count in ((5, 54), (10, 27), (15, 18)):
            data = self.get(f"/api/analysis?symbol=SPY&timeframe={minutes}&minute=270")
            self.assertEqual(len(data["bars"]), count)
            self.assertEqual(data["scanner"]["confidence"], None)
            self.assertTrue(data["scanner"]["synthetic"])

    def test_scan_and_chat(self):
        scan = self.get("/api/scan?timeframe=5")
        self.assertEqual(len(scan["items"]), 2)
        body = json.dumps(dict(symbol="SPY", question="Quelle confiance ?")).encode()
        request = urllib.request.Request(self.url+"/api/chat", data=body, headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(request, timeout=120) as response:
            chat = json.load(response)
        self.assertFalse(chat["orders_enabled"])
        self.assertIn("ne mesure pas", chat["answer"])

    def test_unknown_symbol_and_cross_origin_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.get("/api/analysis?symbol=NOT_ALLOWED")
        self.assertEqual(caught.exception.code, 400)
        request = urllib.request.Request(self.url+"/api/chat", data=b"{}", headers={"Origin":"https://example.com"})
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request)
        self.assertEqual(caught.exception.code, 403)


if __name__ == "__main__":
    unittest.main()
