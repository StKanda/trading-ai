from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from trading_lab.data.market import Snapshot, TZ, completed_bars
from trading_lab.config import ResearchConfig, RiskConfig
from trading_lab.deep_learning.dataset import build_sequence_bundle
from trading_lab.deep_learning.train import train_classifier
from trading_lab.features.engine import (
    FeatureEngine,
    GROUPS,
    MTF_DAILY_COLUMNS,
    MTF_INTRADAY_COLUMNS,
    _asof_features,
    _daily_context,
)
from trading_lab.research.backtest import simulate
from trading_lab.terminal_pipeline import _inner_masks_for_economic


def make_snapshot(days, close_time="16:00", halt_statuses=None):
    schedule_rows = []
    price_rows = []
    for day_number, day in enumerate(days):
        opened = pd.Timestamp(f"{day} 09:30", tz=TZ).tz_convert("UTC")
        closed = pd.Timestamp(f"{day} {close_time}", tz=TZ).tz_convert("UTC")
        schedule_rows.append((day, opened, closed))
        for bar_number, start in enumerate(pd.date_range(opened, closed, freq="5min", inclusive="left")):
            base = 100.0 + day_number * 2.0 + bar_number * 0.01
            price_rows.append((start, base, base + 0.03, base - 0.02, base + 0.01, 100.0 + bar_number))

    schedule = pd.DataFrame(schedule_rows, columns=["Session", "MarketOpen", "MarketClose"]).set_index("Session")
    prices = pd.DataFrame(price_rows, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"]).set_index("Timestamp")
    prices.index = pd.DatetimeIndex(prices.index)
    fetched = schedule.MarketClose.iloc[-1]
    metadata = {"symbol": "SPY", "synthetic": False, "bar_minutes": 5,
                "timestamp_convention": "bar_start", "fetched_at": fetched.isoformat()}
    if halt_statuses:
        metadata["_session_quality_statuses"] = halt_statuses
    return Snapshot("SPY", prices, schedule, metadata)


class MultiTimeframeMarketTests(unittest.TestCase):
    def test_15m_and_1h_ohlcv_aggregation_is_exact(self):
        snapshot = make_snapshot(["2026-09-21"])
        bars15 = completed_bars(snapshot, 15, asof=snapshot.schedule.MarketClose.iloc[-1])
        bars60 = completed_bars(snapshot, 60, asof=snapshot.schedule.MarketClose.iloc[-1])
        source = snapshot.prices
        first15 = bars15.iloc[0]
        part15 = source.iloc[:3]
        self.assertEqual(first15.Open, part15.Open.iloc[0])
        self.assertEqual(first15.High, part15.High.max())
        self.assertEqual(first15.Low, part15.Low.min())
        self.assertEqual(first15.Close, part15.Close.iloc[-1])
        self.assertEqual(first15.Volume, part15.Volume.sum())
        first60 = bars60.iloc[0]
        part60 = source.iloc[:12]
        self.assertEqual(first60.Open, part60.Open.iloc[0])
        self.assertEqual(first60.High, part60.High.max())
        self.assertEqual(first60.Low, part60.Low.min())
        self.assertEqual(first60.Close, part60.Close.iloc[-1])
        self.assertEqual(first60.Volume, part60.Volume.sum())
        self.assertEqual(len(bars60), 6)

    def test_incomplete_higher_timeframe_bucket_is_not_exposed(self):
        snapshot = make_snapshot(["2026-09-21"])
        opened = snapshot.schedule.MarketOpen.iloc[0]
        self.assertTrue(completed_bars(snapshot, 15, asof=opened + pd.Timedelta(minutes=14)).empty)
        self.assertEqual(len(completed_bars(snapshot, 15, asof=opened + pd.Timedelta(minutes=15))), 1)
        self.assertTrue(completed_bars(snapshot, 60, asof=opened + pd.Timedelta(minutes=59)).empty)
        self.assertEqual(len(completed_bars(snapshot, 60, asof=opened + pd.Timedelta(minutes=60))), 1)

    def test_aggregation_never_crosses_sessions(self):
        snapshot = make_snapshot(["2026-09-21", "2026-09-22"])
        bars = completed_bars(snapshot, 60, asof=snapshot.schedule.MarketClose.iloc[-1])
        self.assertEqual(bars.groupby("Session").size().tolist(), [6, 6])
        for _, session_bars in bars.groupby("Session"):
            self.assertTrue((session_bars.Start >= session_bars.SessionOpen).all())
            self.assertTrue((session_bars.index <= session_bars.SessionClose).all())

    def test_early_close_and_dst_keep_session_anchored_hours(self):
        snapshot = make_snapshot(["2026-03-06", "2026-03-09"], close_time="13:00")
        bars15 = completed_bars(snapshot, 15, asof=snapshot.schedule.MarketClose.iloc[-1])
        bars60 = completed_bars(snapshot, 60, asof=snapshot.schedule.MarketClose.iloc[-1])
        self.assertEqual(bars15.groupby("Session").size().tolist(), [14, 14])
        self.assertEqual(bars60.groupby("Session").size().tolist(), [3, 3])
        self.assertEqual(snapshot.schedule.MarketOpen.iloc[0].hour, 14)
        self.assertEqual(snapshot.schedule.MarketOpen.iloc[1].hour, 13)

    def test_certified_market_halt_does_not_create_synthetic_prices(self):
        day = "2020-03-09"
        snapshot = make_snapshot([day])
        halt = pd.DatetimeIndex(pd.to_datetime(["2020-03-09T13:35Z", "2020-03-09T13:40Z"], utc=True))
        snapshot.prices = snapshot.prices.drop(halt)
        snapshot.metadata["_session_quality_statuses"] = {day: {"status": "MARKET_HALT_COMPLETE"}}
        bars = completed_bars(snapshot, 15, asof=snapshot.schedule.MarketClose.iloc[0])
        self.assertEqual(len(bars), 26)
        self.assertEqual(bars.iloc[0].Volume, snapshot.prices.loc[
            (snapshot.prices.index >= bars.iloc[0].Start) & (snapshot.prices.index < bars.index[0]), "Volume"
        ].sum())


class MultiTimeframeFeatureTests(unittest.TestCase):
    def test_higher_feature_is_hidden_before_close_and_visible_at_close(self):
        context_index = pd.DatetimeIndex([pd.Timestamp("2026-09-21T14:30:00Z")])
        context = pd.DataFrame({"M15_RETURN_4": [0.125]}, index=context_index)
        signals = pd.DatetimeIndex(pd.to_datetime([
            "2026-09-21T14:29:00Z", "2026-09-21T14:30:00Z", "2026-09-21T14:35:00Z"
        ], utc=True))
        aligned = _asof_features(signals, context)
        self.assertTrue(pd.isna(aligned.iloc[0, 0]))
        self.assertEqual(aligned.iloc[1, 0], 0.125)
        self.assertEqual(aligned.iloc[2, 0], 0.125)

    def test_daily_context_uses_only_previous_closed_session_intraday(self):
        sessions = pd.bdate_range("2026-01-02", periods=56).strftime("%Y-%m-%d").tolist()
        snapshot = make_snapshot(sessions)
        current_session = sessions[-1]
        opened = snapshot.schedule.loc[current_session, "MarketOpen"]
        close_so_far = opened + pd.Timedelta(minutes=10)
        signal_bars = pd.DataFrame({
            "Session": [current_session], "Close": [250.0],
        }, index=pd.DatetimeIndex([close_so_far]))
        context = _daily_context(signal_bars, snapshot)
        previous_session = sessions[-2]
        previous_daily = snapshot.prices.loc[
            (snapshot.prices.index >= snapshot.schedule.loc[previous_session, "MarketOpen"])
            & (snapshot.prices.index < snapshot.schedule.loc[previous_session, "MarketClose"])
        ]
        expected_high_distance = 250.0 / previous_daily.High.max() - 1
        self.assertAlmostEqual(context["D1_DIST_PREV_HIGH"].iloc[0], expected_high_distance)
        self.assertEqual(context.columns.tolist(), MTF_DAILY_COLUMNS)
        revised_snapshot = Snapshot(snapshot.symbol, snapshot.prices.copy(), snapshot.schedule.copy(),
                       snapshot.metadata.copy())
        current_prices = revised_snapshot.prices.index >= opened
        revised_snapshot.prices.loc[current_prices, ["Open", "High", "Low", "Close"]] *= 3
        revised = _daily_context(signal_bars, revised_snapshot)
        assert_frame_equal(context, revised, check_exact=True)

    def test_mtf_feature_prefix_invariance_and_future_shock(self):
        snapshot = make_snapshot(pd.bdate_range("2026-08-03", periods=8).strftime("%Y-%m-%d").tolist())
        bars = completed_bars(snapshot, 5, asof=snapshot.schedule.MarketClose.iloc[-1], require_complete=True)
        baseline = FeatureEngine().build(bars).values
        explicit_baseline = FeatureEngine().build(bars, mtf_variant="baseline", snapshot=snapshot).values
        assert_frame_equal(baseline, explicit_baseline, check_exact=True)
        self.assertFalse(set(MTF_INTRADAY_COLUMNS + MTF_DAILY_COLUMNS).intersection(GROUPS["D"]))
        full_variant = FeatureEngine().build(bars, mtf_variant="mtf_full", snapshot=snapshot).values
        self.assertTrue(set(MTF_INTRADAY_COLUMNS + MTF_DAILY_COLUMNS).issubset(full_variant.columns))

        cut = len(bars) - 90
        prefix = bars.iloc[:cut].copy()
        prefix.attrs.update(bars.attrs)
        prefix_features = FeatureEngine().build(prefix, mtf_variant="mtf_intraday", snapshot=snapshot).values
        full_features = FeatureEngine().build(bars, mtf_variant="mtf_intraday", snapshot=snapshot).values
        assert_frame_equal(prefix_features, full_features.iloc[:cut], check_exact=True)
        self.assertEqual(full_features[MTF_INTRADAY_COLUMNS].columns.tolist(), MTF_INTRADAY_COLUMNS)

        shocked = bars.copy()
        shocked.loc[shocked.index[cut:], ["Open", "High", "Low", "Close"]] *= 1.7
        shocked.loc[shocked.index[cut:], "Volume"] *= 7
        shocked_snapshot = Snapshot(snapshot.symbol, snapshot.prices.copy(), snapshot.schedule.copy(),
                                    snapshot.metadata.copy())
        future_source = shocked_snapshot.prices.index >= bars.index[cut]
        shocked_snapshot.prices.loc[future_source, ["Open", "High", "Low", "Close"]] *= 1.7
        shocked_snapshot.prices.loc[future_source, "Volume"] *= 7
        shocked_features = FeatureEngine().build(
            shocked, mtf_variant="mtf_intraday", snapshot=shocked_snapshot
        ).values
        assert_frame_equal(full_features.iloc[:cut], shocked_features.iloc[:cut], check_exact=True)

    def test_mtf_scaler_fits_training_rows_only(self):
        n = 80
        index = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
        features = pd.DataFrame({
            "M15_EMA_SPREAD_20_50": np.r_[np.arange(60, dtype=float), np.full(20, 50000.0)],
            "H1_RETURN_3": np.arange(n, dtype=float),
        }, index=index)
        labels = pd.DataFrame({"Target": np.arange(n) % 2}, index=index)
        train = np.zeros(n, dtype=bool); train[:55] = True
        valid = np.zeros(n, dtype=bool); valid[60:] = True
        bundle = build_sequence_bundle(features, labels, train, valid, context_bars=5,
                                       columns=MTF_INTRADAY_COLUMNS[:1] + ["H1_RETURN_3"])
        self.assertLess(bundle.scaler.mean_[0], 100.0)
        self.assertEqual(bundle.columns, MTF_INTRADAY_COLUMNS[:1] + ["H1_RETURN_3"])

    def test_inner_economic_masks_exclude_unavailable_mtf_rows(self):
        sessions = pd.bdate_range("2025-01-02", periods=32).strftime("%Y-%m-%d").tolist()
        rows = []
        index = []
        for session_number, session in enumerate(sessions):
            opened = pd.Timestamp(f"{session} 14:30", tz="UTC")
            for bar_number in range(40):
                index.append(opened + pd.Timedelta(minutes=5 * (bar_number + 1)))
                rows.append((session, opened))
        bars = pd.DataFrame(rows, columns=["Session", "SessionOpen"], index=pd.DatetimeIndex(index))
        features = pd.DataFrame({"base": 1.0, "M15_RETURN_4": 2.0}, index=bars.index)
        labels = pd.DataFrame(1.0, index=bars.index, columns=[
            "ProfitableAfterCosts", "StopHit", "TargetHit", "TimeExit", "RealizedPnL",
            "RealizedReturn", "MFE", "MAE",
        ])
        labels["LabelEnd"] = bars.index - pd.Timedelta(minutes=1)
        candidates = pd.Series(True, index=bars.index)
        outer = np.ones(len(bars), dtype=bool)
        features.iloc[20, features.columns.get_loc("M15_RETURN_4")] = np.nan
        features.iloc[-1, features.columns.get_loc("M15_RETURN_4")] = np.nan
        train, valid, _ = _inner_masks_for_economic(
            bars, features, labels, outer, 5, ResearchConfig(), candidates,
            feature_columns=["base", "M15_RETURN_4"],
        )
        self.assertFalse(train[20])
        self.assertFalse(valid[-1])

    def test_progress_callbacks_do_not_change_seeded_predictions(self):
        n = 72
        index = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
        rng = np.random.default_rng(21)
        features = pd.DataFrame({"M15_RETURN_4": rng.normal(size=n), "H1_ATR14_NORM": rng.normal(size=n)}, index=index)
        labels = pd.DataFrame({"Target": np.arange(n) % 2}, index=index)
        train = np.zeros(n, dtype=bool); train[:50] = True
        valid = np.zeros(n, dtype=bool); valid[55:70] = True
        bundle = build_sequence_bundle(features, labels, train, valid, context_bars=4,
                                       columns=list(features.columns))
        plain = train_classifier(bundle, architecture="mlp", hidden_size=8, max_epochs=3,
                                 patience=2, seed=123)
        events = []
        observed = train_classifier(bundle, architecture="mlp", hidden_size=8, max_epochs=3,
                                    patience=2, seed=123, progress_callback=events.append)
        np.testing.assert_array_equal(plain.validation_scores, observed.validation_scores)
        epoch_events = [event for event in events if event["event"] == "epoch"]
        finished = events[-1]
        self.assertEqual(finished["epochs_run"], len(epoch_events))
        self.assertEqual(finished["early_stopped"], any(event["early_stopping"] for event in epoch_events))
        self.assertEqual(finished["best_epoch"], epoch_events[-1]["best_epoch"])

    def test_simulation_progress_callback_preserves_ledger_and_trades(self):
        snapshot = make_snapshot(["2026-09-21"])
        bars = completed_bars(snapshot, 5, asof=snapshot.schedule.MarketClose.iloc[-1], require_complete=True)
        features = pd.DataFrame({"ATR14Pct": 0.005}, index=bars.index)
        scores = pd.Series(np.nan, index=bars.index)
        scores.iloc[0] = 1.0
        risk = RiskConfig(fee_bps=0, slippage_bps=0, borrowing_bps_per_day=0)
        research = ResearchConfig()
        plain = simulate(bars, scores, features, risk, research)
        updates = []
        observed = simulate(bars, scores, features, risk, research, progress_callback=updates.append)
        assert_frame_equal(plain.ledger, observed.ledger, check_exact=True)
        assert_frame_equal(plain.trades, observed.trades, check_exact=True)
        self.assertEqual(len(updates), len(bars))
        self.assertEqual(updates[-1]["completed"], len(bars))

    def test_cli_forwards_feature_variant_progress_and_max_folds(self):
        from trading_lab.cli import main

        with patch("trading_lab.cli.AppConfig.load"), \
                patch("trading_lab.terminal_pipeline.run_terminal_pipeline") as pipeline:
            main(["pipeline", "--symbol", "SPY", "--data", "snapshot", "--out", "out",
                  "--feature-variant", "mtf_full", "--progress", "--max-folds", "2"])
        self.assertEqual(pipeline.call_args.kwargs["feature_variant"], "mtf_full")
        self.assertTrue(pipeline.call_args.kwargs["progress"])
        self.assertEqual(pipeline.call_args.kwargs["max_folds"], 2)


if __name__ == "__main__":
    unittest.main()
