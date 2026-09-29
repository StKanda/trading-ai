import unittest
import numpy as np
import pandas as pd

from trading_lab.data.providers import DemoProvider
from trading_lab.data.market import completed_bars
from trading_lab.features.engine import FeatureEngine, GROUPS
from trading_lab.config import AppConfig
from trading_lab.terminal_pipeline import (
    _candidate_setup_mask, _economic_score, _inner_masks_for_economic, _walkforward_fold_starts,
)
from trading_lab.deep_learning.economic import build_economic_labels


def _economic_split_fixture(outer_train_sessions=36, total_sessions=42):
    session_dates = pd.bdate_range("2025-01-02", periods=total_sessions)
    starts = []
    sessions = []
    session_opens = []
    session_open_times = []
    for day in session_dates:
        session_open = pd.Timestamp(f"{day:%Y-%m-%d} 14:30", tz="UTC")
        starts.extend(pd.date_range(session_open, periods=20, freq="5min"))
        sessions.extend([day.strftime("%Y-%m-%d")] * 20)
        session_opens.extend([session_open] * 20)
        session_open_times.append(session_open)

    index = pd.DatetimeIndex(starts)
    bars = pd.DataFrame({"Session": sessions, "SessionOpen": session_opens}, index=index)
    features = pd.DataFrame(1.0, index=index, columns=GROUPS["D"])
    economic_labels = pd.DataFrame({
        "ProfitableAfterCosts": 0.0,
        "StopHit": 0.0,
        "TargetHit": 0.0,
        "TimeExit": 1.0,
        "RealizedPnL": 1.0,
        "RealizedReturn": 0.01,
        "MFE": 0.02,
        "MAE": -0.01,
    }, index=index)
    economic_labels["LabelEnd"] = index + pd.Timedelta(minutes=5)
    candidate_mask = pd.Series(True, index=index)

    oos_boundary = session_open_times[outer_train_sessions]
    oos_first_pos = index.searchsorted(oos_boundary, side="left")
    research = AppConfig().research
    outer_gap = research.horizon_minutes // 5 + 1 + research.extra_embargo_bars
    outer_train_mask = (np.arange(len(bars)) < oos_first_pos - outer_gap)
    outer_train_mask &= (economic_labels["LabelEnd"] < oos_boundary).to_numpy()
    oos_mask = bars["Session"].isin(
        [day.strftime("%Y-%m-%d") for day in session_dates[outer_train_sessions:]]
    ).to_numpy()
    return bars, features, economic_labels, candidate_mask, outer_train_mask, oos_mask, research


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

    def test_inner_validation_default_preserves_historical_session_formula(self):
        fixture = _economic_split_fixture(outer_train_sessions=12, total_sessions=18)
        bars, features, labels, candidates, outer_train, _, research = fixture
        _, _, validation_sessions = _inner_masks_for_economic(
            bars, features, labels, outer_train, 5, research, candidates
        )
        self.assertEqual(len(validation_sessions), max(2, min(4, 12 // 4)))

    def test_inner_validation_uses_exact_explicit_session_count(self):
        fixture = _economic_split_fixture()
        bars, features, labels, candidates, outer_train, _, research = fixture
        _, valid_mask, validation_sessions = _inner_masks_for_economic(
            bars, features, labels, outer_train, 5, research, candidates,
            economic_validation_sessions=8,
        )
        eligible_sessions = list(pd.unique(bars.loc[outer_train, "Session"]))
        self.assertEqual(validation_sessions, eligible_sessions[-8:])
        self.assertEqual(bars.loc[valid_mask, "Session"].nunique(), 8)

    def test_explicit_inner_validation_rejects_invalid_unavailable_or_insufficient_split(self):
        fixture = _economic_split_fixture()
        bars, features, labels, candidates, outer_train, _, research = fixture
        for count, message in ((1, "supérieur ou égal à 2"), (37, "seulement 36 séances"),
                               (36, "Impossible d'entraîner")):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, message):
                _inner_masks_for_economic(
                    bars, features, labels, outer_train, 5, research, candidates,
                    economic_validation_sessions=count,
                )

    def test_inner_validation_does_not_overlap_train_or_oos_or_change_oos_plan(self):
        fixture = _economic_split_fixture()
        bars, features, labels, candidates, outer_train, oos_mask, research = fixture
        outer_train_before = outer_train.copy()
        oos_sessions_before = list(pd.unique(bars.loc[oos_mask, "Session"]))
        oos_fold_plan_before = list(_walkforward_fold_starts(42, 10, 10))
        train_mask, valid_mask, validation_sessions = _inner_masks_for_economic(
            bars, features, labels, outer_train, 5, research, candidates,
            economic_validation_sessions=8,
        )

        self.assertFalse(np.any(train_mask & valid_mask))
        self.assertFalse(np.any(train_mask & oos_mask))
        self.assertFalse(np.any(valid_mask & oos_mask))
        np.testing.assert_array_equal(outer_train, outer_train_before)
        oos_sessions = list(pd.unique(bars.loc[oos_mask, "Session"]))
        self.assertEqual(oos_sessions, oos_sessions_before)
        self.assertEqual(list(_walkforward_fold_starts(42, 10, 10)), oos_fold_plan_before)
        self.assertEqual(oos_fold_plan_before, [10, 20, 30, 40])

        validation_boundary = bars.loc[bars.Session == validation_sessions[0], "SessionOpen"].iloc[0]
        oos_boundary = bars.loc[oos_mask, "SessionOpen"].iloc[0]
        self.assertLess(labels.loc[train_mask, "LabelEnd"].max(), validation_boundary)
        self.assertLess(labels.loc[valid_mask, "LabelEnd"].max(), oos_boundary)
