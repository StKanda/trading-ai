"""Réutilise les calculs V7 ; ajoute VWAP/session/volume comparable dans le temps.

Les fenêtres V7 sont ici des nombres de BOUGIES, et non des jours. Les W/M et
niveaux intraday sont réinitialisés chaque séance. Les paramètres sont initiaux,
fixes, non optimisés ; aucune transposition de performance du modèle quotidien.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .legacy_v7 import (classic_features, causal_structure, FeatureConfig, GROUPS as V7_GROUPS,
                        LEVELS, WM, SCORES, BREAKOUT_VOLUME)

SESSION_CLASSIC = ["DistVWAP", "SessionReturn", "SessionProgress", "RelativeVolumeSlot"]
PREVIOUS_LEVELS = ["DistPreviousHigh", "DistPreviousLow", "PreviousLevelsMissing"]
GROUPS = {name: list(columns) + SESSION_CLASSIC + (PREVIOUS_LEVELS if name != "A" else [])
          for name, columns in V7_GROUPS.items()}


def intraday_pattern_config():
    return FeatureConfig(pivot_left=2, pivot_right=2, level_lookback_days=1,
                         level_tolerance=0.0015, pattern_tolerance=0.0025,
                         pattern_min_height=0.0015, pattern_min_span=6,
                         pattern_max_span=40, pattern_max_wait=12,
                         pattern_previous_pivots=5, pattern_cooldown=6, pattern_memory=12)


@dataclass
class FeatureSet:
    values: pd.DataFrame
    pivots: pd.DataFrame
    events: pd.DataFrame
    pending: list


def pending_patterns(prices, pivots, events, cfg):
    """Reconstruit seulement l'état présent à partir de pivots déjà confirmés.

    Aucune formation n'est appelée W/M avant la confirmation du second pivot.
    Un tracé en attente de neckline ne déclenche pas de signal de cassure.
    """
    found, n = [], len(prices)
    records = pivots.to_dict("records")
    processed = set(events.pattern_id) if len(events) else set()
    close = prices.Close.to_numpy()
    for end in records:
        kind = "W" if end["type"] == "LOW" else "M"
        same = [p for p in records if p["type"] == end["type"]
                and cfg.pattern_min_span <= end["pivot_pos"] - p["pivot_pos"] <= cfg.pattern_max_span]
        for start in same[-cfg.pattern_previous_pivots:]:
            middle = [p for p in records if p["type"] != end["type"]
                      and start["pivot_pos"] < p["pivot_pos"] < end["pivot_pos"]]
            if not middle:
                continue
            mid = (max if kind == "W" else min)(middle, key=lambda p: p["price"])
            a, b, neck = start["price"], end["price"], mid["price"]
            mismatch = abs(a - b) / ((a + b) / 2)
            height = (neck - max(a, b)) / neck if kind == "W" else (min(a, b) - neck) / neck
            pid = f"{kind}_{start['pivot_pos']}_{mid['pivot_pos']}_{end['pivot_pos']}"
            formation = end["confirm_pos"]
            if (mismatch > cfg.pattern_tolerance or height < cfg.pattern_min_height or pid in processed
                    or n - 1 - formation > cfg.pattern_max_wait):
                continue
            invalidation = min(a, b) * (1 - cfg.pattern_tolerance) if kind == "W" else max(a, b) * (1 + cfg.pattern_tolerance)
            since = close[formation:]
            invalid = (since < invalidation).any() if kind == "W" else (since > invalidation).any()
            if invalid:
                continue
            balance = min(mid["pivot_pos"] - start["pivot_pos"], end["pivot_pos"] - mid["pivot_pos"]) / max(
                mid["pivot_pos"] - start["pivot_pos"], end["pivot_pos"] - mid["pivot_pos"])
            score = 100 * (0.5 * (1 - mismatch / cfg.pattern_tolerance) + 0.25 * balance + 0.25 * min(height / 0.08, 1))
            found.append(dict(pattern=kind, status="neckline en attente", quality_score=score,
                              neckline=neck, formation_confirm_date=end["confirm_date"],
                              points=[(p["pivot_date"], p["price"]) for p in (start, mid, end)]))
    return found[-8:]


class FeatureEngine:
    def build(self, bars, latest_only=False):
        classic = classic_features(bars)
        structure_parts, pivot_parts, event_parts, pending = [], [], [], []
        cfg = intraday_pattern_config()
        grouped = list(bars.groupby("Session", sort=True))
        for day, part in (grouped[-1:] if latest_only else grouped):
            structure, pivots, events = causal_structure(part, cfg)
            structure_parts.append(structure)
            pivots["Session"] = day
            events["Session"] = day
            pivot_parts.append(pivots)
            event_parts.append(events)
            if day == grouped[-1][0]:
                pending = pending_patterns(part, pivots, events, cfg)
        result = classic.join(pd.concat(structure_parts))
        typical = (bars.High + bars.Low + bars.Close) / 3
        cumvol = bars.Volume.groupby(bars.Session).cumsum()
        vwap = (typical * bars.Volume).groupby(bars.Session).cumsum() / cumvol.replace(0, np.nan)
        result["VWAP"] = vwap
        result["DistVWAP"] = bars.Close / vwap - 1
        session_open = bars.groupby("Session").Open.transform("first")
        result["SessionReturn"] = bars.Close / session_open - 1
        result["SessionProgress"] = ((bars.index.to_series() - bars.SessionOpen).dt.total_seconds()
                                     / (bars.SessionClose - bars.SessionOpen).dt.total_seconds())
        # Un même créneau ne compare son volume qu'aux séances ANTÉRIEURES.
        slot = ((bars.Start - bars.SessionOpen).dt.total_seconds() / 60).astype(int)
        reference = bars.Volume.groupby(slot).transform(lambda s: s.shift(1).rolling(20, min_periods=3).mean())
        result["RelativeVolumeSlot"] = bars.Volume / reference.replace(0, np.nan)
        daily = bars.groupby("Session").agg(High=("High", "max"), Low=("Low", "min"))
        previous_high = bars.Session.map(daily.High.shift(1))
        previous_low = bars.Session.map(daily.Low.shift(1))
        result["PreviousHigh"] = previous_high
        result["PreviousLow"] = previous_low
        result["DistPreviousHigh"] = (previous_high / bars.Close - 1).fillna(0)
        result["DistPreviousLow"] = (bars.Close / previous_low - 1).fillna(0)
        result["PreviousLevelsMissing"] = previous_high.isna().astype(float)
        def combine(parts):
            populated = [part for part in parts if not part.empty]
            return pd.concat(populated, ignore_index=True) if populated else parts[0].copy()
        return FeatureSet(result.replace([np.inf, -np.inf], np.nan),
                          combine(pivot_parts), combine(event_parts), pending)


def build_labels(bars, config, risk):
    """Entrée à l'ouverture suivante, sortie à +30 min, avant aplatissement.

    Classification : rendement net théorique d'une détention à horizon fixe > 0.
    Le moteur de gestion peut sortir plus tôt au stop/objectif : cette différence
    est explicite et les résultats de classification ne sont pas ceux des trades.
    """
    minutes = bars.attrs.get("timeframe") or int((bars.index[0] - bars.Start.iloc[0]).total_seconds() / 60)
    h = config.horizon_minutes // minutes
    entry = bars.Open.shift(-1)
    exit_price = bars.Open.shift(-(h + 1))
    exit_time = bars.Start.shift(-(h + 1))
    valid = (bars.Session.shift(-(h + 1)) == bars.Session) & (
        exit_time <= bars.SessionClose - pd.Timedelta(minutes=minutes))
    fee, slip = risk.fee_bps / 10000, risk.slippage_bps / 10000
    future = exit_price * (1 - slip) * (1 - fee) / (entry * (1 + slip) * (1 + fee)) - 1
    future = future.where(valid)
    return pd.DataFrame({"FutureNetReturn": future, "LabelEnd": exit_time.where(valid),
                         "Target": (future > 0).astype(float).where(future.notna())}, index=bars.index)
