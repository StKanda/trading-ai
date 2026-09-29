"""Features V7 dérivées de trading_ai_v6.py, sans lecture de ses CSV V5.

Toute feature d'une date t est disponible APRES sa clôture. Les pivots sont
datés de leur confirmation, les événements acceptés ne sont jamais révisés.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    pivot_left: int = 3
    pivot_right: int = 3
    level_lookback_days: int = 730
    level_tolerance: float = 0.01
    pattern_tolerance: float = 0.03
    pattern_min_height: float = 0.02
    pattern_min_span: int = 6
    pattern_max_span: int = 120
    pattern_max_wait: int = 30
    pattern_previous_pivots: int = 5
    pattern_cooldown: int = 10
    pattern_memory: int = 30


# Les trois Momentum de la V6 sont des copies exactes de Return5/10/20.
CLASSIC = (
    [f"Return{w}" for w in (1, 2, 3, 5, 10, 20, 50)]
    + [f"DistSMA{w}" for w in (5, 10, 20, 50, 100, 200)]
    + ["SMA20Slope5", "SMA50Slope10", "SMA200Slope20",
       "Trend20Above50", "Trend50Above200", "PriceAbove200"]
    + [f"Volatility{w}" for w in (5, 10, 20, 50)]
    + ["ATR14Pct", "HighLowRange", "OpenCloseRange", "CloseLocation",
       "VolumeRatio5", "VolumeRatio20", "VolumeRatio50", "VolumeChange1",
       "VolumeChange5", "DistHigh20", "DistLow20", "DistHigh50", "DistLow50",
       "RSI14Norm"]
)
LEVELS = [
    "DistSupport", "DistResistance", "SupportTouches", "ResistanceTouches",
    "NearSupport1Pct", "NearSupport2Pct", "NearSupport5Pct",
    "NearResistance1Pct", "NearResistance2Pct", "NearResistance5Pct",
    "LevelBalance", "SupportMissing", "ResistanceMissing",
]
WM = [f"{p}{s}" for p in ("W", "M") for s in
      ("Signal", "Recent5", "Recent10", "Recent20", "Recent30", "Recency",
       "NearLevel", "LevelTouches")]
WM += [f"Last{p}{s}" for p in ("W", "M") for s in ("NearLevel", "LevelTouches")]
SCORES = ["WScore", "MScore", "LastWScoreNorm", "LastMScoreNorm"]
BREAKOUT_VOLUME = ["WVolumeRatio", "MVolumeRatio", "LastWVolumeRatio", "LastMVolumeRatio"]
GROUPS = {
    "A": CLASSIC,
    "B": CLASSIC + LEVELS,
    "C": CLASSIC + LEVELS + WM,
    "C_SCORE": CLASSIC + LEVELS + WM + SCORES,
    "C_VOLUME": CLASSIC + LEVELS + WM + BREAKOUT_VOLUME,
    "D": CLASSIC + LEVELS + WM + SCORES + BREAKOUT_VOLUME,
}
PIVOT_COLUMNS = ["pivot_pos", "confirm_pos", "pivot_date", "confirm_date", "type", "price"]
EVENT_COLUMNS = [
    "pattern_id", "pattern", "start_pos", "middle_pos", "end_pos",
    "start_date", "middle_date", "end_date", "formation_confirm_date",
    "breakout_pos", "breakout_date", "neckline", "quality_score",
    "volume_ratio", "near_level", "level_touches", "accepted", "reason",
]


def classic_features(prices):
    """Reprise des calculs V6 236–767, RSI sans pertes corrigé."""
    out = pd.DataFrame(index=prices.index)
    close, op, hi, lo, vol = (prices[x] for x in ("Close", "Open", "High", "Low", "Volume"))
    for w in (1, 2, 3, 5, 10, 20, 50):
        out[f"Return{w}"] = close.pct_change(w, fill_method=None)
    smas = {w: close.rolling(w).mean() for w in (5, 10, 20, 50, 100, 200)}
    for w, sma in smas.items():
        out[f"DistSMA{w}"] = close / sma - 1
    for w, lag in ((20, 5), (50, 10), (200, 20)):
        out[f"SMA{w}Slope{lag}"] = smas[w] / smas[w].shift(lag) - 1
    out["Trend20Above50"] = (smas[20] > smas[50]).astype(float)
    out["Trend50Above200"] = (smas[50] > smas[200]).astype(float)
    out["PriceAbove200"] = (close > smas[200]).astype(float)
    for w in (5, 10, 20, 50):
        out[f"Volatility{w}"] = out["Return1"].rolling(w).std()
    true_range = pd.concat([hi - lo, (hi - close.shift()).abs(),
                            (lo - close.shift()).abs()], axis=1).max(axis=1)
    out["ATR14Pct"] = true_range.rolling(14).mean() / close
    out["HighLowRange"] = (hi - lo) / close
    out["OpenCloseRange"] = close / op - 1
    out["CloseLocation"] = ((close - lo) / (hi - lo).replace(0, np.nan)).fillna(0.5)
    for w in (5, 20, 50):
        out[f"VolumeRatio{w}"] = vol / vol.rolling(w).mean()
    for w in (1, 5):
        out[f"VolumeChange{w}"] = vol.pct_change(w, fill_method=None)
    for w in (20, 50):
        out[f"DistHigh{w}"] = close / hi.rolling(w).max() - 1
        out[f"DistLow{w}"] = close / lo.rolling(w).min() - 1
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    rsi = rsi.mask((loss == 0) & (gain > 0), 100)
    rsi = rsi.mask((loss == 0) & (gain == 0), 50)
    out["RSI14Norm"] = (rsi - 50) / 50
    return out[CLASSIC].replace([np.inf, -np.inf], np.nan)


def build_level_clusters(prices, tolerance):
    """Même regroupement glouton que la V6, sur les seuls pivots disponibles."""
    clusters = []
    for price in sorted(prices):
        for cluster in clusters:
            center = cluster[0] / cluster[1]
            if abs(price / center - 1) <= tolerance:
                cluster[0] += price
                cluster[1] += 1
                break
        else:
            clusters.append([float(price), 1])
    return [{"price": total / n, "touches": n} for total, n in clusters]


def accept_event(kind, position, last_accepted, cooldown):
    """Premier signal conservé ; un rejet ne prolonge pas le cooldown."""
    previous = last_accepted.get(kind)
    if previous is not None and position - previous <= cooldown:
        return False
    last_accepted[kind] = position
    return True


def causal_structure(prices, cfg=FeatureConfig()):
    """Parcours strictement chronologique : confirmations, niveaux, cassures.

    Un pivot en p devient connu en p + right. Une cassure avant confirmation
    n'est jamais reportée dans le passé. Score géométrique sans volume ni rendement.
    """
    dates = prices.index
    high, low, close, vol = (prices[c].to_numpy(float) for c in ("High", "Low", "Close", "Volume"))
    vol_reference = prices["Volume"].shift(1).rolling(20).mean().to_numpy()
    out = pd.DataFrame(0.0, index=dates, columns=LEVELS + WM + SCORES + BREAKOUT_VOLUME)
    out["NearestSupport"] = np.nan
    out["NearestResistance"] = np.nan
    pivots, events, candidates = [], [], []
    last_accepted = {}
    level_rows = []
    signal_rows = []
    for i, date in enumerate(dates):
        p = i - cfg.pivot_right
        added = []
        if p >= cfg.pivot_left:
            left = p - cfg.pivot_left
            h, l = high[left:i + 1], low[left:i + 1]
            is_high = high[p] == h.max() and np.count_nonzero(h == high[p]) == 1
            is_low = low[p] == l.min() and np.count_nonzero(l == low[p]) == 1
            # Un outside bar peut être les deux : ordre intrabar inconnu, ignoré.
            if is_high != is_low:
                kind = "HIGH" if is_high else "LOW"
                pivot = dict(pivot_pos=p, confirm_pos=i, pivot_date=dates[p],
                             confirm_date=date, type=kind,
                             price=float(high[p] if is_high else low[p]))
                pivots.append(pivot)
                added.append(pivot)
        cutoff = date - pd.Timedelta(days=cfg.level_lookback_days)
        available = [x for x in pivots if x["pivot_date"] >= cutoff]
        supports = build_level_clusters([x["price"] for x in available if x["type"] == "LOW"], cfg.level_tolerance)
        resistances = build_level_clusters([x["price"] for x in available if x["type"] == "HIGH"], cfg.level_tolerance)
        support = max((x for x in supports if x["price"] < close[i]), key=lambda x: x["price"], default=None)
        resistance = min((x for x in resistances if x["price"] > close[i]), key=lambda x: x["price"], default=None)
        ds = close[i] / support["price"] - 1 if support else 0.2
        dr = resistance["price"] / close[i] - 1 if resistance else 0.2
        row = dict(DistSupport=ds, DistResistance=dr,
                   SupportTouches=support["touches"] if support else 0,
                   ResistanceTouches=resistance["touches"] if resistance else 0,
                   SupportMissing=float(support is None), ResistanceMissing=float(resistance is None),
                   LevelBalance=dr - ds if support and resistance else 0,
                   NearestSupport=support["price"] if support else np.nan,
                   NearestResistance=resistance["price"] if resistance else np.nan)
        for pct in (1, 2, 5):
            row[f"NearSupport{pct}Pct"] = int(support is not None and ds <= pct / 100)
            row[f"NearResistance{pct}Pct"] = int(resistance is not None and dr <= pct / 100)
        level_rows.append(row)
        for end in added:
            same = [x for x in pivots[:-1] if x["type"] == end["type"]
                    and cfg.pattern_min_span <= p - x["pivot_pos"] <= cfg.pattern_max_span]
            for start in same[-cfg.pattern_previous_pivots:]:
                middle = [x for x in pivots if x["type"] != end["type"]
                          and start["pivot_pos"] < x["pivot_pos"] < p]
                if not middle:
                    continue
                kind = "W" if end["type"] == "LOW" else "M"
                mid = (max if kind == "W" else min)(middle, key=lambda x: x["price"])
                a, b, neck = start["price"], end["price"], mid["price"]
                mismatch = abs(a - b) / ((a + b) / 2)
                height = (neck - max(a, b)) / neck if kind == "W" else (min(a, b) - neck) / neck
                if mismatch > cfg.pattern_tolerance or height < cfg.pattern_min_height:
                    continue
                balance = min(mid["pivot_pos"] - start["pivot_pos"], p - mid["pivot_pos"]) / max(mid["pivot_pos"] - start["pivot_pos"], p - mid["pivot_pos"])
                score = 100 * (0.5 * (1 - mismatch / cfg.pattern_tolerance)
                               + 0.25 * balance + 0.25 * min(height / 0.08, 1))
                candidates.append(dict(
                    pattern_id=f"{kind}_{start['pivot_pos']}_{mid['pivot_pos']}_{p}",
                    pattern=kind, start_pos=start["pivot_pos"], middle_pos=mid["pivot_pos"], end_pos=p,
                    start_date=start["pivot_date"], middle_date=mid["pivot_date"], end_date=end["pivot_date"],
                    formation_confirm_date=date, formation_pos=i, neckline=neck,
                    quality_score=score, level_center=(a + b) / 2,
                    invalidation=min(a, b) * (1 - cfg.pattern_tolerance) if kind == "W"
                    else max(a, b) * (1 + cfg.pattern_tolerance)))
        pending = []
        for candidate in candidates:
            kind, neck = candidate["pattern"], candidate["neckline"]
            invalid = close[i] < candidate["invalidation"] if kind == "W" else close[i] > candidate["invalidation"]
            if invalid or i - candidate["formation_pos"] > cfg.pattern_max_wait:
                continue
            crossed = i > 0 and (close[i - 1] <= neck < close[i] if kind == "W" else close[i - 1] >= neck > close[i])
            if not crossed:
                pending.append(candidate)
                continue
            accepted = accept_event(kind, i, last_accepted, cfg.pattern_cooldown)
            clusters = supports if kind == "W" else resistances
            near = [x for x in clusters if abs(x["price"] / candidate["level_center"] - 1) <= cfg.level_tolerance]
            level = min(near, key=lambda x: abs(x["price"] - candidate["level_center"]), default=None)
            vr = vol[i] / vol_reference[i] if np.isfinite(vol_reference[i]) and vol_reference[i] > 0 else 0.0
            event = {k: candidate[k] for k in EVENT_COLUMNS if k in candidate}
            event.update(breakout_pos=i, breakout_date=date, volume_ratio=float(vr),
                         near_level=int(level is not None), level_touches=level["touches"] if level else 0,
                         accepted=accepted, reason="first_causal_signal" if accepted else "cooldown_same_type")
            events.append(event)
            if accepted:
                signal_rows.append((i, kind, event))
        candidates = pending
    levels_frame = pd.DataFrame(level_rows, index=dates)
    out[levels_frame.columns] = levels_frame
    for i, kind, event in signal_rows:
        date = dates[i]
        for suffix, value in (("Signal", 1), ("Score", event["quality_score"]),
                              ("VolumeRatio", event["volume_ratio"]), ("NearLevel", event["near_level"]),
                              ("LevelTouches", event["level_touches"])):
            out.loc[date, kind + suffix] = value
    for kind in ("W", "M"):
        signal = out[kind + "Signal"]
        for window in (5, 10, 20, 30):
            out[f"{kind}Recent{window}"] = signal.rolling(window, min_periods=1).max()
        last_index = pd.Series(np.where(signal.to_numpy() == 1, np.arange(len(out)), np.nan), index=dates).ffill()
        age = (pd.Series(np.arange(len(out)), index=dates) - last_index).fillna(cfg.pattern_memory + 1)
        out[kind + "Recency"] = (1 - age / (cfg.pattern_memory + 1)).clip(0, 1)
        for suffix in ("Score", "VolumeRatio", "NearLevel", "LevelTouches"):
            carried = out[kind + suffix].where(signal == 1).ffill().fillna(0).where(age <= cfg.pattern_memory, 0)
            name = f"Last{kind}{suffix}" + ("Norm" if suffix == "Score" else "")
            out[name] = carried / 100 if suffix == "Score" else carried
    return out, pd.DataFrame(pivots, columns=PIVOT_COLUMNS), pd.DataFrame(events, columns=EVENT_COLUMNS)


def build_features(prices, cfg=FeatureConfig()):
    classic = classic_features(prices)
    structure, pivots, events = causal_structure(prices, cfg)
    return pd.concat([classic, structure], axis=1), pivots, events


def build_labels(prices, horizon=10, target_return=0.01):
    """Target V6 conservée ; les targets inconnues restent NaN, jamais classe 0."""
    out = pd.DataFrame(index=prices.index)
    out["FutureReturn"] = prices["Close"].shift(-horizon) / prices["Close"] - 1
    out["LabelEndDate"] = pd.Series(prices.index, index=prices.index).shift(-horizon)
    out["Target"] = (out["FutureReturn"] > target_return).astype(float).where(out["FutureReturn"].notna())
    return out
