from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

UTILITY_HORIZONS = (15, 30, 60)
CLASS_TARGETS = tuple(f"TargetBeforeStop_{h}" for h in UTILITY_HORIZONS)
REG_TARGETS = ("MFE_30", "MAE_30", "NetReturn_15", "NetReturn_30", "NetReturn_60")
ALL_TARGETS = CLASS_TARGETS + REG_TARGETS


def build_utility_labels(bars: pd.DataFrame, features: pd.DataFrame, research, risk,
                         horizons=UTILITY_HORIZONS) -> pd.DataFrame:
    if bars.empty:
        raise ValueError("Bars vides.")
    minutes = int(bars.attrs.get("timeframe") or (bars.index[0] - bars.Start.iloc[0]).total_seconds() / 60)
    horizons = tuple(int(h) for h in horizons)
    if any(h <= 0 or h % minutes for h in horizons):
        raise ValueError("Chaque horizon doit être un multiple positif du timeframe.")
    n = len(bars)
    fee = risk.fee_bps / 10000.0
    slip = risk.slippage_bps / 10000.0
    atr = features["ATR14Pct"].reindex(bars.index)
    out = pd.DataFrame(index=bars.index)
    for horizon in horizons:
        steps = horizon // minutes
        tbs = np.full(n, np.nan, dtype=float)
        net = np.full(n, np.nan, dtype=float)
        label_end = pd.Series(pd.NaT, index=bars.index, dtype="datetime64[ns, UTC]")
        for i in range(n):
            entry_i, exit_i = i + 1, i + steps + 1
            if exit_i >= n or entry_i >= n:
                continue
            if bars.Session.iloc[i] != bars.Session.iloc[entry_i] or bars.Session.iloc[i] != bars.Session.iloc[exit_i]:
                continue
            flatten_at = bars.SessionClose.iloc[entry_i] - pd.Timedelta(minutes=minutes)
            if bars.Start.iloc[entry_i] + pd.Timedelta(minutes=horizon) > flatten_at:
                continue
            atr_i = atr.iloc[i]
            if not np.isfinite(atr_i):
                continue
            stop_pct = max(float(risk.min_stop_pct), float(risk.stop_atr) * float(atr_i))
            if not (0 < stop_pct < 0.1):
                continue
            entry_fill = float(bars.Open.iloc[entry_i]) * (1 + slip)
            stop = entry_fill * (1 - stop_pct)
            target = entry_fill * (1 + float(risk.reward_risk) * stop_pct)
            hit_target_first = 0.0
            for j in range(entry_i, entry_i + steps):
                row = bars.iloc[j]
                if float(row.Low) <= stop:  # priorité pessimiste identique au backtester
                    hit_target_first = 0.0
                    break
                if float(row.High) >= target:
                    hit_target_first = 1.0
                    break
            tbs[i] = hit_target_first
            exit_fill = float(bars.Open.iloc[exit_i]) * (1 - slip)
            net[i] = exit_fill * (1 - fee) / (entry_fill * (1 + fee)) - 1
            label_end.iloc[i] = bars.Start.iloc[exit_i]
        out[f"TargetBeforeStop_{horizon}"] = tbs
        out[f"NetReturn_{horizon}"] = net
        out[f"LabelEnd_{horizon}"] = label_end

    steps = 30 // minutes
    mfe = np.full(n, np.nan, dtype=float)
    mae = np.full(n, np.nan, dtype=float)
    for i in range(n):
        entry_i, exit_i = i + 1, i + steps + 1
        if exit_i >= n or bars.Session.iloc[i] != bars.Session.iloc[exit_i] or pd.isna(out.iloc[i]["LabelEnd_30"]):
            continue
        entry_fill = float(bars.Open.iloc[entry_i]) * (1 + slip)
        window = bars.iloc[entry_i:entry_i + steps]
        best_exit = float(window.High.max()) * (1 - slip)
        worst_exit = float(window.Low.min()) * (1 - slip)
        denom = entry_fill * (1 + fee)
        mfe[i] = best_exit * (1 - fee) / denom - 1
        mae[i] = worst_exit * (1 - fee) / denom - 1
    out["MFE_30"] = mfe
    out["MAE_30"] = mae
    return out


@dataclass
class UtilitySequenceBundle:
    x_train: np.ndarray
    y_class_train: np.ndarray
    y_reg_train: np.ndarray
    x_valid: np.ndarray
    y_class_valid: np.ndarray
    y_reg_valid: np.ndarray
    train_index: pd.DatetimeIndex
    valid_index: pd.DatetimeIndex
    scaler: StandardScaler
    fill_values: pd.Series
    reg_mean: np.ndarray
    reg_std: np.ndarray
    columns: list[str]
    context_bars: int
    class_names: tuple[str, ...] = CLASS_TARGETS
    reg_names: tuple[str, ...] = REG_TARGETS


def _make_sequences(frame, labels, positions, context):
    xs, yc, yr, idx = [], [], [], []
    values = frame.to_numpy(dtype=np.float32)
    cls = labels.loc[:, CLASS_TARGETS].to_numpy(dtype=float)
    reg = labels.loc[:, REG_TARGETS].to_numpy(dtype=float)
    for pos in positions:
        if pos < context - 1 or pos >= len(frame):
            continue
        block = values[pos-context+1:pos+1]
        if np.isfinite(block).all() and np.isfinite(cls[pos]).all() and np.isfinite(reg[pos]).all():
            xs.append(block); yc.append(cls[pos]); yr.append(reg[pos]); idx.append(frame.index[pos])
    if not xs:
        return (np.empty((0, context, frame.shape[1]), dtype=np.float32),
                np.empty((0, len(CLASS_TARGETS)), dtype=np.float32),
                np.empty((0, len(REG_TARGETS)), dtype=np.float32), pd.DatetimeIndex([]))
    return np.stack(xs), np.asarray(yc, dtype=np.float32), np.asarray(yr, dtype=np.float32), pd.DatetimeIndex(idx)


def _make_x_sequences(frame, positions, context):
    xs, idx = [], []
    values = frame.to_numpy(dtype=np.float32)
    for pos in positions:
        if pos < context - 1 or pos >= len(frame):
            continue
        block = values[pos-context+1:pos+1]
        if np.isfinite(block).all():
            xs.append(block); idx.append(frame.index[pos])
    if not xs:
        return np.empty((0, context, frame.shape[1]), dtype=np.float32), pd.DatetimeIndex([])
    return np.stack(xs), pd.DatetimeIndex(idx)


def build_utility_sequence_bundle(features, labels, train_mask, valid_mask, context_bars=12, columns=None):
    if context_bars < 2:
        raise ValueError("context_bars must be >= 2")
    columns = list(columns or features.columns)
    x = features[columns].astype(float).replace([np.inf, -np.inf], np.nan)
    train_positions = np.flatnonzero(np.asarray(train_mask, dtype=bool))
    valid_positions = np.flatnonzero(np.asarray(valid_mask, dtype=bool))
    fit_rows = x.iloc[train_positions].dropna()
    if fit_rows.empty:
        raise ValueError("No complete training rows for scaler.")
    med = fit_rows.median()
    scaler = StandardScaler().fit(fit_rows.to_numpy())
    scaled = pd.DataFrame(scaler.transform(x.fillna(med).to_numpy()), index=x.index, columns=columns)
    xtr, yctr, yrtr, itr = _make_sequences(scaled, labels, train_positions, context_bars)
    xva, ycva, yrva, iva = _make_sequences(scaled, labels, valid_positions, context_bars)
    if len(xtr) < 20:
        raise ValueError("Insufficient utility DL training sequences.")
    if len(xva) < 5:
        raise ValueError("Insufficient utility DL validation sequences.")
    reg_mean = yrtr.mean(axis=0).astype(np.float32)
    reg_std = yrtr.std(axis=0).astype(np.float32)
    reg_std = np.where(reg_std < 1e-6, 1.0, reg_std).astype(np.float32)
    yrtr_n = (yrtr - reg_mean) / reg_std
    yrva_n = (yrva - reg_mean) / reg_std
    return UtilitySequenceBundle(xtr, yctr, yrtr_n, xva, ycva, yrva_n, itr, iva,
                                 scaler, med, reg_mean, reg_std, columns, context_bars)


def transform_utility_sequences(features, positions, bundle: UtilitySequenceBundle):
    """Transformer des lignes futures sans utiliser leurs labels ni refitter le scaler."""
    x = features[bundle.columns].astype(float).replace([np.inf, -np.inf], np.nan)
    scaled = pd.DataFrame(bundle.scaler.transform(x.fillna(bundle.fill_values).to_numpy()),
                          index=x.index, columns=bundle.columns)
    return _make_x_sequences(scaled, np.flatnonzero(np.asarray(positions, dtype=bool)), bundle.context_bars)
