from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..research.backtest import simulate

CLASS_TARGETS = ("ProfitableAfterCosts", "StopHit", "TargetHit", "TimeExit")
REG_TARGETS = ("RealizedPnL", "RealizedReturn", "MFE", "MAE")
ALL_TARGETS = CLASS_TARGETS + REG_TARGETS


def build_economic_labels(bars: pd.DataFrame, features: pd.DataFrame, research, risk,
                          candidate_mask: pd.Series | None = None, progress_callback=None) -> pd.DataFrame:
    """Labels économiques directs d'un trade isolé déclenché à la clôture t.

    Chaque candidat est rejoué avec le même simulateur que le backtest: signal à t,
    entrée à l'open t+1, stop prioritaire, target, time-exit, frais/slippage et no overnight.
    Ces labels regardent le futur et ne doivent servir qu'au train/validation avec purge.
    """
    if bars.empty:
        raise ValueError("Bars vides.")
    mask = pd.Series(True, index=bars.index) if candidate_mask is None else pd.Series(candidate_mask, index=bars.index).fillna(False).astype(bool)
    out = pd.DataFrame(index=bars.index)
    for c in CLASS_TARGETS + REG_TARGETS:
        out[c] = np.nan
    out["LabelEnd"] = pd.Series(pd.NaT, index=bars.index, dtype="datetime64[ns, UTC]")
    out["ExitReason"] = pd.Series(index=bars.index, dtype="object")
    minutes = int(bars.attrs.get("timeframe") or (bars.index[0] - bars.Start.iloc[0]).total_seconds() / 60)
    horizon_rows = int(research.horizon_minutes // minutes) + 2
    fee = risk.fee_bps / 10000.0
    slip = risk.slippage_bps / 10000.0

    candidates = bars.index[mask]
    for candidate_number, signal_time in enumerate(candidates, start=1):
        if progress_callback is not None:
            progress_callback({"completed": candidate_number, "total": len(candidates),
                               "detail": "candidats évalués"})
        session = bars.at[signal_time, "Session"]
        sb_all = bars.loc[bars.Session == session]
        try:
            pos = int(sb_all.index.get_loc(signal_time))
        except (KeyError, TypeError):
            continue
        sb = sb_all.iloc[pos:min(len(sb_all), pos + horizon_rows)].copy()
        if len(sb) < 2:
            continue
        score = pd.Series([1.0], index=pd.DatetimeIndex([signal_time]), name="ECON_LABEL")
        try:
            sim = simulate(sb, score, features, risk, research)
        except ValueError:
            continue
        if sim.trades.empty:
            continue
        tr = sim.trades.iloc[0]
        pnl = float(tr["pnl"])
        exit_reason = str(tr["exit_reason"])
        entry_time = pd.Timestamp(tr["entry_time"])
        exit_time = pd.Timestamp(tr["exit_time"])
        entry_cost = float(tr["entry_cost"])
        qty = max(float(tr["quantity"]), 1.0)
        entry_fill = float(tr["entry_price"])

        # Excursions jusqu'au moment de sortie. Pour une time-exit au Start d'une barre,
        # la barre de sortie n'est pas incluse, conformément au simulateur.
        w = sb_all.loc[(sb_all.Start >= entry_time) & (sb_all.Start < exit_time)]
        if w.empty:
            mfe = mae = 0.0
        else:
            denom = entry_fill * (1 + fee)
            best = float(w.High.max()) * (1 - slip) * (1 - fee) / denom - 1
            worst = float(w.Low.min()) * (1 - slip) * (1 - fee) / denom - 1
            mfe, mae = best, worst

        out.at[signal_time, "ProfitableAfterCosts"] = float(pnl > 0)
        out.at[signal_time, "StopHit"] = float(exit_reason in {"stop", "daily_loss", "daily_loss_gap"})
        out.at[signal_time, "TargetHit"] = float(exit_reason == "target")
        out.at[signal_time, "TimeExit"] = float(exit_reason in {"time_exit", "session_flat"})
        out.at[signal_time, "RealizedPnL"] = pnl
        out.at[signal_time, "RealizedReturn"] = pnl / float(risk.initial_capital)
        out.at[signal_time, "MFE"] = mfe
        out.at[signal_time, "MAE"] = mae
        out.at[signal_time, "LabelEnd"] = exit_time
        out.at[signal_time, "ExitReason"] = exit_reason
    if progress_callback is not None and len(candidates) == 0:
        progress_callback({"completed": 0, "total": 0})
    return out


@dataclass
class EconomicSequenceBundle:
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


def build_economic_sequence_bundle(features, labels, train_mask, valid_mask, context_bars=12, columns=None):
    columns = list(columns or features.columns)
    x = features[columns].astype(float).replace([np.inf, -np.inf], np.nan)
    train_positions = np.flatnonzero(np.asarray(train_mask, dtype=bool))
    valid_positions = np.flatnonzero(np.asarray(valid_mask, dtype=bool))
    fit_rows = x.iloc[train_positions].dropna()
    if fit_rows.empty:
        raise ValueError("No complete economic training rows for scaler.")
    med = fit_rows.median()
    scaler = StandardScaler().fit(fit_rows.to_numpy())
    scaled = pd.DataFrame(scaler.transform(x.fillna(med).to_numpy()), index=x.index, columns=columns)
    xtr, yctr, yrtr, itr = _make_sequences(scaled, labels, train_positions, context_bars)
    xva, ycva, yrva, iva = _make_sequences(scaled, labels, valid_positions, context_bars)
    if len(xtr) < 20:
        raise ValueError("Insufficient economic DL training sequences.")
    if len(xva) < 5:
        raise ValueError("Insufficient economic DL validation sequences.")
    reg_mean = yrtr.mean(axis=0).astype(np.float32)
    reg_std = yrtr.std(axis=0).astype(np.float32)
    reg_std = np.where(reg_std < 1e-6, 1.0, reg_std).astype(np.float32)
    return EconomicSequenceBundle(
        xtr, yctr, (yrtr-reg_mean)/reg_std,
        xva, ycva, (yrva-reg_mean)/reg_std,
        itr, iva, scaler, med, reg_mean, reg_std, columns, context_bars
    )


def transform_economic_sequences(features, positions, bundle: EconomicSequenceBundle):
    x = features[bundle.columns].astype(float).replace([np.inf, -np.inf], np.nan)
    scaled = pd.DataFrame(bundle.scaler.transform(x.fillna(bundle.fill_values).to_numpy()), index=x.index, columns=bundle.columns)
    return _make_x_sequences(scaled, np.flatnonzero(np.asarray(positions, dtype=bool)), bundle.context_bars)
