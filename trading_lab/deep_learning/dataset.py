from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class SequenceBundle:
    x_train: np.ndarray
    y_train: np.ndarray
    x_valid: np.ndarray
    y_valid: np.ndarray
    train_index: pd.DatetimeIndex
    valid_index: pd.DatetimeIndex
    scaler: StandardScaler
    columns: list[str]
    context_bars: int


def _sequences(frame, target, positions, context):
    xs, ys, idx = [], [], []
    values = frame.to_numpy(dtype=np.float32)
    y = target.to_numpy(dtype=float)
    for pos in positions:
        if pos < context - 1 or pos >= len(frame) or not np.isfinite(y[pos]):
            continue
        block = values[pos-context+1:pos+1]
        if np.isfinite(block).all():
            xs.append(block)
            ys.append(int(y[pos]))
            idx.append(frame.index[pos])
    if not xs:
        return np.empty((0, context, frame.shape[1]), dtype=np.float32), np.empty(0, dtype=np.int64), pd.DatetimeIndex([])
    return np.stack(xs), np.asarray(ys, dtype=np.int64), pd.DatetimeIndex(idx)


def build_sequence_bundle(features, labels, train_mask, valid_mask, context_bars=24, columns=None):
    if context_bars < 2:
        raise ValueError("context_bars must be >= 2")
    columns = list(columns or features.columns)
    x = features[columns].astype(float).replace([np.inf, -np.inf], np.nan)
    y = labels["Target"] if isinstance(labels, pd.DataFrame) else labels
    train_positions = np.flatnonzero(np.asarray(train_mask, dtype=bool))
    valid_positions = np.flatnonzero(np.asarray(valid_mask, dtype=bool))
    # Fit scaler strictly on training rows only.
    fit_rows = x.iloc[train_positions].dropna()
    if fit_rows.empty:
        raise ValueError("No complete training rows for scaler.")
    scaler = StandardScaler().fit(fit_rows.to_numpy())
    scaled = pd.DataFrame(scaler.transform(x.fillna(fit_rows.median()).to_numpy()), index=x.index, columns=columns)
    xtr, ytr, itr = _sequences(scaled, y, train_positions, context_bars)
    xva, yva, iva = _sequences(scaled, y, valid_positions, context_bars)
    if len(xtr) < 20 or len(np.unique(ytr)) < 2:
        raise ValueError("Insufficient DL training sequences or classes.")
    if len(xva) < 5:
        raise ValueError("Insufficient DL validation sequences.")
    return SequenceBundle(xtr, ytr, xva, yva, itr, iva, scaler, columns, context_bars)
