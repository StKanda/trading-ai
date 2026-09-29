from __future__ import annotations
import numpy as np
import pandas as pd


def pnl_concentration(trades):
    if trades is None or len(trades) == 0 or "pnl" not in trades:
        return {"trades": 0, "total_pnl": 0.0, "without_best": 0.0,
                "without_top3": 0.0, "without_top5pct": 0.0, "highly_concentrated": True}
    pnl = pd.Series(trades["pnl"], dtype=float).dropna().sort_values(ascending=False)
    total = float(pnl.sum())
    top_n = max(1, int(np.ceil(len(pnl) * 0.05)))
    without_best = float(pnl.iloc[1:].sum()) if len(pnl) > 1 else 0.0
    without_top3 = float(pnl.iloc[min(3, len(pnl)):].sum())
    without_top5 = float(pnl.iloc[top_n:].sum())
    highly = bool(total > 0 and (without_best <= 0 or without_top5 <= 0))
    return {"trades": int(len(pnl)), "total_pnl": total, "without_best": without_best,
            "without_top3": without_top3, "without_top5pct": without_top5,
            "highly_concentrated": highly}


def neighborhood_stability(values, selected_key=None):
    """values is mapping parameter->objective. Stable if neighbors do not collapse."""
    if len(values) < 3:
        return {"stable": False, "reason": "Too few neighboring configurations"}
    keys = sorted(values)
    if selected_key is None:
        selected_key = max(values, key=values.get)
    if selected_key not in values:
        raise ValueError("Selected parameter absent from surface.")
    i = keys.index(selected_key)
    neighbors = [keys[j] for j in (i-1, i+1) if 0 <= j < len(keys)]
    if not neighbors:
        return {"stable": False, "reason": "No parameter neighbors"}
    center = float(values[selected_key])
    nv = [float(values[k]) for k in neighbors]
    tolerance = max(abs(center) * 0.35, 1e-9)
    stable = all(v >= center - tolerance for v in nv)
    return {"stable": bool(stable), "selected": selected_key, "selected_value": center,
            "neighbor_values": dict(zip(neighbors, nv)),
            "reason": "plateau" if stable else "FRAGILE PARAMETER REGION"}


def robustness_gate(metrics, concentration=None, positive_folds=None, total_folds=None,
                    stressed_return=None, min_trades=20):
    reasons = []
    trades = int(metrics.get("trades", 0))
    expectancy = float(metrics.get("expectancy", metrics.get("avg_trade", 0.0)) or 0.0)
    max_dd = float(metrics.get("max_drawdown", 0.0) or 0.0)
    if trades < min_trades:
        reasons.append("too_few_trades")
    if expectancy <= 0:
        reasons.append("non_positive_expectancy")
    if max_dd < -0.20:
        reasons.append("drawdown_too_large")
    if concentration and concentration.get("highly_concentrated"):
        reasons.append("pnl_highly_concentrated")
    if stressed_return is not None and stressed_return <= 0:
        reasons.append("fails_cost_stress")
    if total_folds and positive_folds is not None and positive_folds / total_folds < 0.5:
        reasons.append("insufficient_positive_folds")
    return {"passes": not reasons, "reason": ",".join(reasons) if reasons else "passes robustness gate",
            "reasons": reasons}
