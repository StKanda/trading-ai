from __future__ import annotations
from dataclasses import asdict
import pandas as pd
from ..data.market import completed_bars
from ..data.freshness import evaluate_freshness
from ..features.engine import FeatureEngine
from ..research.rules import rule_predictions, RULE_IDS
from ..risk import detailed_position_size


def decision_from_latest(snapshot, config, champion="RULE_MOMENTUM", timeframe=5, asof=None,
                         equity=None, risk_buffer_pct=0.10):
    freshness = evaluate_freshness(snapshot, asof=asof, timeframe=timeframe)
    if not freshness.usable_for_signal:
        return {"decision": "NO TRADE", "reason": freshness.status + " MARKET DATA",
                "freshness": freshness.to_dict(), "champion": champion, "symbol": snapshot.symbol}
    bars = completed_bars(snapshot, timeframe, asof=asof)
    features = FeatureEngine().build(bars)
    last = bars.index[-1:]
    if champion not in RULE_IDS:
        raise ValueError("V10 live worker smoke path accepts a validated RULE_* champion. ML/DL artifacts require explicit exported checkpoints.")
    pred = rule_predictions(bars, features.values, last)
    signal = float(pred.loc[last[0], champion])
    row = bars.loc[last[0]]
    atr = features.values.loc[last[0], "ATR14Pct"]
    if pd.isna(atr):
        return {"decision": "INSUFFICIENT CONFIDENCE", "reason": "ATR unavailable",
                "freshness": freshness.to_dict(), "champion": champion, "symbol": snapshot.symbol}
    equity = float(config.risk.initial_capital if equity is None else equity)
    stop_pct = max(config.risk.min_stop_pct, config.risk.stop_atr * float(atr))
    entry = float(row.Close) * (1 + config.risk.slippage_bps / 10000)
    stop = entry * (1 - stop_pct)
    sizing = detailed_position_size(equity, entry, stop, float(row.Volume), config.risk,
                                    risk_buffer_pct=risk_buffer_pct)
    decision = "LONG CANDIDATE" if signal >= config.research.threshold and sizing.quantity > 0 else "NO TRADE"
    reason = "rule signal + risk constraints" if decision == "LONG CANDIDATE" else (
        sizing.reason if sizing.quantity == 0 else "rule below threshold")
    return {
        "decision": decision, "reason": reason, "champion": champion, "symbol": snapshot.symbol,
        "market_timestamp": last[0].isoformat(), "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "price": float(row.Close), "entry_estimate": entry, "stop": stop,
        "target": entry * (1 + config.risk.reward_risk * stop_pct), "model_score": signal,
        "score_is_calibrated_probability": False, "freshness": freshness.to_dict(),
        "sizing": asdict(sizing), "data_provider": snapshot.metadata.get("source", "unknown")}
