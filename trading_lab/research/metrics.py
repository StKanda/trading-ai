"""Mesures V9 : performance quotidienne + statistiques de trades."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, roc_auc_score,
                             precision_score, recall_score, brier_score_loss)


def performance(simulation, initial_capital):
    ledger = simulation.ledger
    daily_capital = ledger.groupby("Session").Capital.last()
    daily_returns = daily_capital.pct_change()
    daily_returns.iloc[0] = daily_capital.iloc[0] / initial_capital - 1
    std = daily_returns.std(ddof=1)
    trades = simulation.trades
    if len(trades):
        pnl = trades.pnl.astype(float)
        wins, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
        trade_stats = dict(trades=int(len(trades)), win_rate=float((pnl > 0).mean()),
                           mean_trade_pnl=float(pnl.mean()), median_trade_pnl=float(pnl.median()),
                           profit_factor=float(wins / losses) if losses > 0 else (np.inf if wins > 0 else np.nan))
    else:
        trade_stats = dict(trades=0, win_rate=np.nan, mean_trade_pnl=np.nan,
                           median_trade_pnl=np.nan, profit_factor=np.nan)
    return dict(total_return=float(ledger.Capital.iloc[-1] / initial_capital - 1),
                final_capital=float(ledger.Capital.iloc[-1]), max_drawdown=float(ledger.Drawdown.min()),
                sharpe=float(np.sqrt(252) * daily_returns.mean() / std) if std > 0 else np.nan,
                exposure=float(ledger.Exposed.mean()), signals=int(ledger.Signal.sum()),
                position_changes=int(ledger.PositionChanges.sum()), entries=int(ledger.Entries.sum()),
                sessions=len(daily_capital), daily_halts=int(ledger.groupby("Session").Halted.max().sum()),
                mean_daily_return=float(daily_returns.mean()), **trade_stats)


def classification(labels, probabilities, threshold):
    joined = pd.concat([labels.Target.rename("y"), probabilities.rename("p")], axis=1).dropna()
    if joined.empty:
        return dict(labeled_rows=0)
    y, p = joined.y.astype(int), joined.p
    guessed = p >= threshold
    two = y.nunique() == 2
    return dict(labeled_rows=len(y), positive_rate=float(y.mean()),
                accuracy=float(accuracy_score(y, guessed)),
                balanced_accuracy=float(balanced_accuracy_score(y, guessed)) if two else np.nan,
                roc_auc=float(roc_auc_score(y, p)) if two else np.nan,
                precision=float(precision_score(y, guessed, zero_division=0)),
                recall=float(recall_score(y, guessed, zero_division=0)),
                brier_score=float(brier_score_loss(y, p)),
                always_up_accuracy=float(y.mean()))


def ablations(metrics):
    pairs = [("A", "B", "supports_resistances"), ("B", "C", "W_M"),
             ("C", "C_SCORE", "scores"), ("C", "C_VOLUME", "breakout_volume"),
             ("C", "D", "scores_and_volume")]
    indexed = metrics.set_index("model")
    rows = []
    for family in ("RF", "GB"):
        for low, high, component in pairs:
            a, b = indexed.loc[f"{family}_{low}"], indexed.loc[f"{family}_{high}"]
            rows.append(dict(family=family, component=component, base=low, augmented=high,
                             delta_auc=b.roc_auc - a.roc_auc,
                             delta_capital=b.final_capital - a.final_capital,
                             delta_sharpe=b.sharpe - a.sharpe))
    return pd.DataFrame(rows)
