"""Simulation long/cash, un actif et une position, sans ordre réel.

Stop et objectif touchés dans la même bougie : stop prioritaire. Un gap défavorable
est exécuté à l'open et peut dépasser le budget. Dernière bougie de séance :
fermeture à son ouverture, donc aucune utilisation magique de sa clôture.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from ..risk import DailyRiskGuard

TRADE_COLUMNS = ["signal_time", "entry_time", "entry_price", "quantity", "entry_cost", "entry_fee",
                 "stop", "target", "planned_exit", "interest", "entry_equity", "notional_leverage",
                 "exit_time", "exit_price", "exit_reason", "intrabar_time_unknown", "pnl", "exit_fee"]


@dataclass
class Simulation:
    ledger: pd.DataFrame
    trades: pd.DataFrame


def simulate(bars, probabilities, features, config, research):
    if bars.empty:
        raise ValueError("Période de simulation vide.")
    fee, slip = config.fee_bps / 10000, config.slippage_bps / 10000
    minutes = int((bars.index[0] - bars.Start.iloc[0]).total_seconds() / 60)
    scores = probabilities.reindex(bars.index)
    known = scores.dropna()
    if not np.isfinite(known).all() or not known.between(0, 1).all():
        raise ValueError("Score de modèle hors de [0, 1].")
    atr_values = features.ATR14Pct.reindex(bars.index)
    guard, cash, position = DailyRiskGuard(config), config.initial_capital, None
    rows, trades = [], []
    previous_equity = cash
    peak = cash

    def equity_at(price):
        return cash + (position["quantity"] * price * (1 - slip) * (1 - fee) if position else 0)

    def close_position(price, when, reason, intrabar=False):
        nonlocal cash, position
        fill = price * (1 - slip)
        proceeds = position["quantity"] * fill * (1 - fee)
        cash += proceeds
        trades.append(dict(**position, exit_time=when, exit_price=fill, exit_reason=reason,
                           intrabar_time_unknown=intrabar,
                           pnl=proceeds - position["entry_cost"] - position["interest"],
                           exit_fee=position["quantity"] * fill * fee))
        position = None
        guard.observe(cash)

    for i, (end, bar) in enumerate(bars.iterrows()):
        changes, active, entry_flag = 0, position is not None, 0
        if bar.Session != guard.session:
            if position is not None:
                raise AssertionError("Une position a traversé la nuit : calendrier incomplet.")
            guard.begin(bar.Session, cash)
        flatten_at = bar.SessionClose - pd.Timedelta(minutes=minutes)
        if position is not None:
            if guard.observe(equity_at(bar.Open)):
                close_position(bar.Open, bar.Start, "daily_loss_gap")
                changes += 1
            elif bar.Start >= position["planned_exit"] or bar.Start >= flatten_at:
                close_position(bar.Open, bar.Start, "session_flat" if bar.Start >= flatten_at else "time_exit")
                changes += 1
        # Signal à End[i-1] ; prix d'exécution jamais connu à la fabrication du signal.
        can_enter = (position is None and not guard.halted and changes == 0 and i > 0
                     and bars.Session.iloc[i - 1] == bar.Session
                     and bars.index[i - 1] == bar.Start
                     and bar.Start + pd.Timedelta(minutes=research.horizon_minutes) <= flatten_at)
        if can_enter and pd.notna(scores.iloc[i - 1]) and scores.iloc[i - 1] >= research.threshold:
            atr = atr_values.iloc[i - 1]
            if pd.notna(atr):
                stop_pct = max(config.min_stop_pct, config.stop_atr * atr)
                if 0 < stop_pct < 0.1:
                    fill = bar.Open * (1 + slip)
                    stop = fill * (1 - stop_pct)
                    quantity = guard.quantity(cash, fill, stop, bars.Volume.iloc[i - 1])
                    if quantity:
                        entry_cost = quantity * fill * (1 + fee)
                        cash -= entry_cost
                        position = dict(signal_time=bars.index[i - 1], entry_time=bar.Start,
                                        entry_price=fill, quantity=quantity, entry_cost=entry_cost,
                                        entry_fee=quantity * fill * fee, stop=stop,
                                        target=fill * (1 + config.reward_risk * stop_pct),
                                        planned_exit=bar.Start + pd.Timedelta(minutes=research.horizon_minutes),
                                        interest=0.0, entry_equity=previous_equity,
                                        notional_leverage=quantity * fill / previous_equity)
                        active, entry_flag = True, 1
                        changes += 1
        if position is not None:
            # Financement simulé, proportionnel au temps et au cash débiteur.
            interest = max(-cash, 0) * config.borrowing_bps_per_day / 10000 * minutes / 390
            cash -= interest
            position["interest"] += interest
            guard_price = (guard.floor - cash) / (position["quantity"] * (1 - slip) * (1 - fee))
            barrier = max(position["stop"], guard_price)
            if bar.Low <= barrier:
                daily_hit = guard_price >= position["stop"]
                close_position(min(bar.Open, barrier), bar.Start if bar.Open <= barrier else end,
                               "daily_loss" if daily_hit else "stop", intrabar=bar.Open > barrier)
                if daily_hit:
                    guard.halted = True
                changes += 1
            elif bar.High >= position["target"]:
                close_position(position["target"], end, "target", intrabar=True)
                changes += 1
        capital = equity_at(bar.Close)
        guard.observe(capital)
        if capital <= 0:
            raise ValueError("Capital simulé non positif ; le levier a épuisé le compte.")
        peak = max(peak, capital)
        rows.append(dict(End=end, Session=bar.Session, Capital=capital,
                         NetReturn=capital / previous_equity - 1, Drawdown=capital / peak - 1,
                         Position=int(position is not None), Exposed=int(active), Entries=entry_flag,
                         PositionChanges=changes, Halted=int(guard.halted),
                         DailyPnL=capital - guard.opening_equity,
                         Signal=int(pd.notna(scores.iloc[i]) and scores.iloc[i] >= research.threshold),
                         ModelScore=scores.iloc[i]))
        previous_equity = capital
    if position is not None:
        raise ValueError("Fin de données avant fermeture planifiée. Fournir des séances complètes.")
    ledger = pd.DataFrame(rows).set_index("End")
    return Simulation(ledger, pd.DataFrame(trades, columns=TRADE_COLUMNS))


def reference_curve(bars, risk, intraday=False):
    """Benchmarks x1 : Buy & Hold prix seul et achat open/vente avant clôture.

    Pas de dividendes, spread inclus dans le slippage hypothétique. Même fenêtre
    d'évaluation que les modèles ; aucune limite de perte appliquée au benchmark.
    """
    fee, slip = risk.fee_bps / 10000, risk.slippage_bps / 10000
    cash, qty, previous, peak = risk.initial_capital, 0.0, risk.initial_capital, risk.initial_capital
    minutes = int((bars.index[0] - bars.Start.iloc[0]).total_seconds() / 60)
    rows = []
    for i, (end, bar) in enumerate(bars.iterrows()):
        changes, entry = 0, 0
        if i == 0 or (intraday and bar.Start == bar.SessionOpen):
            qty = cash / (bar.Open * (1 + slip) * (1 + fee))
            cash = 0.0
            changes += 1
            entry = 1
        exposed = int(qty > 0)
        if (intraday and bar.Start >= bar.SessionClose - pd.Timedelta(minutes=minutes)) or i == len(bars) - 1:
            cash += qty * bar.Open * (1 - slip) * (1 - fee)
            changes += int(qty > 0)
            qty = 0.0
        capital = cash + qty * bar.Close * (1 - slip) * (1 - fee)
        peak = max(peak, capital)
        rows.append(dict(End=end, Session=bar.Session, Capital=capital, NetReturn=capital / previous - 1,
                         Drawdown=capital / peak - 1, Position=int(qty > 0), Exposed=exposed,
                         Entries=entry, PositionChanges=changes, Halted=0, Signal=entry, ModelScore=np.nan))
        previous = capital
    return Simulation(pd.DataFrame(rows).set_index("End"), pd.DataFrame())
