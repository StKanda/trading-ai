"""Classement descriptif, figé avant toute preuve d'alpha.

Score 0–100 de liquidité approximée, activité relative et amplitude/coûts.
Ce n'est NI une probabilité de gain NI un système de sélection backtesté.
"""
import math
import numpy as np
from .data.market import expected_last_close


class StockScanner:
    def evaluate(self, snapshot, bars, features, asof, timeframe, risk):
        last = bars.iloc[-1]
        f = features.values.iloc[-1]
        session = bars.loc[bars.Session == last.Session]
        dollar_volume = float((session.Close * session.Volume).sum())
        rv = float(f.RelativeVolumeSlot) if np.isfinite(f.RelativeVolumeSlot) else None
        atr = float(f.ATR14Pct) if np.isfinite(f.ATR14Pct) else None
        expected = expected_last_close(snapshot.schedule, asof, timeframe)
        lag = max(0.0, (expected - bars.index[-1]).total_seconds() / 60) if expected is not None else 0
        missing = bars.attrs.get("missing_bars", 0)
        costs = 2 * (risk.fee_bps + risk.slippage_bps) / 10000
        # Poids fixés, sans estimation de rendement futur ni données fin de journée.
        liquidity = np.clip((math.log10(max(dollar_volume, 1)) - 6) / 3, 0, 1)
        activity = np.clip((rv or 0) / 2, 0, 1)
        movement = np.clip((atr or 0) / max(costs * 4, 1e-9), 0, 1)
        quality = 1.0 if lag == 0 and missing == 0 else 0.0
        score = round(100 * (0.35 * liquidity + 0.35 * activity + 0.20 * movement + 0.10 * quality))
        up = f.get("DistSMA20", np.nan) > 0 and f.get("SMA20Slope5", np.nan) > 0
        down = f.get("DistSMA20", np.nan) < 0 and f.get("SMA20Slope5", np.nan) < 0
        return dict(symbol=snapshot.symbol, price=float(last.Close),
                    session_change=float(last.Close / session.Open.iloc[0] - 1),
                    selection_score=score if quality else None, relative_volume=rv,
                    dollar_volume=dollar_volume, atr_pct=atr,
                    trend="Haussière" if up else "Baissière" if down else "Neutre",
                    data_lag_minutes=lag, missing_bars=missing,
                    source=snapshot.metadata["source"], synthetic=bool(snapshot.metadata.get("synthetic")),
                    last_bar=bars.index[-1].isoformat(),
                    status="Données à vérifier" if not quality else "Recherche possible",
                    confidence=None, confidence_reason="Probabilité de rentabilité non mesurée")
