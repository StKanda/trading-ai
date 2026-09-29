"""Gestion de risque pure, indépendante de l'interface et du modèle."""
from dataclasses import dataclass
import math
from .config import RiskConfig


@dataclass
class DailyRiskGuard:
    config: RiskConfig
    session: str = ""
    opening_equity: float = 0.0
    halted: bool = False

    def begin(self, session, equity):
        if session != self.session:
            self.session, self.opening_equity = session, equity
            self.halted = equity <= 0

    @property
    def floor(self):
        return self.opening_equity * (1 - self.config.daily_loss_limit)

    def observe(self, liquidation_equity):
        if liquidation_equity <= self.floor + 1e-8:
            self.halted = True
        return self.halted

    def quantity(self, equity, entry_fill, stop_price, previous_volume):
        if self.halted or equity <= self.floor or previous_volume <= 0:
            return 0
        cfg = self.config
        fee, slip = cfg.fee_bps / 10000, cfg.slippage_bps / 10000
        stop_fill = stop_price * (1 - slip)
        loss_per_share = entry_fill * (1 + fee) - stop_fill * (1 - fee)
        # L'intérêt intraday est réservé pour une séance entière : majoration prudente.
        loss_per_share += entry_fill * cfg.borrowing_bps_per_day / 10000
        risk_budget = min(equity * cfg.risk_per_trade, max(0, equity - self.floor))
        notional_limit = equity * cfg.max_leverage / (entry_fill * (1 + fee))
        volume_limit = previous_volume * cfg.max_volume_fraction
        return max(0, math.floor(min(risk_budget / max(loss_per_share, 1e-9), notional_limit, volume_limit)))


def trade_plan(price, atr_pct, equity, previous_volume, config):
    """Scénario indicatif à recalculer au prochain open, jamais ordre ferme."""
    if not all(math.isfinite(float(v)) for v in (price, atr_pct, equity, previous_volume)):
        return None
    stop_pct = max(config.min_stop_pct, config.stop_atr * atr_pct)
    if not 0 < stop_pct < 0.1:
        return None
    fill = price * (1 + config.slippage_bps / 10000)
    stop = fill * (1 - stop_pct)
    guard = DailyRiskGuard(config)
    guard.begin("scenario", equity)
    qty = guard.quantity(equity, fill, stop, previous_volume)
    return dict(reference=price, entry_estimate=fill, stop=stop,
                target=fill * (1 + config.reward_risk * stop_pct), shares=qty,
                notional=qty * fill, leverage=qty * fill / equity,
                risk_budget=equity * config.risk_per_trade,
                daily_budget=equity * config.daily_loss_limit,
                reward_risk=config.reward_risk,
                note="Scénario achat, taille estimée ; recalcul obligatoire au prochain prix d'exécution.")


@dataclass(frozen=True)
class PositionSizingResult:
    quantity: int
    equity: float
    risk_budget: float
    risk_per_unit: float
    quantity_by_risk: int
    quantity_by_leverage: int
    quantity_by_volume: int
    notional: float
    planned_risk: float
    planned_risk_pct: float
    daily_risk_remaining: float
    status: str
    reason: str


def detailed_position_size(equity, entry_fill, stop_price, previous_volume, config,
                           opening_equity=None, realized_daily_pnl=0.0, risk_buffer_pct=0.10,
                           max_position_notional_pct=1.0):
    """Transparent paper sizing used by V10 UI/tests.

    It mirrors DailyRiskGuard conservatively and exposes each cap. Risk is planned,
    not guaranteed: gaps can make realized losses larger.
    """
    values = (equity, entry_fill, stop_price, previous_volume, risk_buffer_pct,
              max_position_notional_pct, realized_daily_pnl)
    if not all(math.isfinite(float(v)) for v in values):
        raise ValueError("Position sizing inputs must be finite.")
    if equity <= 0 or entry_fill <= 0 or stop_price <= 0 or previous_volume < 0:
        raise ValueError("Position sizing inputs invalid.")
    opening_equity = equity if opening_equity is None else float(opening_equity)
    fee, slip = config.fee_bps / 10000, config.slippage_bps / 10000
    stop_fill = stop_price * (1 - slip)
    base_loss = entry_fill * (1 + fee) - stop_fill * (1 - fee)
    base_loss += entry_fill * config.borrowing_bps_per_day / 10000
    risk_per_unit = max(base_loss * (1 + max(0.0, risk_buffer_pct)), 1e-9)
    nominal_trade_budget = equity * config.risk_per_trade
    daily_budget = opening_equity * config.daily_loss_limit
    daily_used = max(0.0, -float(realized_daily_pnl))
    daily_remaining = max(0.0, daily_budget - daily_used)
    risk_budget = min(nominal_trade_budget, daily_remaining)
    q_risk = max(0, math.floor(risk_budget / risk_per_unit))
    max_notional = equity * min(config.max_leverage, max_position_notional_pct)
    q_lev = max(0, math.floor(max_notional / (entry_fill * (1 + fee))))
    q_vol = max(0, math.floor(previous_volume * config.max_volume_fraction))
    quantity = min(q_risk, q_lev, q_vol)
    if daily_remaining <= 0:
        status, reason = "NO_TRADE", "DAILY RISK LIMIT"
        quantity = 0
    elif quantity < 1:
        status, reason = "NO_TRADE", "POSITION TOO SMALL FOR RISK LIMIT"
        quantity = 0
    else:
        status, reason = "OK", "Risk, leverage and liquidity constraints satisfied"
    notional = quantity * entry_fill
    planned = quantity * risk_per_unit
    return PositionSizingResult(
        quantity=quantity, equity=float(equity), risk_budget=float(risk_budget),
        risk_per_unit=float(risk_per_unit), quantity_by_risk=q_risk,
        quantity_by_leverage=q_lev, quantity_by_volume=q_vol, notional=float(notional),
        planned_risk=float(planned), planned_risk_pct=float(planned / equity if equity else 0),
        daily_risk_remaining=float(daily_remaining), status=status, reason=reason)
