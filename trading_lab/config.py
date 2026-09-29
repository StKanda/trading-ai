"""Paramètres explicites, fixés avant toute évaluation finale."""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


@dataclass(frozen=True)
class RiskConfig:
    initial_capital: float = 10_000.0
    max_leverage: float = 1.0
    risk_per_trade: float = 0.0025
    daily_loss_limit: float = 0.01
    fee_bps: float = 2.0
    slippage_bps: float = 3.0
    borrowing_bps_per_day: float = 2.0
    stop_atr: float = 1.5
    min_stop_pct: float = 0.0015
    reward_risk: float = 2.0
    max_volume_fraction: float = 0.005

    def __post_init__(self):
        if not (self.initial_capital > 0 and 1 <= self.max_leverage <= 3):
            raise ValueError("Capital positif ; levier simulé entre 1 et 3.")
        if not (0 < self.risk_per_trade <= self.daily_loss_limit < 0.20):
            raise ValueError("0 < risque par trade <= perte quotidienne < 20 %.")
        if not all(0 <= v < 100 for v in
                   (self.fee_bps, self.slippage_bps, self.borrowing_bps_per_day)):
            raise ValueError("Coûts en points de base, entre 0 et 100 exclus.")
        if not (self.stop_atr > 0 and 0 < self.min_stop_pct < 0.1
                and self.reward_risk > 0 and 0 < self.max_volume_fraction <= 0.05):
            raise ValueError("Paramètres de taille/stop invalides.")


@dataclass(frozen=True)
class ResearchConfig:
    horizon_minutes: int = 30
    threshold: float = 0.55
    train_sessions: int = 15
    fold_sessions: int = 5
    min_train_rows: int = 350
    extra_embargo_bars: int = 1
    rf_trees: int = 120
    gb_iterations: int = 100
    seed: int = 42
    jobs: int = 4
    min_validation_trades: int = 5
    min_validation_sessions_with_trades: int = 2
    max_validation_top1_abs_share: float = 0.60

    def __post_init__(self):
        if not (self.horizon_minutes > 0 and self.horizon_minutes % 15 == 0):
            raise ValueError("Horizon en multiples de 15 minutes.")
        if not (0 < self.threshold < 1 and self.train_sessions >= 10
                and self.fold_sessions >= 1 and self.min_train_rows >= 100
                and self.extra_embargo_bars >= 0 and self.rf_trees >= 10
                and self.gb_iterations >= 10 and 1 <= self.jobs <= 12
                and self.min_validation_trades >= 2
                and self.min_validation_sessions_with_trades >= 1
                and 0 < self.max_validation_top1_abs_share <= 1):
            raise ValueError("Configuration de recherche invalide.")


@dataclass(frozen=True)
class AppConfig:
    symbols: tuple = ("SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA")
    calendar: str = "NYSE"
    currency: str = "USD"
    risk: RiskConfig = field(default_factory=RiskConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)

    def __post_init__(self):
        import re
        if not self.symbols or len(self.symbols) > 20 or any(
                not re.fullmatch(r"[A-Z0-9.^=-]{1,15}", s) for s in self.symbols):
            raise ValueError("Univers : 1 à 20 symboles Yahoo explicites.")
        if self.calendar != "NYSE" or self.currency != "USD":
            raise ValueError("Cette version couvre les actions/ETF US en USD, heures régulières.")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path=None):
        if path is None:
            return cls()
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(symbols=tuple(raw.get("symbols", cls().symbols)),
                   calendar=raw.get("calendar", "NYSE"), currency=raw.get("currency", "USD"),
                   risk=RiskConfig(**raw.get("risk", {})),
                   research=ResearchConfig(**raw.get("research", {})))
