from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from .market import expected_last_close, utc


@dataclass(frozen=True)
class FreshnessPolicy:
    fresh_minutes: float = 5.0
    usable_minutes: float = 10.0
    stale_minutes: float = 20.0

    def __post_init__(self):
        if not (0 <= self.fresh_minutes <= self.usable_minutes <= self.stale_minutes):
            raise ValueError("Seuils de fraîcheur invalides.")


@dataclass(frozen=True)
class FreshnessResult:
    status: str
    usable_for_signal: bool
    lag_minutes: float | None
    expected_last_close: str | None
    observed_last_close: str | None
    timeframe: int

    def to_dict(self):
        return asdict(self)


def evaluate_freshness(snapshot, asof=None, timeframe=5, policy=None) -> FreshnessResult:
    policy = policy or FreshnessPolicy()
    now = utc(asof) if asof is not None else pd.Timestamp.now(tz="UTC")
    expected = expected_last_close(snapshot.schedule, now, timeframe)
    observed = None
    if not snapshot.prices.empty:
        observed = snapshot.prices.index[-1] + pd.Timedelta(minutes=5)
        observed = utc(observed)
    if expected is None or observed is None:
        return FreshnessResult("NO DATA", False, None,
                               expected.isoformat() if expected is not None else None,
                               observed.isoformat() if observed is not None else None, int(timeframe))
    captured = snapshot.metadata.get("fetched_at")
    if captured and utc(captured) < observed:
        observed = utc(captured)
    lag = max(0.0, (expected - observed).total_seconds() / 60.0)
    if lag <= policy.fresh_minutes:
        status, usable = "FRESH", True
    elif lag <= policy.usable_minutes:
        status, usable = "DELAYED", True
    else:
        status, usable = "STALE", False
    return FreshnessResult(status, usable, float(lag), expected.isoformat(), observed.isoformat(), int(timeframe))