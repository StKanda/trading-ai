from __future__ import annotations

from dataclasses import dataclass
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


TZ = ZoneInfo("America/New_York")
_BASE_MINUTES = 5
_SUPPORTED_TIMEFRAMES = (5, 10, 15, 30)
_PRICE_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


def utc(value) -> pd.Timestamp:
    """Convert an explicitly zoned timestamp to UTC; reject ambiguous local times."""
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError("Horodatage sans fuseau horaire.")
    return stamp.tz_convert("UTC")


@dataclass
class Snapshot:
    symbol: str
    prices: pd.DataFrame
    schedule: pd.DataFrame
    metadata: dict


def validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(prices, pd.DataFrame):
        raise ValueError("Les prix doivent être un DataFrame OHLCV.")
    missing = [column for column in _PRICE_COLUMNS if column not in prices.columns]
    if missing:
        raise ValueError(f"Colonnes OHLCV manquantes: {', '.join(missing)}")
    if not isinstance(prices.index, pd.DatetimeIndex) or prices.index.tz is None:
        raise ValueError("L'index des prix doit contenir des horodatages avec fuseau.")
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError("Les horodatages doivent être uniques et croissants.")
    values = prices.loc[:, _PRICE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    array = values.to_numpy(dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("Les prix OHLCV doivent être numériques et finis.")
    if (values[["Open", "High", "Low", "Close"]] <= 0).any().any() or (values.Volume < 0).any():
        raise ValueError("Prix strictement positifs et volume positif ou nul requis.")
    invalid = ((values.High < values[["Open", "Low", "Close"]].max(axis=1)) |
               (values.Low > values[["Open", "High", "Close"]].min(axis=1)))
    if invalid.any():
        raise ValueError("Bougie OHLC incohérente.")
    return prices


def _schedule_frame(schedule: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(schedule, pd.DataFrame) or not {"MarketOpen", "MarketClose"}.issubset(schedule.columns):
        raise ValueError("Le calendrier doit définir MarketOpen et MarketClose.")
    result = schedule.copy()
    if not result.index.is_unique or not result.index.is_monotonic_increasing:
        raise ValueError("Les séances du calendrier doivent être uniques et triées.")
    for column in ("MarketOpen", "MarketClose"):
        parsed = pd.to_datetime(result[column], errors="raise")
        if getattr(parsed.dt, "tz", None) is None:
            raise ValueError("Les heures du calendrier doivent inclure un fuseau.")
        result[column] = parsed.dt.tz_convert("UTC")
    if (result.MarketClose <= result.MarketOpen).any():
        raise ValueError("La fermeture doit suivre l'ouverture de chaque séance.")
    return result


def _quality_halts(snapshot: Snapshot, session: str) -> pd.DatetimeIndex:
    statuses = snapshot.metadata.get("_session_quality_statuses", {})
    status = statuses.get(str(session), {})
    if status.get("status") != "MARKET_HALT_COMPLETE":
        return pd.DatetimeIndex([], tz="UTC")
    known = {
        "2020-03-09": ("2020-03-09T13:35:00Z", "2020-03-09T13:40:00Z"),
        "2020-03-12": ("2020-03-12T13:35:00Z", "2020-03-12T13:40:00Z"),
        "2020-03-16": ("2020-03-16T13:30:00Z", "2020-03-16T13:35:00Z", "2020-03-16T13:40:00Z"),
        "2020-03-18": ("2020-03-18T17:00:00Z", "2020-03-18T17:05:00Z", "2020-03-18T17:10:00Z"),
    }
    return pd.DatetimeIndex(pd.to_datetime(known.get(str(session), ()), utc=True))


def completed_bars(snapshot: Snapshot, timeframe: int = 5, asof=None,
                   require_complete: bool = False) -> pd.DataFrame:
    timeframe = int(timeframe)
    if timeframe not in _SUPPORTED_TIMEFRAMES:
        raise ValueError(f"Intervalle non supporté: {timeframe} minutes.")
    validate_prices(snapshot.prices)
    schedule = _schedule_frame(snapshot.schedule)
    requested_asof = utc(asof) if asof is not None else pd.Timestamp.now(tz="UTC")
    captured = snapshot.metadata.get("fetched_at")
    effective_asof = min(requested_asof, utc(captured)) if captured else requested_asof
    prices = snapshot.prices.copy()
    prices.index = prices.index.tz_convert("UTC")
    prices = prices.loc[:, _PRICE_COLUMNS]
    rows = []
    missing_count = 0
    partial_capture = False

    for session, sched in schedule.iterrows():
        market_open, market_close = sched.MarketOpen, sched.MarketClose
        if market_open > requested_asof:
            continue
        full_grid = pd.date_range(market_open, market_close, freq="5min", inclusive="left")
        halts = _quality_halts(snapshot, str(session))
        expected_grid = full_grid.difference(halts)
        in_session = prices.loc[(prices.index >= market_open) & (prices.index < market_close)]
        observed = in_session.index
        if market_close <= requested_asof and effective_asof < market_close:
            partial_capture = True

        completed_expected = expected_grid[expected_grid + pd.Timedelta(minutes=5) <= effective_asof]
        missing = completed_expected.difference(observed)
        missing_count += len(missing)
        if require_complete and market_close <= requested_asof:
            all_missing = expected_grid.difference(observed)
            if len(all_missing) or effective_asof < market_close:
                raise ValueError(f"Séance {session} incomplète: {len(all_missing)} bougie(s) manquante(s).")

        session_delta = pd.Timedelta(minutes=timeframe)
        bucket_starts = full_grid[:: timeframe // _BASE_MINUTES]
        for start in bucket_starts:
            end = min(start + session_delta, market_close)
            if end > effective_asof:
                continue
            components = expected_grid[(expected_grid >= start) & (expected_grid < end)]
            if components.empty:
                continue
            data = in_session.reindex(components)
            if data.isna().any().any():
                continue
            row = {
                "Start": start,
                "Session": str(session),
                "SessionOpen": market_open,
                "SessionClose": market_close,
                "Open": float(data.Open.iloc[0]),
                "High": float(data.High.max()),
                "Low": float(data.Low.min()),
                "Close": float(data.Close.iloc[-1]),
                "Volume": float(data.Volume.sum()),
            }
            rows.append((end, row))

    columns = ["Start", "Session", "SessionOpen", "SessionClose", *_PRICE_COLUMNS]
    if rows:
        result = pd.DataFrame([row for _, row in rows], index=pd.DatetimeIndex([end for end, _ in rows]))
        result.index.name = "Timestamp"
        result = result.loc[:, columns]
    else:
        result = pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], tz="UTC", name="Timestamp"))
    result.attrs.update({
        "timeframe": timeframe,
        "base_timeframe": _BASE_MINUTES,
        "missing_bars": int(missing_count),
        "partial_capture": bool(partial_capture),
    })
    if require_complete and partial_capture:
        raise ValueError("La dernière séance est incomplète au moment de capture.")
    return result


def expected_last_close(schedule: pd.DataFrame, asof, timeframe: int = 5):
    timeframe = int(timeframe)
    if timeframe not in _SUPPORTED_TIMEFRAMES:
        raise ValueError(f"Intervalle non supporté: {timeframe} minutes.")
    schedule = _schedule_frame(schedule)
    now = utc(asof)
    active = schedule.loc[(schedule.MarketOpen <= now) & (schedule.MarketClose >= now)]
    if not active.empty:
        row = active.iloc[-1]
    else:
        ended = schedule.loc[schedule.MarketClose < now]
        if ended.empty:
            return None
        row = ended.iloc[-1]
    opened, closed = row.MarketOpen, row.MarketClose
    if now >= closed:
        return closed
    elapsed = now - opened
    finished = (elapsed // pd.Timedelta(minutes=timeframe)) * pd.Timedelta(minutes=timeframe)
    return min(opened + finished, closed)