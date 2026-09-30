from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
import os
from pathlib import Path

import pandas as pd

from .market import Snapshot, TZ, validate_prices
from .providers import _calendar_schedule, _parse_zoned


_CIRCUIT_BREAKER_HALT_STARTS = {
    "2020-03-09": ("2020-03-09T13:35:00Z", "2020-03-09T13:40:00Z"),
    "2020-03-12": ("2020-03-12T13:35:00Z", "2020-03-12T13:40:00Z"),
    "2020-03-16": ("2020-03-16T13:30:00Z", "2020-03-16T13:35:00Z", "2020-03-16T13:40:00Z"),
    "2020-03-18": ("2020-03-18T17:00:00Z", "2020-03-18T17:05:00Z", "2020-03-18T17:10:00Z"),
}


@dataclass
class ArchiveResult:
    symbol: str
    adjustment: str
    raw_bars: int = 0
    clean_bars: int = 0
    clean_sessions: int = 0
    excluded_sessions: int = 0
    status: str = "OK"
    output: str = ""


def exchange_schedule(start, end, calendar="NYSE") -> pd.DataFrame:
    return _calendar_schedule(start, end, calendar)


def _normalise_bars(bars: pd.DataFrame) -> pd.DataFrame:
    frame = bars.copy()
    if not isinstance(frame.index, pd.DatetimeIndex):
        if "Timestamp" not in frame.columns:
            raise ValueError("Les barres Alpaca doivent avoir un index Timestamp.")
        frame.index = _parse_zoned(frame.pop("Timestamp"), "Timestamp")
    elif frame.index.tz is None:
        raise ValueError("Les horodatages Alpaca doivent inclure un fuseau.")
    frame.index = frame.index.tz_convert("UTC")
    frame.index.name = "Timestamp"
    validate_prices(frame)
    return frame.loc[:, ["Open", "High", "Low", "Close", "Volume"]]


def clean_regular_sessions(bars: pd.DataFrame, schedule: pd.DataFrame, asof=None):
    frame = _normalise_bars(bars)
    schedule = schedule.copy()
    if not {"MarketOpen", "MarketClose"}.issubset(schedule.columns):
        raise ValueError("Le calendrier doit définir MarketOpen et MarketClose.")
    for column in ("MarketOpen", "MarketClose"):
        parsed = pd.to_datetime(schedule[column], errors="raise")
        if parsed.dt.tz is None:
            raise ValueError("Les heures du calendrier doivent inclure un fuseau.")
        schedule[column] = parsed.dt.tz_convert("UTC")
    cutoff = pd.Timestamp.now(tz="UTC") if asof is None else pd.Timestamp(asof)
    if cutoff.tzinfo is None:
        raise ValueError("asof doit inclure un fuseau horaire.")
    cutoff = cutoff.tz_convert("UTC")

    keep = []
    clean_schedule_rows = []
    audit_rows = []
    for session, sched in schedule.iterrows():
        opened, closed = sched.MarketOpen, sched.MarketClose
        grid = pd.date_range(opened, closed, freq="5min", inclusive="left")
        observed = frame.loc[(frame.index >= opened) & (frame.index < closed)]
        known_halt_times = grid.intersection(pd.DatetimeIndex(pd.to_datetime(
            _CIRCUIT_BREAKER_HALT_STARTS.get(str(session), ()), utc=True)))
        absent_halts = known_halt_times.difference(observed.index)
        expected = grid.difference(absent_halts)
        missing = expected.difference(observed.index)
        extras = int(((frame.index >= opened) & (frame.index < closed) & ~frame.index.isin(grid)).sum())
        row = {"Session": str(session), "expected": len(expected), "observed": len(observed),
               "missing": len(missing), "extra": extras, "halt_bars": 0}
        if closed > cutoff:
            row["status"] = "OPEN_OR_FUTURE"
        elif len(missing) == 0 and len(absent_halts) > 0:
            row["status"] = "MARKET_HALT_COMPLETE"
            row["halt_bars"] = len(absent_halts)
            keep.extend(observed.index.intersection(expected))
            clean_schedule_rows.append(session)
        elif len(missing) == 0 and len(observed) == len(grid):
            row["status"] = "COMPLETE" if len(grid) == 78 else "EARLY_CLOSE_COMPLETE"
            keep.extend(observed.index.intersection(expected))
            clean_schedule_rows.append(session)
        else:
            row["status"] = "MISSING_BARS"
        audit_rows.append(row)

    clean = frame.loc[frame.index.isin(pd.DatetimeIndex(keep))].sort_index().copy()
    clean_schedule = schedule.loc[schedule.index.isin(clean_schedule_rows)].copy()
    audit = pd.DataFrame(audit_rows, columns=["Session", "status", "expected", "observed", "missing", "extra", "halt_bars"])
    return clean, clean_schedule, audit


def _rth_session_bounds(frame: pd.DataFrame):
    if frame.empty:
        raise ValueError("L'archive Alpaca ne contient aucune barre.")
    local = frame.index.tz_convert(TZ)
    minutes = local.hour * 60 + local.minute
    in_rth = (minutes >= 9 * 60 + 30) & (minutes < 16 * 60)
    dates = local[in_rth].strftime("%Y-%m-%d")
    if len(dates) == 0:
        raise ValueError("Aucune barre des heures régulières pour borner le calendrier.")
    return min(dates), max(dates)


def _read_extended(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    stamp_col = "Timestamp" if "Timestamp" in raw.columns else raw.columns[0]
    raw.index = _parse_zoned(raw.pop(stamp_col), str(path))
    raw.index.name = "Timestamp"
    return _normalise_bars(raw)


def _write_archive(symbol, adjustment, extended, schedule, clean, clean_schedule, audit,
                   metadata, out_root) -> ArchiveResult:
    symbol_root = Path(out_root) / symbol
    adjustment_root = symbol_root / adjustment
    adjustment_root.mkdir(parents=True, exist_ok=True)
    extended = extended.copy()
    extended.index.name = "Timestamp"
    extended.to_csv(adjustment_root / f"{symbol}_extended_5m.csv", index_label="Timestamp", float_format="%.17g")
    clean.index.name = "Timestamp"
    clean.to_csv(adjustment_root / f"{symbol}_5m.csv", index_label="Timestamp", float_format="%.17g")
    calendar = clean_schedule.copy()
    calendar.insert(0, "Session", calendar.index.astype(str))
    calendar.to_csv(adjustment_root / "sessions.csv", index=False)
    (adjustment_root / f"{symbol}_5m.meta.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    audit_root = symbol_root / "audit"
    audit_root.mkdir(parents=True, exist_ok=True)
    audit_frame = audit.copy()
    audit_frame.insert(0, "adjustment", adjustment)
    audit_frame.to_csv(audit_root / "session_quality.csv", index=False)
    result = ArchiveResult(symbol, adjustment, len(extended), len(clean), len(clean_schedule),
                           int((audit.status == "MISSING_BARS").sum()), output=str(adjustment_root))
    (audit_root / "reclean_summary.json").write_text(
        json.dumps(result.__dict__, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def reclean_archive(symbol, source, out, adjustments=("raw", "split"), asof=None, calendar="NYSE"):
    symbol = str(symbol).upper()
    results = []
    for adjustment in adjustments:
        source_root = Path(source) / symbol / adjustment
        extended_path = source_root / f"{symbol}_extended_5m.csv"
        if not extended_path.is_file():
            raise FileNotFoundError(f"Archive étendue introuvable: {extended_path}")
        extended = _read_extended(extended_path)
        start, end = _rth_session_bounds(extended)
        schedule = exchange_schedule(start, end, calendar=calendar)
        clean, clean_schedule, audit = clean_regular_sessions(extended, schedule, asof=asof)
        metadata_path = source_root / f"{symbol}_5m.meta.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        metadata.update({"symbol": symbol, "source": metadata.get("source", "Alpaca Market Data API"),
                         "synthetic": False, "bar_minutes": 5, "timestamp_convention": "bar_start",
                         "currency": "USD", "adjustment": adjustment, "calendar": calendar})
        metadata.setdefault("fetched_at", (pd.Timestamp.now(tz="UTC") if asof is None else pd.Timestamp(asof)).isoformat())
        results.append(_write_archive(symbol, adjustment, extended, schedule, clean, clean_schedule,
                                      audit, metadata, out))
    return results


def _alpaca_bars(symbol, start, end, adjustment, feed):
    key = os.environ.get("APCA_API_KEY_ID")
    secret = os.environ.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise ValueError("Définir APCA_API_KEY_ID et APCA_API_SECRET_KEY dans l'environnement.")
    try:
        import requests
    except ImportError as exc:
        raise ImportError("Installer requests pour télécharger Alpaca.") from exc
    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    token = None
    chunks = []
    while True:
        params = {"timeframe": "5Min", "start": start, "end": end, "limit": 10000,
              "adjustment": adjustment, "feed": feed, "sort": "asc"}
        if token:
            params["page_token"] = token
        response = requests.get(url, headers=headers, params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()
        chunks.extend(payload.get("bars", []))
        token = payload.get("next_page_token")
        if not token:
            break
    if not chunks:
        raise ValueError(f"Alpaca n'a retourné aucune barre pour {symbol}/{adjustment}.")
    frame = pd.DataFrame(chunks).rename(columns={"t": "Timestamp", "o": "Open", "h": "High",
                                                "l": "Low", "c": "Close", "v": "Volume"})
    frame.index = _parse_zoned(frame.pop("Timestamp"), "Alpaca Timestamp")
    return _normalise_bars(frame)


def download_archive(symbol, out, start="2016-01-01", end=None, adjustments=("raw", "split"),
                     feed="sip", safety_minutes=20, calendar="NYSE"):
    symbol = str(symbol).upper()
    captured = pd.Timestamp.now(tz="UTC")
    safe_end = captured - pd.Timedelta(minutes=int(safety_minutes))
    requested_end = safe_end
    if end is not None:
        requested_end = pd.Timestamp(end)
        requested_end = (requested_end.tz_localize("UTC") if requested_end.tzinfo is None
                         else requested_end.tz_convert("UTC"))
        requested_end = min(requested_end, safe_end)
    results = []
    for adjustment in adjustments:
        extended = _alpaca_bars(symbol, str(start), requested_end.isoformat(), adjustment, feed)
        first, last = _rth_session_bounds(extended)
        schedule = exchange_schedule(first, last, calendar=calendar)
        clean, clean_schedule, audit = clean_regular_sessions(extended, schedule, asof=safe_end)
        metadata = {"symbol": symbol, "source": "Alpaca Market Data API", "synthetic": False,
                    "bar_minutes": 5, "timestamp_convention": "bar_start", "currency": "USD",
                    "adjustment": adjustment, "fetched_at": captured.isoformat(), "calendar": calendar,
                    "feed": feed, "requested_start": str(start), "effective_end": requested_end.isoformat()}
        results.append(_write_archive(symbol, adjustment, extended, schedule, clean, clean_schedule,
                                      audit, metadata, out))
    return results