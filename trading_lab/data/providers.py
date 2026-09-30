from __future__ import annotations

from abc import ABC, abstractmethod
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from .market import Snapshot, TZ, _schedule_frame, validate_prices


class DataProvider(ABC):
    @abstractmethod
    def fetch(self, symbol: str) -> Snapshot:
        raise NotImplementedError


def _parse_zoned(values, label: str) -> pd.DatetimeIndex:
    stamps = [pd.Timestamp(value) for value in values]
    if any(stamp.tzinfo is None for stamp in stamps):
        raise ValueError(f"{label}: un fuseau horaire explicite est requis.")
    return pd.DatetimeIndex(stamps).tz_convert("UTC")


def _calendar_schedule(start, end, calendar="NYSE") -> pd.DataFrame:
    try:
        import pandas_market_calendars as mcal
    except ImportError as exc:
        raise ImportError("Installer pandas_market_calendars via requirements_market.txt.") from exc
    schedule = mcal.get_calendar(calendar).schedule(start_date=str(start), end_date=str(end))
    if schedule.empty:
        raise ValueError(f"Aucune séance {calendar} entre {start} et {end}.")
    schedule = schedule.rename(columns={"market_open": "MarketOpen", "market_close": "MarketClose"})
    schedule.index = pd.Index(schedule.index.strftime("%Y-%m-%d"), name="Session")
    return _schedule_frame(schedule[["MarketOpen", "MarketClose"]])


def _read_schedule(path: Path, prices: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    file = path / "sessions.csv"
    if file.exists():
        raw = pd.read_csv(file, dtype={"Session": str})
        if not {"Session", "MarketOpen", "MarketClose"}.issubset(raw.columns):
            raise ValueError(f"Calendrier invalide: {file}")
        schedule = raw.set_index("Session")[["MarketOpen", "MarketClose"]]
        schedule.index = pd.Index(schedule.index.astype(str), name="Session")
        for column in ("MarketOpen", "MarketClose"):
            schedule[column] = _parse_zoned(schedule[column], f"{file}:{column}")
        return _schedule_frame(schedule)
    if prices.empty:
        raise ValueError("Un calendrier sessions.csv est requis pour un snapshot vide.")
    local = prices.index.tz_convert(TZ)
    return _calendar_schedule(local.min().date(), local.max().date(), metadata.get("calendar", "NYSE"))


class CSVProvider(DataProvider):
    def __init__(self, directory):
        self.directory = Path(directory)

    def _symbol_directory(self, symbol: str) -> Path:
        direct = self.directory / f"{symbol}_5m.csv"
        if direct.exists():
            return self.directory
        nested = self.directory / symbol
        if (nested / f"{symbol}_5m.csv").exists():
            return nested
        return self.directory

    def fetch(self, symbol: str) -> Snapshot:
        symbol = str(symbol).upper()
        directory = self._symbol_directory(symbol)
        price_path = directory / f"{symbol}_5m.csv"
        meta_path = directory / f"{symbol}_5m.meta.json"
        if not price_path.is_file():
            raise FileNotFoundError(f"Snapshot CSV introuvable: {price_path}")
        if not meta_path.is_file():
            raise FileNotFoundError(f"Métadonnées du snapshot introuvables: {meta_path}")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        raw = pd.read_csv(price_path)
        timestamp_column = "Timestamp" if "Timestamp" in raw.columns else raw.columns[0]
        index = _parse_zoned(raw.pop(timestamp_column), f"{price_path}:Timestamp")
        raw.index = index
        raw.index.name = "Timestamp"
        validate_prices(raw)
        schedule = _read_schedule(directory, raw, metadata)

        audit_path = directory.parent / "audit" / "session_quality.csv"
        if not audit_path.exists():
            audit_path = directory / "audit" / "session_quality.csv"
        if audit_path.exists():
            audit = pd.read_csv(audit_path)
            if {"Session", "status"}.issubset(audit.columns):
                metadata["_session_quality_statuses"] = {
                    str(row.Session): row.dropna().to_dict() for _, row in audit.iterrows()
                    if str(row.get("adjustment", metadata.get("adjustment", ""))) == str(metadata.get("adjustment", ""))
                }
        return Snapshot(symbol, raw, schedule, metadata)


class DemoProvider(CSVProvider):
    """Charge les fixtures synthétiques explicitement identifiées du dépôt."""
    def __init__(self, directory=None):
        root = Path(__file__).resolve().parents[2]
        super().__init__(directory or root / "verification" / "data_demo")


class YahooProvider(DataProvider):
    def fetch(self, symbol: str) -> Snapshot:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError("Installer yfinance via requirements_market.txt.") from exc
        symbol = str(symbol).upper()
        fetched_at = pd.Timestamp.now(tz="UTC")
        frame = yf.download(symbol, period="60d", interval="5m", auto_adjust=False,
                            prepost=False, progress=False, threads=False)
        if frame is None or frame.empty:
            raise ValueError(f"Yahoo n'a retourné aucune bougie pour {symbol}.")
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        frame = frame.rename(columns={"Adj Close": "AdjClose"})
        frame = frame.rename(columns={column: column.title() for column in frame.columns})
        frame = frame.loc[:, ["Open", "High", "Low", "Close", "Volume"]].copy()
        frame.index = pd.DatetimeIndex(frame.index)
        if frame.index.tz is None:
            raise ValueError("Yahoo a retourné des horodatages sans fuseau.")
        frame.index = frame.index.tz_convert("UTC")
        frame = frame.sort_index()
        validate_prices(frame)
        local = frame.index.tz_convert(TZ)
        schedule = _calendar_schedule(local.min().date(), local.max().date())
        # Ne garder que les bar-starts de séances régulières réellement observés.
        keep = pd.Series(False, index=frame.index)
        for row in schedule.itertuples():
            keep |= (frame.index >= row.MarketOpen) & (frame.index < row.MarketClose)
        frame = frame.loc[keep.to_numpy()]
        metadata = {
            "symbol": symbol, "source": "Yahoo Finance / yfinance", "synthetic": False,
            "bar_minutes": 5, "timestamp_convention": "bar_start", "currency": "USD",
            "adjustment": "raw", "fetched_at": fetched_at.isoformat(), "calendar": "NYSE",
        }
        return Snapshot(symbol, frame, schedule, metadata)


def write_snapshot(snapshot: Snapshot, directory) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    validate_prices(snapshot.prices)
    prices = snapshot.prices.copy()
    prices.index = prices.index.tz_convert("UTC")
    prices.index.name = "Timestamp"
    prices.to_csv(directory / f"{snapshot.symbol}_5m.csv", index_label="Timestamp", float_format="%.17g")
    schedule = _schedule_frame(snapshot.schedule).copy()
    schedule.insert(0, "Session", schedule.index.astype(str))
    schedule.to_csv(directory / "sessions.csv", index=False)
    metadata = dict(snapshot.metadata)
    metadata.setdefault("symbol", snapshot.symbol)
    (directory / f"{snapshot.symbol}_5m.meta.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    return directory


def fingerprint(snapshot: Snapshot, before=None) -> str:
    prices = snapshot.prices.loc[:, ["Open", "High", "Low", "Close", "Volume"]].copy()
    prices = prices.apply(pd.to_numeric, errors="raise").astype("float64")
    prices.index = prices.index.tz_convert("UTC")
    schedule = _schedule_frame(snapshot.schedule)
    if before is not None:
        cutoff = str(before)[:10]
        local_days = prices.index.tz_convert(TZ).strftime("%Y-%m-%d")
        prices = prices.loc[local_days < cutoff]
        schedule = schedule.loc[schedule.index.astype(str) < cutoff]
    digest = sha256()
    digest.update(str(snapshot.symbol).encode())
    digest.update(pd.util.hash_pandas_object(prices, index=True).values.tobytes())
    digest.update(pd.util.hash_pandas_object(schedule, index=True).values.tobytes())
    stable_metadata = {key: value for key, value in snapshot.metadata.items()
                       if key != "_session_quality_statuses"}
    digest.update(json.dumps(stable_metadata, sort_keys=True, default=str).encode())
    return digest.hexdigest()