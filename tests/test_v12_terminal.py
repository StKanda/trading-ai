import json
from pathlib import Path
import pandas as pd

from trading_lab.data.alpaca import clean_regular_sessions
from trading_lab.data.market import Snapshot
from trading_lab.data.providers import CSVProvider, write_snapshot


def _bars_for_schedule(schedule):
    rows=[]
    px=100.0
    for _, r in schedule.iterrows():
        for t in pd.date_range(r.MarketOpen, r.MarketClose, freq="5min", inclusive="left"):
            rows.append((t, px, px+1, px-1, px+.2, 1000))
            px += .01
    return pd.DataFrame(rows, columns=["Timestamp","Open","High","Low","Close","Volume"]).set_index("Timestamp")


def test_v12_calendar_cleaner_supports_early_close_and_excludes_hole():
    schedule = pd.DataFrame({
        "MarketOpen": [pd.Timestamp("2024-11-27 14:30Z"), pd.Timestamp("2024-11-29 14:30Z")],
        "MarketClose": [pd.Timestamp("2024-11-27 21:00Z"), pd.Timestamp("2024-11-29 18:00Z")],
    }, index=pd.Index(["2024-11-27","2024-11-29"], name="Session"))
    bars = _bars_for_schedule(schedule)
    assert len(bars.loc[(bars.index >= schedule.iloc[1].MarketOpen) & (bars.index < schedule.iloc[1].MarketClose)]) == 42
    # Remove one bar only from the normal session.
    bars = bars.drop(pd.Timestamp("2024-11-27 16:00Z"))
    clean, clean_schedule, audit = clean_regular_sessions(bars, schedule, asof="2024-12-01T00:00:00Z")
    assert list(clean_schedule.index) == ["2024-11-29"]
    assert len(clean) == 42
    assert audit.set_index("Session").loc["2024-11-27", "status"] == "MISSING_BARS"
    assert audit.set_index("Session").loc["2024-11-29", "status"] == "EARLY_CLOSE_COMPLETE"


def test_v12_csv_provider_accepts_split_adjusted_snapshot(tmp_path):
    schedule = pd.DataFrame({
        "MarketOpen": [pd.Timestamp("2024-01-02 14:30Z")],
        "MarketClose": [pd.Timestamp("2024-01-02 21:00Z")],
    }, index=pd.Index(["2024-01-02"], name="Session"))
    bars = _bars_for_schedule(schedule)
    snap = Snapshot("NVDA", bars, schedule, {
        "symbol":"NVDA", "source":"test", "synthetic":False, "bar_minutes":5,
        "timestamp_convention":"bar_start", "currency":"USD", "adjustment":"split",
        "fetched_at":"2024-01-03T00:00:00Z"
    })
    write_snapshot(snap, tmp_path)
    loaded = CSVProvider(tmp_path).fetch("NVDA")
    assert loaded.metadata["adjustment"] == "split"
    assert len(loaded.prices) == 78


def test_v12_1_market_wide_halt_is_not_treated_as_data_hole():
    schedule = pd.DataFrame({
        "MarketOpen": [pd.Timestamp("2020-03-09 13:30Z")],
        "MarketClose": [pd.Timestamp("2020-03-09 20:00Z")],
    }, index=pd.Index(["2020-03-09"], name="Session"))
    bars = _bars_for_schedule(schedule)
    # Official circuit-breaker gap observed in Alpaca SIP: the 13:35 and 13:40
    # bar starts do not exist because the market was halted.
    bars = bars.drop([pd.Timestamp("2020-03-09 13:35Z"), pd.Timestamp("2020-03-09 13:40Z")])
    clean, clean_schedule, audit = clean_regular_sessions(bars, schedule, asof="2020-03-10T00:00:00Z")
    row = audit.set_index("Session").loc["2020-03-09"]
    assert row["status"] == "MARKET_HALT_COMPLETE"
    assert row["expected"] == 76
    assert row["observed"] == 76
    assert row["missing"] == 0
    assert row["halt_bars"] == 2
    assert len(clean) == 76
    assert list(clean_schedule.index) == ["2020-03-09"]


def test_v1211_reclean_local_archive_no_network(tmp_path, monkeypatch):
    import json
    import pandas as pd
    import trading_lab.data.alpaca as alpaca_mod
    from trading_lab.data.alpaca import reclean_archive

    source = tmp_path / "old"
    out = tmp_path / "new"
    sym = "NVDA"
    adj = "split"
    d = source / sym / adj
    d.mkdir(parents=True)
    # Known 2020-03-09 halt: omit the two halted 5m starts; this should still be accepted.
    op = pd.Timestamp("2020-03-09T13:30:00Z")
    cl = pd.Timestamp("2020-03-09T20:00:00Z")
    grid = pd.date_range(op, cl, freq="5min", inclusive="left")
    halt = pd.DatetimeIndex(pd.to_datetime(["2020-03-09T13:35:00Z", "2020-03-09T13:40:00Z"], utc=True))
    idx = grid.difference(halt)
    frame = pd.DataFrame({"Open":1.0,"High":1.1,"Low":0.9,"Close":1.0,"Volume":100.0}, index=idx)
    frame.index.name = "Timestamp"
    frame.to_csv(d / f"{sym}_extended_5m.csv")
    schedule = pd.DataFrame({
        "MarketOpen": [op], "MarketClose": [cl]
    }, index=pd.Index(["2020-03-09"], name="Session"))
    monkeypatch.setattr(alpaca_mod, "exchange_schedule", lambda *a, **k: schedule)
    (d / f"{sym}_5m.meta.json").write_text(json.dumps({
        "symbol":sym,"source":"Alpaca Market Data API","synthetic":False,"bar_minutes":5,
        "timestamp_convention":"bar_start","currency":"USD","adjustment":adj,
        "fetched_at":"2020-03-10T00:00:00+00:00","calendar":"NYSE"
    }))

    results = reclean_archive(sym, source, out, adjustments=(adj,), asof="2020-03-10T00:00:00Z")
    assert results[0].clean_sessions == 1
    audit = pd.read_csv(out / sym / "audit" / "session_quality.csv")
    assert audit.loc[0, "status"] == "MARKET_HALT_COMPLETE"
    assert (out / sym / adj / f"{sym}_extended_5m.csv").exists()
    assert (out / sym / adj / f"{sym}_5m.csv").exists()
    assert (out / sym / "audit" / "reclean_summary.json").exists()


def test_v1212_reclean_ignores_previous_local_date_from_utc_boundary(tmp_path, monkeypatch):
    import json
    import pandas as pd
    import trading_lab.data.alpaca as alpaca_mod
    from trading_lab.data.alpaca import reclean_archive

    source = tmp_path / "old"
    out = tmp_path / "new"
    sym = "SPY"
    adj = "split"
    d = source / sym / adj
    d.mkdir(parents=True)

    # Extended-hours bar at 2016-01-01 00:00Z belongs to 2015-12-31 in New York,
    # but the first actual RTH session in this archive is 2016-01-04.
    ext = pd.Timestamp("2016-01-01T00:00:00Z")
    op = pd.Timestamp("2016-01-04T14:30:00Z")
    cl = pd.Timestamp("2016-01-04T21:00:00Z")
    idx = pd.DatetimeIndex([ext]).append(pd.date_range(op, cl, freq="5min", inclusive="left"))
    frame = pd.DataFrame({"Open":1.0,"High":1.1,"Low":0.9,"Close":1.0,"Volume":100.0}, index=idx)
    frame.index.name = "Timestamp"
    frame.to_csv(d / f"{sym}_extended_5m.csv")

    calls = []
    def fake_schedule(start, end, calendar="NYSE"):
        calls.append((str(start), str(end)))
        return pd.DataFrame({"MarketOpen":[op], "MarketClose":[cl]}, index=pd.Index(["2016-01-04"], name="Session"))
    monkeypatch.setattr(alpaca_mod, "exchange_schedule", fake_schedule)

    (d / f"{sym}_5m.meta.json").write_text(json.dumps({
        "symbol":sym,"source":"Alpaca Market Data API","synthetic":False,"bar_minutes":5,
        "timestamp_convention":"bar_start","currency":"USD","adjustment":adj,
        "fetched_at":"2016-01-05T00:00:00+00:00","calendar":"NYSE"
    }))

    results = reclean_archive(sym, source, out, adjustments=(adj,), asof="2016-01-05T00:00:00Z")
    assert results[0].clean_sessions == 1
    assert calls == [("2016-01-04", "2016-01-04")]
    audit = pd.read_csv(out / sym / "audit" / "session_quality.csv")
    assert "2015-12-31" not in set(audit["Session"].astype(str))
    assert audit.loc[0, "status"] == "COMPLETE"


def test_v1213_filter_date_range_is_inclusive():
    from trading_lab.terminal_pipeline import _filter_date_range
    idx = pd.date_range("2025-01-02 14:30Z", periods=6, freq="5min")
    bars = pd.DataFrame({
        "Session": ["2025-01-02"] * 2 + ["2025-06-03"] * 2 + ["2025-12-31"] * 2,
        "Open": 1.0, "High": 1.1, "Low": 0.9, "Close": 1.0, "Volume": 100.0,
    }, index=idx)
    got = _filter_date_range(bars, "2025-06-03", "2025-12-31")
    assert list(pd.unique(got["Session"])) == ["2025-06-03", "2025-12-31"]
    assert len(got) == 4


def test_v1213_filter_date_range_rejects_bad_order():
    from trading_lab.terminal_pipeline import _filter_date_range
    bars = pd.DataFrame({"Session": ["2025-01-02"]}, index=[pd.Timestamp("2025-01-02T14:30Z")])
    import pytest
    with pytest.raises(ValueError, match="antérieure ou égale"):
        _filter_date_range(bars, "2025-12-31", "2025-01-01")


def test_v1214_csv_provider_loads_market_halt_audit(tmp_path):
    import json
    import pandas as pd
    from trading_lab.data.providers import CSVProvider, write_snapshot
    from trading_lab.data.market import Snapshot

    sym = "NVDA"
    root = tmp_path / sym
    split = root / "split"
    audit_dir = root / "audit"
    split.mkdir(parents=True)
    audit_dir.mkdir(parents=True)
    op = pd.Timestamp("2020-03-09T13:30:00Z")
    cl = pd.Timestamp("2020-03-09T20:00:00Z")
    grid = pd.date_range(op, cl, freq="5min", inclusive="left")
    missing = pd.DatetimeIndex(pd.to_datetime(["2020-03-09T13:35:00Z","2020-03-09T13:40:00Z"], utc=True))
    idx = grid.difference(missing)
    bars = pd.DataFrame({"Open":1.0,"High":1.1,"Low":0.9,"Close":1.0,"Volume":100.0}, index=idx)
    bars.index.name = "Timestamp"
    schedule = pd.DataFrame({"MarketOpen":[op],"MarketClose":[cl]}, index=pd.Index(["2020-03-09"], name="Session"))
    snap = Snapshot(sym, bars, schedule, {
        "symbol":sym,"source":"Alpaca Market Data API","synthetic":False,"bar_minutes":5,
        "timestamp_convention":"bar_start","currency":"USD","adjustment":"split",
        "fetched_at":"2020-03-10T00:00:00Z"
    })
    write_snapshot(snap, split)
    pd.DataFrame([{
        "adjustment":"split","Session":"2020-03-09","status":"MARKET_HALT_COMPLETE",
        "expected":76,"observed":76,"missing":0,"extra":0,"halt_bars":2
    }]).to_csv(audit_dir / "session_quality.csv", index=False)
    loaded = CSVProvider(split).fetch(sym)
    assert loaded.metadata["_session_quality_statuses"]["2020-03-09"]["status"] == "MARKET_HALT_COMPLETE"
    assert loaded.metadata["_session_quality_statuses"]["2020-03-09"]["halt_bars"] == 2


def test_v1214_completed_bars_accepts_certified_market_halt():
    from trading_lab.data.market import Snapshot, completed_bars
    op = pd.Timestamp("2020-03-09T13:30:00Z")
    cl = pd.Timestamp("2020-03-09T20:00:00Z")
    grid = pd.date_range(op, cl, freq="5min", inclusive="left")
    missing = pd.DatetimeIndex(pd.to_datetime(["2020-03-09T13:35:00Z","2020-03-09T13:40:00Z"], utc=True))
    idx = grid.difference(missing)
    bars = pd.DataFrame({"Open":1.0,"High":1.1,"Low":0.9,"Close":1.0,"Volume":100.0}, index=idx)
    schedule = pd.DataFrame({"MarketOpen":[op],"MarketClose":[cl]}, index=pd.Index(["2020-03-09"], name="Session"))
    snap = Snapshot("NVDA", bars, schedule, {
        "symbol":"NVDA","source":"Alpaca Market Data API","synthetic":False,"bar_minutes":5,
        "timestamp_convention":"bar_start","currency":"USD","adjustment":"split",
        "fetched_at":"2020-03-10T00:00:00Z",
        "_session_quality_statuses":{"2020-03-09":{"status":"MARKET_HALT_COMPLETE","halt_bars":2}},
    })
    got = completed_bars(snap, 5, asof=cl, require_complete=True)
    assert len(got) == 76


def test_v1214_slice_snapshot_before_validation_ignores_old_out_of_range_halt():
    from trading_lab.data.market import Snapshot, completed_bars
    from trading_lab.terminal_pipeline import _slice_snapshot_date_range
    # Old incomplete session plus complete 2025 session. A 2025 request must not inspect 2020.
    op0, cl0 = pd.Timestamp("2020-03-09T13:30Z"), pd.Timestamp("2020-03-09T20:00Z")
    op1, cl1 = pd.Timestamp("2025-01-02T14:30Z"), pd.Timestamp("2025-01-02T21:00Z")
    idx0 = pd.date_range(op0, cl0, freq="5min", inclusive="left").delete([1,2])
    idx1 = pd.date_range(op1, cl1, freq="5min", inclusive="left")
    idx = idx0.append(idx1)
    bars = pd.DataFrame({"Open":1.0,"High":1.1,"Low":0.9,"Close":1.0,"Volume":100.0}, index=idx)
    schedule = pd.DataFrame({"MarketOpen":[op0,op1],"MarketClose":[cl0,cl1]}, index=pd.Index(["2020-03-09","2025-01-02"], name="Session"))
    snap = Snapshot("NVDA", bars, schedule, {
        "symbol":"NVDA","source":"local","synthetic":False,"bar_minutes":5,
        "timestamp_convention":"bar_start","currency":"USD","adjustment":"split","fetched_at":"2025-01-03T00:00:00Z"
    })
    sliced = _slice_snapshot_date_range(snap, "2025-01-01", "2025-12-31")
    got = completed_bars(sliced, 5, asof=sliced.schedule.MarketClose.max(), require_complete=True)
    assert len(got) == 78
    assert set(got["Session"].astype(str)) == {"2025-01-02"}
