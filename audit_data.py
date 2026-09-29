"""Audit local d'un snapshot V9. N'accède à aucun courtier."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from trading_lab.data.providers import CSVProvider
from trading_lab.data.market import TZ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--symbol', default='SPY')
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    snap = CSVProvider(a.data).fetch(a.symbol)
    p, s = snap.prices, snap.schedule
    local_days = pd.Index(p.index.tz_convert(TZ).strftime('%Y-%m-%d'))
    price_sessions = sorted(set(local_days))
    details, holes = [], []
    for day in price_sessions:
        if day not in s.index:
            holes.append({'session': day, 'error': 'prix sans calendrier'})
            continue
        row = s.loc[day]
        expected = pd.date_range(row.MarketOpen, row.MarketClose, freq='5min', inclusive='left')
        actual = p.index[local_days == day]
        missing = expected.difference(actual)
        extra = actual.difference(expected)
        details.append({'session': day, 'bars': len(actual), 'expected_bars': len(expected),
                        'missing': len(missing), 'extra': len(extra)})
        holes.extend({'session': day, 'missing_timestamp': x.isoformat()} for x in missing)
        holes.extend({'session': day, 'extra_timestamp': x.isoformat()} for x in extra)
    ohlc_ok = bool((p.High >= p[['Open','Close','Low']].max(axis=1)).all() and
                   (p.Low <= p[['Open','Close','High']].min(axis=1)).all())
    report = {
        'symbol': a.symbol, 'rows': len(p), 'price_sessions': len(price_sessions),
        'first_timestamp_utc': p.index[0].isoformat(), 'last_timestamp_utc': p.index[-1].isoformat(),
        'first_session': price_sessions[0], 'last_session': price_sessions[-1],
        'timezone_source': 'timestamps explicites, normalisés UTC; calendrier NYSE',
        'duplicates': int(p.index.duplicated().sum()), 'missing_values': int(p.isna().sum().sum()),
        'finite_ohlcv': bool(np.isfinite(p.to_numpy()).all()), 'ohlc_consistent': ohlc_ok,
        'negative_volume_rows': int((p.Volume < 0).sum()),
        'grid_issues': holes, 'sessions_detail': details,
        'calendar_sessions_without_prices': [d for d in s.index if d not in price_sessions],
        'metadata': snap.metadata,
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if a.out:
        a.out.write_text(text, encoding='utf-8')
    print(text)
    if report['duplicates'] or report['missing_values'] or not report['finite_ohlcv'] or not ohlc_ok or holes:
        raise SystemExit(2)

if __name__ == '__main__':
    main()
