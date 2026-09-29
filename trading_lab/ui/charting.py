from __future__ import annotations

from pathlib import Path
import json
import math
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..data.market import TZ
from ..research.rules import RULE_IDS, rule_predictions


def discover_local_symbols(root: str | Path) -> list[str]:
    """Découvre les symboles locaux sans prendre les jeux de démonstration/validation."""
    root = Path(root).expanduser()
    if not root.exists():
        return []
    excluded = {".venv", "verification", "data_demo", "__pycache__"}
    symbols: set[str] = set()
    for path in root.rglob("*_5m.csv"):
        if any(part in excluded for part in path.parts):
            continue
        name = path.name[:-7]  # retire _5m.csv
        meta = path.with_name(f"{name}_5m.meta.json")
        if meta.exists():
            symbols.add(name.upper())
    return sorted(symbols)


def resolve_symbol_directory(root: str | Path, symbol: str) -> Path:
    root = Path(root).expanduser()
    symbol = symbol.upper().strip()
    direct = root / f"{symbol}_5m.csv"
    if direct.exists():
        return root
    excluded = {".venv", "verification", "data_demo", "__pycache__"}
    candidates = [p.parent for p in root.rglob(f"{symbol}_5m.csv")
                  if not any(part in excluded for part in p.parts)] if root.exists() else []
    candidates = [p for p in candidates if (p / f"{symbol}_5m.meta.json").exists()]
    if not candidates:
        raise FileNotFoundError(f"Aucune donnée locale complète pour {symbol} sous {root}.")
    candidates.sort(key=lambda p: (len(p.parts), str(p)))
    return candidates[0]


def _to_ny(series_or_index):
    idx = pd.DatetimeIndex(pd.to_datetime(series_or_index, utc=True))
    return idx.tz_convert(TZ)


def _safe_float(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except Exception:
        return None



def discover_trade_models(result_dir: str | Path) -> list[str]:
    root = Path(result_dir).expanduser()
    if not root.exists():
        return []
    return sorted(p.stem.replace("trades_", "", 1) for p in root.glob("trades_*.csv"))

def load_trades(result_dir: str | Path, champion: str) -> tuple[pd.DataFrame, str | None]:
    result_dir = Path(result_dir).expanduser()
    preferred = result_dir / f"trades_{champion}.csv"
    candidates = [preferred] if preferred.exists() else sorted(result_dir.glob("trades_*.csv"))
    if not candidates:
        return pd.DataFrame(), None
    path = candidates[0]
    try:
        trades = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(), None
    return trades, path.name


def _add_trade_markers(fig, trades: pd.DataFrame, visible_start, visible_end):
    if trades.empty:
        return
    if {"entry_time", "entry_price"}.issubset(trades.columns):
        x = pd.to_datetime(trades.entry_time, utc=True, errors="coerce")
        mask = x.notna() & (x >= visible_start.tz_convert("UTC")) & (x <= visible_end.tz_convert("UTC"))
        sub = trades.loc[mask].copy()
        if len(sub):
            ex = pd.DatetimeIndex(pd.to_datetime(sub.entry_time, utc=True)).tz_convert(TZ)
            hover = []
            for row in sub.itertuples():
                hover.append(
                    f"Entrée<br>prix={getattr(row,'entry_price',float('nan')):.4f}"
                    f"<br>qty={getattr(row,'quantity','—')}<br>stop={getattr(row,'stop','—')}"
                    f"<br>target={getattr(row,'target','—')}"
                )
            fig.add_trace(go.Scatter(x=ex, y=sub.entry_price, mode="markers", name="Entrées",
                                     marker=dict(symbol="triangle-up", size=11), text=hover,
                                     hovertemplate="%{text}<extra></extra>"), row=1, col=1)
    if {"exit_time", "exit_price"}.issubset(trades.columns):
        x = pd.to_datetime(trades.exit_time, utc=True, errors="coerce")
        mask = x.notna() & (x >= visible_start.tz_convert("UTC")) & (x <= visible_end.tz_convert("UTC"))
        sub = trades.loc[mask].copy()
        if len(sub):
            xx = pd.DatetimeIndex(pd.to_datetime(sub.exit_time, utc=True)).tz_convert(TZ)
            reason = sub.get("exit_reason", pd.Series(["sortie"] * len(sub), index=sub.index)).astype(str)
            hover = [f"Sortie<br>prix={p:.4f}<br>raison={r}" for p, r in zip(sub.exit_price, reason)]
            fig.add_trace(go.Scatter(x=xx, y=sub.exit_price, mode="markers", name="Sorties",
                                     marker=dict(symbol="triangle-down", size=11), text=hover,
                                     hovertemplate="%{text}<extra></extra>"), row=1, col=1)


def build_market_figure(
    bars: pd.DataFrame,
    feature_set,
    symbol: str,
    champion: str,
    threshold: float,
    sessions: int = 1,
    show_vwap: bool = True,
    show_levels: bool = True,
    show_probable_zones: bool = True,
    show_opening_range: bool = True,
    show_patterns: bool = True,
    show_pattern_geometry: bool = False,
    show_trades: bool = True,
    trades: pd.DataFrame | None = None,
    runtime_state: dict | None = None,
    base_bars_5m: pd.DataFrame | None = None,
):
    if bars.empty:
        raise ValueError("Aucune bougie à afficher.")
    sessions = max(1, int(sessions))
    selected_sessions = list(pd.unique(bars.Session))[-sessions:]
    view = bars.loc[bars.Session.isin(selected_sessions)].copy()
    feats = feature_set.values.reindex(view.index)
    x = view.index.tz_convert(TZ)
    timeframe = int(bars.attrs.get("timeframe", 5))

    score_title = "Score du Champion" if timeframe == 5 else "Score descriptif · timeframe non validé"
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.025,
                        row_heights=[0.68, 0.17, 0.15],
                        subplot_titles=(f"{symbol} — {timeframe} min", "Volume", score_title))
    fig.add_trace(go.Candlestick(x=x, open=view.Open, high=view.High, low=view.Low, close=view.Close,
                                 name=symbol, increasing_line_width=1, decreasing_line_width=1), row=1, col=1)

    if show_vwap and "VWAP" in feats:
        fig.add_trace(go.Scatter(x=x, y=feats.VWAP, mode="lines", name="VWAP", line=dict(width=1.5)), row=1, col=1)

    if show_levels:
        if "NearestSupport" in feats:
            fig.add_trace(go.Scatter(x=x, y=feats.NearestSupport, mode="lines", name="Support confirmé",
                                     line=dict(width=1.4, dash="dot"), connectgaps=False), row=1, col=1)
        if "NearestResistance" in feats:
            fig.add_trace(go.Scatter(x=x, y=feats.NearestResistance, mode="lines", name="Résistance confirmée",
                                     line=dict(width=1.4, dash="dot"), connectgaps=False), row=1, col=1)
        if show_probable_zones and len(feats):
            # Zones heuristiques autour du dernier niveau causal connu. Le mot "probable"
            # décrit une zone technique, pas une probabilité statistique calibrée.
            latest = feats.iloc[-1]
            for col, touches_col, label in (("NearestSupport", "SupportTouches", "Zone support probable"),
                                             ("NearestResistance", "ResistanceTouches", "Zone résistance probable")):
                value = latest.get(col)
                if pd.notna(value):
                    value = float(value)
                    touches = int(latest.get(touches_col, 0) or 0)
                    # 0,15 % correspond à la tolérance de regroupement historique V8/V10.
                    band = value * 0.0015
                    fig.add_hrect(y0=value-band, y1=value+band, opacity=min(0.05 + 0.03 * max(touches,1), 0.16),
                                  line_width=0, annotation_text=f"{label} · {touches} touche(s)",
                                  annotation_position="right", row=1, col=1)

    if show_opening_range:
        # OR20 exact à partir des bougies source 5 min, même si la vue est agrégée en 10/15/30 min.
        base = base_bars_5m if base_bars_5m is not None else bars
        base = base.loc[base.Session.isin(selected_sessions)].copy()
        hi_x, hi_y, lo_x, lo_y = [], [], [], []
        for day, part in base.groupby("Session", sort=True):
            if part.empty:
                continue
            session_open = part.SessionOpen.iloc[0]
            session_close = part.SessionClose.iloc[0]
            first20 = part.loc[part.Start < session_open + pd.Timedelta(minutes=20)]
            # Exige bien les quatre bougies 5m terminées pour ne pas inventer une OR partielle.
            if int(base.attrs.get("timeframe", 5)) == 5 and len(first20) >= 4:
                start = session_open + pd.Timedelta(minutes=20)
                hi, lo = float(first20.iloc[:4].High.max()), float(first20.iloc[:4].Low.min())
                hi_x.extend([start.tz_convert(TZ), session_close.tz_convert(TZ), None]); hi_y.extend([hi, hi, None])
                lo_x.extend([start.tz_convert(TZ), session_close.tz_convert(TZ), None]); lo_y.extend([lo, lo, None])
        if hi_x:
            fig.add_trace(go.Scatter(x=hi_x, y=hi_y, mode="lines", name="OR20 High",
                                     line=dict(width=1, dash="dash"), connectgaps=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=lo_x, y=lo_y, mode="lines", name="OR20 Low",
                                     line=dict(width=1, dash="dash"), connectgaps=False), row=1, col=1)

    visible_start, visible_end = x.min(), x.max()
    events = feature_set.events.copy()
    if show_patterns and not events.empty:
        events = events.loc[events.accepted.astype(bool)] if "accepted" in events else events
        if len(events):
            bdates = pd.to_datetime(events.breakout_date, utc=True, errors="coerce")
            mask = bdates.notna() & (bdates >= visible_start.tz_convert("UTC")) & (bdates <= visible_end.tz_convert("UTC"))
            ev = events.loc[mask].copy()
            for row in ev.itertuples():
                t = pd.Timestamp(row.breakout_date).tz_convert(TZ)
                nearest_ix = view.index.get_indexer([pd.Timestamp(row.breakout_date)], method="nearest")[0]
                y = float(view.Close.iloc[nearest_ix])
                text = (f"Pattern {row.pattern}<br>confirmé: {pd.Timestamp(row.formation_confirm_date).tz_convert(TZ)}"
                        f"<br>breakout: {t}<br>neckline={float(row.neckline):.4f}"
                        f"<br>qualité={float(row.quality_score):.1f}<br>volume ratio={float(row.volume_ratio):.2f}")
                fig.add_trace(go.Scatter(x=[t], y=[y], mode="markers+text", text=[str(row.pattern)],
                                         textposition="top center", name=f"Pattern {row.pattern}",
                                         marker=dict(symbol="diamond", size=10), hovertext=[text],
                                         hovertemplate="%{hovertext}<extra></extra>", showlegend=False), row=1, col=1)
                if show_pattern_geometry:
                    px = _to_ny([row.start_date, row.middle_date, row.end_date])
                    py = []
                    for dt in pd.to_datetime([row.start_date, row.middle_date, row.end_date], utc=True):
                        i = view.index.get_indexer([dt], method="nearest")[0]
                        if row.pattern == "W":
                            py.append(float(view.Low.iloc[i]) if len(py) != 1 else float(view.High.iloc[i]))
                        else:
                            py.append(float(view.High.iloc[i]) if len(py) != 1 else float(view.Low.iloc[i]))
                    fig.add_trace(go.Scatter(x=px, y=py, mode="lines+markers", name=f"Géométrie {row.pattern}",
                                             line=dict(width=1, dash="dot"), showlegend=False,
                                             hovertemplate="Tracé géométrique visible seulement après confirmation<extra></extra>"), row=1, col=1)

    # Patterns formés mais neckline encore en attente, seulement état courant déjà causal.
    if show_patterns:
        for pending in feature_set.pending or []:
            confirm = pd.Timestamp(pending["formation_confirm_date"])
            if confirm.tzinfo is None:
                continue
            confirm = confirm.tz_convert(TZ)
            if visible_start <= confirm <= visible_end:
                fig.add_trace(go.Scatter(x=[confirm], y=[float(pending["neckline"])], mode="markers",
                                         name="Pattern en attente", marker=dict(symbol="circle-open", size=10),
                                         hovertext=[f"{pending['pattern']} confirmé, neckline en attente"
                                                    f"<br>neckline={float(pending['neckline']):.4f}"
                                                    f"<br>qualité={float(pending['quality_score']):.1f}"],
                                         hovertemplate="%{hovertext}<extra></extra>", showlegend=False), row=1, col=1)

    if show_trades and trades is not None:
        _add_trade_markers(fig, trades, visible_start, visible_end)

    state = runtime_state or {}
    state_symbol = str(state.get("symbol", symbol)).upper()
    if state_symbol == symbol.upper() and state.get("market_timestamp"):
        try:
            signal_t = pd.Timestamp(state["market_timestamp"])
            if signal_t.tzinfo is not None:
                signal_t = signal_t.tz_convert(TZ)
                if visible_start <= signal_t <= visible_end:
                    y = _safe_float(state.get("price"))
                    if y is not None:
                        fig.add_trace(go.Scatter(x=[signal_t], y=[y], mode="markers", name=state.get("decision", "Signal"),
                                                 marker=dict(symbol="star", size=14),
                                                 hovertext=[f"{state.get('decision','SIGNAL')}<br>{state.get('reason','')}"],
                                                 hovertemplate="%{hovertext}<extra></extra>"), row=1, col=1)
            for key, label, dash in (("entry_estimate", "Entrée théorique", "solid"),
                                     ("stop", "Stop", "dash"), ("target", "Target", "dash")):
                value = _safe_float(state.get(key))
                if value is not None:
                    fig.add_hline(y=value, line_dash=dash, line_width=1, annotation_text=label,
                                  annotation_position="right", row=1, col=1)
        except Exception:
            pass

    fig.add_trace(go.Bar(x=x, y=view.Volume, name="Volume", showlegend=False), row=2, col=1)

    if champion in RULE_IDS:
        try:
            scores = rule_predictions(bars, feature_set.values, view.index)[champion]
            score_name = f"Score {champion}" if timeframe == 5 else f"Score descriptif {champion}"
            fig.add_trace(go.Scatter(x=x, y=scores, mode="lines", name=score_name), row=3, col=1)
            fig.add_hline(y=float(threshold), line_dash="dash", line_width=1,
                          annotation_text=f"seuil {threshold:.2f}", row=3, col=1)
            fig.update_yaxes(range=[-0.05, 1.05], row=3, col=1)
        except Exception:
            pass

    fig.update_layout(height=820, margin=dict(l=16, r=16, t=56, b=16), template="plotly_dark",
                      xaxis_rangeslider_visible=False, hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor")
    fig.update_yaxes(title_text="Prix USD", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="Score", row=3, col=1)
    return fig


def read_runtime_state(symbol: str, runtime_root: str | Path = "runtime") -> dict:
    root = Path(runtime_root)
    candidates = [root / symbol.upper() / "runtime_state.json", root / "runtime_state.json"]
    for p in candidates:
        if p.exists():
            try:
                state = json.loads(p.read_text(encoding="utf-8"))
                state_symbol = state.get("symbol")
                if state_symbol and str(state_symbol).upper() == symbol.upper():
                    return state
                # Compatibilité V10.0.x: l'ancien état sans symbole était SPY-only.
                if not state_symbol and symbol.upper() == "SPY":
                    return state
            except Exception:
                continue
    return {}
