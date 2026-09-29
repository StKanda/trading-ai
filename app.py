"""Trading AI V10.2 dashboard. Research/paper only; no broker execution."""
from pathlib import Path
import json
import os
import sqlite3
import sys
import time
import pandas as pd
import streamlit as st

from trading_lab.config import AppConfig
from trading_lab.data.freshness import evaluate_freshness
from trading_lab.data.market import completed_bars
from trading_lab.data.providers import CSVProvider, YahooProvider, write_snapshot
from trading_lab.features.engine import FeatureEngine
from trading_lab.ui.charting import (
    build_market_figure,
    discover_local_symbols,
    discover_trade_models,
    load_trades,
    read_runtime_state,
    resolve_symbol_directory,
)
from trading_lab.ui.jobs import launch_job, read_job, request_stop

st.set_page_config(page_title="Trading AI V10.3", page_icon="📈", layout="wide", initial_sidebar_state="expanded")
cfg = AppConfig.load(Path("config.json"))


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def _load_ui_settings():
    p = Path("runtime/ui_settings.json")
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_ui_settings(values):
    p = Path("runtime/ui_settings.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(values, indent=2, ensure_ascii=False), encoding="utf-8")


@st.cache_data(ttl=10, show_spinner=False)
def _fetch_yahoo(symbol: str):
    # Cache court: évite des appels doublés pendant un même rerun tout en laissant
    # le mode direct rafraîchir régulièrement la dernière bougie complète.
    return YahooProvider().fetch(symbol)


def _load_snapshot(source: str, data_root: str, symbol: str):
    if source == "Local CSV":
        if not data_root:
            raise ValueError("Indique le répertoire de données local dans la barre latérale.")
        symbol_dir = resolve_symbol_directory(data_root, symbol)
        return CSVProvider(symbol_dir).fetch(symbol), f"Local · {symbol_dir}"
    return _fetch_yahoo(symbol), "Yahoo / yfinance · délai non garanti"


def _job_data_dir(source: str, data_root: str, symbol: str, purpose: str) -> Path:
    """Retourne un dossier CSV figé pour un job de calcul.

    Yahoo est capturé avant le calcul afin que le backtest / DL reste reproductible.
    """
    if source == "Local CSV":
        return resolve_symbol_directory(data_root, symbol)
    snap = _fetch_yahoo(symbol)
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    out = Path("runtime") / "ui_snapshots" / symbol / f"{purpose}_{stamp}"
    write_snapshot(snap, out)
    return out


def _available_sessions(source: str, data_root: str, symbol: str) -> list[str]:
    snap, _ = _load_snapshot(source, data_root, symbol)
    idx = snap.prices.index.tz_convert("America/New_York")
    return sorted(pd.unique(idx.strftime("%Y-%m-%d")).tolist())


def _job_panel(job_id: str, result_dir: Path | None = None, *, key_prefix: str):
    job = read_job(job_id)
    state = str(job.get("state", "IDLE"))
    cols = st.columns([1.2, 1, 1, 4])
    cols[0].metric("État", state)
    cols[1].metric("Code retour", "—" if job.get("returncode") is None else job.get("returncode"))
    started = job.get("started_at") or job.get("created_at")
    if started:
        elapsed = max(0, time.time() - float(started))
        cols[2].metric("Durée", f"{elapsed:.0f} s")
    else:
        cols[2].metric("Durée", "—")
    if state in {"QUEUED", "RUNNING"}:
        cols[3].progress(0.35 if state == "QUEUED" else 0.65, text="Calcul en cours — journal actualisé automatiquement")
    elif state == "COMPLETED":
        cols[3].success("Terminé")
    elif state == "FAILED":
        cols[3].error("Échec — voir le journal")
    if job.get("log_tail"):
        st.code(job["log_tail"], language="text")
    if state in {"QUEUED", "RUNNING"}:
        if st.button("Arrêter", key=f"{key_prefix}_stop"):
            request_stop(job_id)
    if result_dir and state == "COMPLETED" and result_dir.exists():
        metrics = result_dir / "metrics.csv"
        if metrics.exists():
            df = pd.read_csv(metrics)
            keep = [c for c in ["model","final_capital","total_return","max_drawdown","sharpe","trades","roc_auc"] if c in df.columns]
            st.dataframe(df[keep] if keep else df, use_container_width=True, hide_index=True)
            if "final_capital" in df.columns:
                st.bar_chart(df.set_index("model")["final_capital"])
        cap = result_dir / "capital_daily.csv"
        if cap.exists():
            st.line_chart(pd.read_csv(cap, index_col=0))


def _fmt_price(value):
    try:
        return f"{float(value):,.2f}"
    except Exception:
        return "—"


def _fmt_status(state: dict, provider_freshness: dict | None):
    decision = state.get("decision", "NO DATA")
    market_status = (provider_freshness or {}).get("status", "UNKNOWN")
    return decision, market_status


st.markdown(
    """
<style>
:root { --ta-accent:#5eead4; --ta-panel:#111827; --ta-border:#263244; --ta-muted:#94a3b8; }
[data-testid="stAppViewContainer"] { background: linear-gradient(180deg,#0a0f18 0%,#0d131e 34%,#0b111a 100%); }
[data-testid="stHeader"] { background: rgba(10,15,24,.84); backdrop-filter: blur(10px); }
[data-testid="stSidebar"] { background:#0a1019; border-right:1px solid #1f2a39; }
[data-testid="stSidebar"] .block-container { padding-top:1.25rem; }
.block-container { max-width:1680px; padding-top:1.4rem; padding-bottom:2rem; }
.ta-hero { border:1px solid var(--ta-border); background:linear-gradient(120deg,rgba(17,24,39,.95),rgba(15,23,42,.72)); border-radius:16px; padding:18px 22px; margin-bottom:14px; display:flex; justify-content:space-between; align-items:center; gap:18px; }
.ta-kicker { color:var(--ta-accent); font-size:.72rem; letter-spacing:.13em; font-weight:700; }
.ta-title { font-size:1.65rem; line-height:1.05; font-weight:750; margin-top:4px; }
.ta-sub { color:var(--ta-muted); margin-top:6px; font-size:.88rem; }
.ta-live { border:1px solid #2d3b4e; border-radius:999px; padding:7px 11px; color:#cbd5e1; white-space:nowrap; font-size:.78rem; }
.ta-live b { color:var(--ta-accent); }
[data-testid="stMetric"] { border:1px solid var(--ta-border); background:rgba(17,24,39,.74); padding:12px 14px; border-radius:12px; }
[data-testid="stMetricLabel"] { color:#9ca3af; }
[data-testid="stTabs"] button { font-weight:650; }
div[data-testid="stPlotlyChart"] { border:1px solid var(--ta-border); border-radius:14px; overflow:hidden; background:#0b111b; padding:4px; }
.ta-callout { border:1px solid var(--ta-border); border-radius:12px; padding:12px 14px; background:rgba(15,23,42,.68); color:#cbd5e1; margin:.25rem 0 .8rem; }
.ta-callout strong { color:#f8fafc; }
.ta-small { color:var(--ta-muted); font-size:.78rem; }
</style>
""",
    unsafe_allow_html=True,
)

settings = _load_ui_settings()
default_data = os.environ.get("TRADING_AI_DATA_DIR", settings.get("data_root", ""))
default_watch = settings.get("watchlist", list(cfg.symbols))

# ---------------------------------------------------------------------------
# Sidebar = cockpit. Everything that changes the chart is in one place.
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Marché")
    source = st.radio(
        "Source de données",
        ["Local CSV", "Yahoo"],
        index=0 if settings.get("source", "Local CSV") == "Local CSV" else 1,
        help="Yahoo permet le suivi automatique, mais son délai n'est pas garanti. Local CSV sert au replay et aux snapshots.",
    )
    data_root = st.text_input(
        "Répertoire local",
        default_data,
        disabled=source != "Local CSV",
        placeholder="/var/home/giron/trading-ai/.../data_spy_02~",
    )
    watch_text = st.text_input("Watchlist", ",".join(default_watch), help="Ex. SPY,QQQ,NVDA,AAPL")
    watchlist = [x.strip().upper() for x in watch_text.split(",") if x.strip()]
    local_symbols = discover_local_symbols(data_root) if source == "Local CSV" and data_root else []
    choices = list(dict.fromkeys(watchlist + local_symbols)) or ["SPY"]
    current = str(settings.get("symbol", "SPY")).upper()
    idx = choices.index(current) if current in choices else 0
    symbol = st.selectbox("Action / ETF", choices, index=idx)
    custom_symbol = st.text_input("Autre symbole", "", placeholder="AMD")
    if custom_symbol.strip():
        symbol = custom_symbol.strip().upper()

    st.divider()
    st.markdown("### Graphique")
    timeframe = st.selectbox(
        "Intervalle des bougies",
        [5, 10, 15, 30],
        index=[5, 10, 15, 30].index(int(settings.get("timeframe", 5))) if int(settings.get("timeframe", 5)) in [5, 10, 15, 30] else 0,
        format_func=lambda x: f"{x} min",
        help="Les signaux du Champion restent validés sur 5 min. Les autres intervalles servent à la lecture du marché.",
    )
    window_label = st.selectbox(
        "Période affichée",
        ["1 séance", "2 jours", "5 jours", "10 jours", "20 jours"],
        index=0,
    )
    n_sessions = {"1 séance": 1, "2 jours": 2, "5 jours": 5, "10 jours": 10, "20 jours": 20}[window_label]

    direct_mode = st.toggle(
        "Mode direct · actualisation auto",
        value=bool(settings.get("direct_mode", False)) and source == "Yahoo",
        disabled=source != "Yahoo",
        help="Rafraîchit Yahoo automatiquement. Ce n'est pas un flux tick-by-tick ni une garantie de temps réel boursier.",
    )
    refresh_seconds = st.select_slider(
        "Rafraîchissement",
        options=[15, 30, 60, 120],
        value=int(settings.get("refresh_seconds", 30)) if int(settings.get("refresh_seconds", 30)) in [15, 30, 60, 120] else 30,
        format_func=lambda x: f"{x} s",
        disabled=not direct_mode,
    )
    if source == "Yahoo":
        st.caption("Mode direct = polling automatique des dernières bougies Yahoo complètes. Délai fournisseur non garanti.")

    st.divider()
    with st.expander("Calques du graphique", expanded=True):
        show_vwap = st.checkbox("VWAP", True)
        show_levels = st.checkbox("Supports / résistances confirmés", True)
        show_probable_zones = st.checkbox("Zones support / résistance probables", True, help="Zones heuristiques autour des niveaux causaux confirmés; ce ne sont pas des probabilités calibrées.")
        show_or = st.checkbox("Opening Range 20 min", True)
        show_patterns = st.checkbox("Patterns W / M", True)
        show_geometry = st.checkbox("Géométrie W / M", False)
        show_trades = st.checkbox("Entrées / sorties", True)
    result_dir_signal = st.text_input("Dossier des trades", st.session_state.get("latest_backtest_result", settings.get("result_dir", "results")))
    trade_models = discover_trade_models(result_dir_signal)
    trade_overlay_model = st.selectbox("Trades affichés", ["AUTO / Champion"] + trade_models,
                                       help="Choisis explicitement le modèle dont les entrées/sorties doivent être superposées au graphique.")

    if st.button("Enregistrer les préférences", use_container_width=True):
        _save_ui_settings({
            "source": source,
            "data_root": data_root,
            "watchlist": watchlist,
            "symbol": symbol,
            "timeframe": timeframe,
            "direct_mode": direct_mode,
            "refresh_seconds": refresh_seconds,
            "result_dir": result_dir_signal,
        })
        st.success("Préférences enregistrées.")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
mode_text = f"DIRECT · {refresh_seconds}s" if direct_mode and source == "Yahoo" else "MANUEL"
st.markdown(
    f"""
<div class="ta-hero">
  <div>
    <div class="ta-kicker">TRADING AI · V10.3 · PAPER / RESEARCH</div>
    <div class="ta-title">{symbol} <span style="color:#64748b;font-weight:500">· {timeframe} min</span></div>
    <div class="ta-sub">Price action, niveaux causaux, patterns, risque et décisions sur le même écran.</div>
  </div>
  <div class="ta-live"><b>{mode_text}</b> · {source}</div>
</div>
""",
    unsafe_allow_html=True,
)

st.caption("Aucune exécution courtier. Le mode direct utilise des requêtes périodiques; Yahoo peut être retardé et n'est pas présenté comme un feed temps réel garanti.")

sig, bt, dl_tab, rob, lab, model, data_tab, logs = st.tabs([
    "Marché & Signal", "Backtest", "Deep Learning", "Robustesse", "Research Lab", "Modèle", "Données", "Logs"
])


def _render_market_and_signal():
    state = read_runtime_state(symbol)
    try:
        snap, provider_note = _load_snapshot(source, data_root, symbol)
        provider_fresh = evaluate_freshness(snap, timeframe=5).to_dict()
        base_5m = completed_bars(snap, 5)
        bars = base_5m if timeframe == 5 else completed_bars(snap, timeframe)
        feature_set = FeatureEngine().build(bars)
        champion = state.get("champion", "RULE_MOMENTUM")
        overlay_model = champion if trade_overlay_model == "AUTO / Champion" else trade_overlay_model
        trades, trade_file = load_trades(result_dir_signal, overlay_model) if show_trades else (pd.DataFrame(), None)

        decision, market_status = _fmt_status(state, provider_fresh)
        latest_price = float(base_5m.Close.iloc[-1])
        prev_close = float(base_5m.Close.iloc[-2]) if len(base_5m) > 1 else latest_price
        delta_pct = (latest_price / prev_close - 1) * 100 if prev_close else 0.0
        lag_bars = provider_fresh.get("lag_bars")

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Prix", f"${latest_price:,.2f}", f"{delta_pct:+.2f}% / bougie")
        m2.metric("Signal worker", decision)
        m3.metric("Données", market_status)
        m4.metric("Intervalle", f"{timeframe} min")
        m5.metric("Champion", champion)
        m6.metric("Lag feed", f"{lag_bars} barre(s)" if lag_bars is not None else "—")

        if timeframe != 5:
            st.markdown(
                '<div class="ta-callout"><strong>Lecture multi-timeframe.</strong> Le graphique est agrégé en '
                f'{timeframe} min, mais le Champion de recherche reste un Champion 5 min. Son score sur ce graphique est descriptif, pas une validation séparée.</div>',
                unsafe_allow_html=True,
            )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Entrée théorique", _fmt_price(state.get("entry_estimate")))
        c2.metric("Stop", _fmt_price(state.get("stop")))
        c3.metric("Target", _fmt_price(state.get("target")))
        c4.metric("Score", _fmt_price(state.get("model_score")), help="Score interne, pas une probabilité validée.")

        latest_feat = feature_set.values.reindex(bars.index).iloc[-1]
        slevel = latest_feat.get("NearestSupport")
        rlevel = latest_feat.get("NearestResistance")
        stouches = int(latest_feat.get("SupportTouches", 0) or 0)
        rtouches = int(latest_feat.get("ResistanceTouches", 0) or 0)
        lv1, lv2, lv3 = st.columns([1,1,2])
        lv1.metric("Support probable", _fmt_price(slevel), f"{stouches} touche(s)", help="Zone heuristique issue de pivots déjà confirmés; pas une probabilité calibrée.")
        lv2.metric("Résistance probable", _fmt_price(rlevel), f"{rtouches} touche(s)", help="Zone heuristique issue de pivots déjà confirmés; pas une probabilité calibrée.")
        lv3.caption("Les zones probables sont construites uniquement avec les pivots connus à cet instant. Elles ne sont jamais projetées rétroactivement.")

        fig = build_market_figure(
            bars,
            feature_set,
            symbol,
            champion,
            cfg.research.threshold,
            sessions=n_sessions,
            show_vwap=show_vwap,
            show_levels=show_levels,
            show_probable_zones=show_probable_zones,
            show_opening_range=show_or,
            show_patterns=show_patterns,
            show_pattern_geometry=show_geometry,
            show_trades=show_trades,
            trades=trades,
            runtime_state=state,
            base_bars_5m=base_5m,
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displaylogo": False,
                "scrollZoom": False,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            },
        )

        latest_ny = base_5m.index[-1].tz_convert("America/New_York")
        n1, n2, n3, n4 = st.columns(4)
        n1.caption(provider_note)
        n2.caption(f"Dernière bougie 5m complète : {latest_ny:%Y-%m-%d %H:%M ET}")
        n3.caption(f"Bougies affichables : {len(bars):,}")
        n4.caption(f"Ledger : {trade_file or 'aucun'}")

        with st.expander("État du feed et détails"):
            st.json(provider_fresh)
            if state:
                st.markdown("**État du worker**")
                st.json(state)
        if feature_set.pending:
            with st.expander("Patterns confirmés en attente de neckline"):
                st.json(feature_set.pending)
    except Exception as exc:
        st.error(f"Marché indisponible pour {symbol}: {exc}")
        if source == "Yahoo":
            st.info("Vérifie `python -m pip install -r requirements_market.txt` et ta connexion réseau.")


with sig:
    control_left, control_right = st.columns([5, 1])
    with control_left:
        st.markdown("#### Graphique de décision")
        st.caption("Molette = défilement de page. Zoom du graphique uniquement via la barre Plotly ou sélection manuelle; survol pour OHLC, niveaux, patterns et trades.")
    with control_right:
        if st.button("Rafraîchir", use_container_width=True, help="Force un nouveau rendu. Le cache Yahoo expire en 10 secondes."):
            _fetch_yahoo.clear()
            st.rerun()

    run_every = f"{refresh_seconds}s" if direct_mode and source == "Yahoo" else None
    if hasattr(st, "fragment"):
        st.fragment(run_every=run_every)(_render_market_and_signal)()
    else:
        _render_market_and_signal()

    with st.expander("Glossaire du graphique"):
        st.markdown(
            """
- **VWAP** : prix moyen de séance pondéré par le volume.
- **Support / résistance** : niveaux calculés à partir de pivots déjà confirmés; aucun niveau n'est projeté rétroactivement avant sa confirmation.
- **Opening Range 20 min** : plus haut / plus bas exact des quatre premières bougies 5 min, même lorsque le graphique est affiché en 10/15/30 min.
- **W / M** : double creux / double sommet confirmé causalement.
- **Entrée / sortie** : marqueurs provenant d'un ledger de simulation/backtest lorsqu'il existe.
- **Stop / Target** : niveaux théoriques du signal courant.
- **Mode direct** : actualisation automatique du fournisseur Yahoo; ce n'est pas un flux tick-by-tick garanti.
            """
        )

with bt:
    st.markdown("### Backtest walk-forward")
    st.caption("Configure et lance le même moteur `main.py research` depuis l'interface. Le calcul tourne en arrière-plan et le journal est actualisé pendant l'exécution.")
    try:
        sessions = _available_sessions(source, data_root, symbol)
        if len(sessions) < 16:
            st.warning(f"Seulement {len(sessions)} séances disponibles. Le résultat restera exploratoire.")
        b1, b2, b3 = st.columns(3)
        with b1:
            bt_timeframe = st.selectbox("Timeframe backtest", [5,10,15], index=0, format_func=lambda x:f"{x} min", key="bt_tf")
        with b2:
            default_start_i = max(0, len(sessions)-10)
            bt_start = st.selectbox("Début test", sessions, index=default_start_i, key="bt_start", help="Les séances antérieures servent à l'entraînement walk-forward.")
        with b3:
            start_i = sessions.index(bt_start)
            end_choices = sessions[start_i+1:]
            if end_choices:
                chosen_last = end_choices[-1]
                bt_end = (pd.Timestamp(chosen_last) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                st.text_input("Fin exclusive", bt_end, key="bt_end_display", disabled=True)
            else:
                bt_end = (pd.Timestamp(bt_start) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                st.text_input("Fin exclusive", bt_end, key="bt_end_display", disabled=True)
        with st.expander("Paramètres du protocole", expanded=False):
            st.write({"capital_initial": cfg.risk.initial_capital, "frais_bps": cfg.risk.fee_bps,
                      "slippage_bps": cfg.risk.slippage_bps, "seuil": cfg.research.threshold,
                      "horizon_minutes": cfg.research.horizon_minutes})
            st.caption("Ces paramètres restent ceux du protocole config.json afin d'éviter un réglage opportuniste dans l'interface.")

        job_id = f"backtest_{symbol}"
        if st.button("▶ Lancer le backtest", type="primary", use_container_width=True, key="launch_bt"):
            job_data = _job_data_dir(source, data_root, symbol, "backtest")
            stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
            result_dir = Path("runtime") / "ui_results" / f"{symbol}_{bt_timeframe}m_{stamp}"
            command = [sys.executable, "main.py", "research", "--data", str(job_data), "--symbol", symbol,
                       "--timeframe", str(bt_timeframe), "--test-start", bt_start, "--test-end", bt_end,
                       "--out", str(result_dir)]
            launch_job(job_id, command)
            st.session_state[f"{job_id}_result"] = str(result_dir)
            st.session_state["latest_backtest_result"] = str(result_dir)
            st.rerun()
        result_dir = Path(st.session_state.get(f"{job_id}_result", result_dir_signal))

        def _render_bt_job():
            _job_panel(job_id, result_dir, key_prefix="bt")
            if result_dir.exists():
                models = discover_trade_models(result_dir)
                if models:
                    selected_model = st.selectbox("Afficher les trades du modèle", models, key="bt_trade_model")
                    trades, _ = load_trades(result_dir, selected_model)
                    if not trades.empty:
                        st.dataframe(trades, use_container_width=True, hide_index=True)
                    st.caption(f"Pour superposer ces entrées/sorties sur le graphique principal, mets le dossier des trades sur `{result_dir}` et choisis `{selected_model}`.")
        if hasattr(st, "fragment"):
            st.fragment(run_every="2s")(_render_bt_job)()
        else:
            _render_bt_job()
    except Exception as exc:
        st.error(f"Backtest indisponible : {exc}")

with dl_tab:
    st.markdown("### Entraînement Deep Learning")
    st.caption("Lance MLP ou TCN sur un snapshot figé du symbole actif. Le résultat reste DEVELOPMENT ONLY tant qu'il n'est pas validé dans le walk-forward économique.")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        dl_arch = st.selectbox("Architecture", ["mlp","tcn"], format_func=lambda x: x.upper(), key="dl_arch")
    with d2:
        dl_context = st.selectbox("Contexte", [12,24,48,78], index=1, format_func=lambda x:f"{x} bougies", key="dl_context")
    with d3:
        dl_epochs = st.number_input("Epochs max", min_value=3, max_value=100, value=20, step=1, key="dl_epochs")
    with d4:
        dl_patience = st.number_input("Early stopping", min_value=1, max_value=20, value=4, step=1, key="dl_patience")
    dl_seed = st.number_input("Seed", min_value=0, max_value=999999, value=42, step=1, key="dl_seed")
    st.caption("Le scaling est appris sur train uniquement. CUDA est utilisé automatiquement si PyTorch CUDA est disponible.")
    dl_job_id = f"dl_{symbol}"
    if st.button("▶ Lancer l'apprentissage DL", type="primary", use_container_width=True, key="launch_dl"):
        try:
            job_data = _job_data_dir(source, data_root, symbol, "dl")
            stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
            out_dir = Path("runtime") / "ui_dl" / f"{symbol}_{dl_arch}_{dl_context}_{stamp}"
            command = [sys.executable, "main.py", "train-dl", "--data", str(job_data), "--symbol", symbol,
                       "--context", str(dl_context), "--architecture", dl_arch, "--epochs", str(int(dl_epochs)),
                       "--patience", str(int(dl_patience)), "--seed", str(int(dl_seed)), "--out", str(out_dir)]
            launch_job(dl_job_id, command)
            st.session_state[f"{dl_job_id}_out"] = str(out_dir)
            st.rerun()
        except Exception as exc:
            st.error(f"Impossible de lancer le DL : {exc}")
    dl_out = Path(st.session_state.get(f"{dl_job_id}_out", "runtime/dl_smoke"))

    def _render_dl_job():
        _job_panel(dl_job_id, None, key_prefix="dl")
        report = dl_out / "dl_smoke.json"
        if report.exists():
            result = json.loads(report.read_text(encoding="utf-8"))
            r1,r2,r3,r4 = st.columns(4)
            r1.metric("Best epoch", result.get("best_epoch", "—"))
            r2.metric("Train loss", f"{result.get('train_loss', float('nan')):.4f}" if result.get('train_loss') is not None else "—")
            r3.metric("Valid loss", f"{result.get('valid_loss', float('nan')):.4f}" if result.get('valid_loss') is not None else "—")
            r4.metric("Device", result.get("device", "—"))
            st.json(result)
    if hasattr(st, "fragment"):
        st.fragment(run_every="2s")(_render_dl_job)()
    else:
        _render_dl_job()

with rob:
    result_dir = Path(st.text_input("Dossier robustesse", "results", key="robdir"))
    for name in ("cost_stress.csv", "bootstrap_daily.csv", "ablation_deltas.csv", "split_audit.csv"):
        p = result_dir / name
        if p.exists():
            st.markdown(f"**{name}**")
            st.dataframe(pd.read_csv(p), use_container_width=True)

with lab:
    db_candidates = [Path("runtime") / symbol / "experiments.db", Path("runtime/experiments.db")]
    db = next((p for p in db_candidates if p.exists()), db_candidates[0])
    if db.exists():
        with sqlite3.connect(db) as con:
            exp = pd.read_sql_query("select * from experiments order by created_at desc limit 200", con)
        st.metric("Expériences enregistrées", len(exp))
        st.dataframe(exp, use_container_width=True)
    else:
        st.info(f"Aucune expérience enregistrée pour {symbol}.")
    report_candidates = [Path("runtime") / symbol / "research_cycle_report.json", Path("runtime/research_cycle_report.json")]
    report = next((p for p in report_candidates if p.exists()), report_candidates[0])
    if report.exists():
        st.json(json.loads(report.read_text()))
    st.warning("Le Research Worker V10.2 reste le smoke test borné tant que le moteur de recherche réel n'est pas branché.")

with model:
    state = read_runtime_state(symbol)
    st.write("Symbole actif", symbol)
    st.write("Champion actif", state.get("champion", "UNKNOWN"))
    st.write("Les promotions sont manuelles par défaut. Un score n'est pas une probabilité validée.")
    st.warning("Un Champion validé sur un symbole ou un timeframe n'est pas automatiquement validé sur un autre.")

with data_tab:
    audit_candidates = [Path("runtime") / symbol / "data_audit.json", Path("runtime/data_audit.json")]
    audit = next((p for p in audit_candidates if p.exists()), audit_candidates[0])
    if audit.exists():
        st.json(json.loads(audit.read_text()))
    else:
        st.info("Lancer `python main.py audit-v10 ...` pour générer l'audit.")
    if source == "Local CSV" and data_root:
        st.write("Symboles locaux détectés", local_symbols or "aucun")

with logs:
    logdirs = [Path("logs") / symbol, Path("logs")]
    files = []
    for logdir in logdirs:
        if logdir.exists():
            files.extend(sorted(logdir.glob("*.log")))
    files = list(dict.fromkeys(files))
    if files:
        selected = st.selectbox("Log", files, format_func=lambda p: p.name)
        st.code("\n".join(selected.read_text(errors="replace").splitlines()[-200:]))
    else:
        st.info("Aucun log disponible.")
