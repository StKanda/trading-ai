"""V9 : recherche pré-test puis évaluation finale séparée et verrouillée."""
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
import pandas as pd
import sklearn
from ..config import AppConfig, RiskConfig, ResearchConfig
from ..data.market import Snapshot, completed_bars, TZ
from ..data.providers import fingerprint
from ..features.engine import FeatureEngine, GROUPS, build_labels
from .models import MODEL_IDS, eligible_train, fit_predict
from .rules import RULE_IDS, rule_predictions
from .backtest import simulate, reference_curve, Simulation
from .metrics import classification, performance, ablations

ALL_ACTIVE_IDS = MODEL_IDS + RULE_IDS
OFFICIAL_CANDIDATES = ["RF_A", "RF_D", "GB_D", "LR_A", "LR_D"] + RULE_IDS


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def code_hashes():
    root = Path(__file__).resolve().parents[1]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*.py"))}


def versions():
    return dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__)


def slice_snapshot(snapshot, before, after=None):
    days = snapshot.prices.index.tz_convert(TZ).strftime("%Y-%m-%d")
    mask = days < before
    if after is not None:
        mask &= days >= after
    prices = snapshot.prices.loc[mask]
    if prices.empty:
        raise ValueError("Aucune donnée dans la période demandée.")
    schedule = snapshot.schedule.loc[snapshot.schedule.index < before]
    if after is not None:
        schedule = schedule.loc[schedule.index >= after]
    return Snapshot(snapshot.symbol, prices, schedule, snapshot.metadata.copy())


def make_output(path):
    p = Path(path)
    if p.exists():
        raise ValueError(f"{p} existe déjà. Choisir un nouveau dossier pour préserver les résultats.")
    p.mkdir(parents=True)
    return p


def cash_curve(bars, risk):
    rows = []
    for end, bar in bars.iterrows():
        rows.append(dict(End=end, Session=bar.Session, Capital=risk.initial_capital, NetReturn=0.0,
                         Drawdown=0.0, Position=0, Exposed=0, Entries=0, PositionChanges=0,
                         Halted=0, DailyPnL=0.0, Signal=0, ModelScore=np.nan))
    return Simulation(pd.DataFrame(rows).set_index("End"), pd.DataFrame())


def bootstrap_daily(curves, evaluated, initial_capital, seed=42, draws=4000):
    rng = np.random.default_rng(seed)
    rows = []
    for name, capital in curves.items():
        daily = capital.groupby(evaluated.Session).last()
        ret = daily.pct_change()
        if len(ret):
            ret.iloc[0] = daily.iloc[0] / initial_capital - 1
        values = ret.dropna().to_numpy(dtype=float)
        if len(values) < 5:
            rows.append(dict(model=name, sessions=len(values), bootstrap_draws=0,
                             mean_daily_return=np.nan, ci95_low=np.nan, ci95_high=np.nan,
                             positive_mean_fraction=np.nan))
            continue
        means = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
        rows.append(dict(model=name, sessions=len(values), bootstrap_draws=draws,
                         mean_daily_return=float(values.mean()),
                         ci95_low=float(np.quantile(means, .025)),
                         ci95_high=float(np.quantile(means, .975)),
                         positive_mean_fraction=float((means > 0).mean())))
    return pd.DataFrame(rows)


def stress_table(evaluated, predictions, features, cfg):
    rows = []
    for extra_slip in (2.0, 5.0):
        stressed = replace(cfg.risk, slippage_bps=cfg.risk.slippage_bps + extra_slip)
        for model_id in OFFICIAL_CANDIDATES:
            sim = simulate(evaluated, predictions[model_id], features, stressed, cfg.research)
            p = performance(sim, stressed.initial_capital)
            rows.append(dict(model=model_id, extra_slippage_bps=extra_slip,
                             total_return=p["total_return"], final_capital=p["final_capital"],
                             max_drawdown=p["max_drawdown"], trades=p["trades"]))
    return pd.DataFrame(rows)


def export_results(out, snapshot, bars, fset, labels, predictions, audits, cfg, timeframe, phase):
    out = Path(out)
    metrics, curves = [], {}
    evaluation_days = bars.loc[bars.index.isin(predictions.index), "Session"].unique()
    evaluated = bars.loc[bars.Session.isin(evaluation_days)].copy()
    if not len(evaluated):
        raise ValueError("Aucune période hors échantillon disponible.")

    for model_id in ALL_ACTIVE_IDS:
        sim = simulate(evaluated, predictions[model_id], fset.values, cfg.risk, cfg.research)
        row = dict(model=model_id, kind="rule" if model_id.startswith("RULE_") else "ml",
                   **performance(sim, cfg.risk.initial_capital))
        if model_id in MODEL_IDS:
            row.update(classification(labels, predictions[model_id], cfg.research.threshold))
        metrics.append(row)
        curves[model_id] = sim.ledger.Capital
        sim.ledger.to_csv(out / f"ledger_{model_id}.csv")
        sim.trades.to_csv(out / f"trades_{model_id}.csv", index=False)

    for name, intraday in (("BUY_HOLD_X1", False), ("INTRADAY_HOLD_X1", True)):
        sim = reference_curve(evaluated, cfg.risk, intraday)
        metrics.append(dict(model=name, kind="benchmark", **performance(sim, cfg.risk.initial_capital)))
        curves[name] = sim.ledger.Capital
        sim.ledger.to_csv(out / f"ledger_{name}.csv")

    cash = cash_curve(evaluated, cfg.risk)
    metrics.append(dict(model="CASH", kind="benchmark", **performance(cash, cfg.risk.initial_capital)))
    curves["CASH"] = cash.ledger.Capital

    table = pd.DataFrame(metrics)
    table.to_csv(out / "metrics.csv", index=False)
    ablations(table).to_csv(out / "ablation_deltas.csv", index=False)
    stress = stress_table(evaluated, predictions, fset.values, cfg)
    stress.to_csv(out / "cost_stress.csv", index=False)
    bootstrap_daily(curves, evaluated, cfg.risk.initial_capital, cfg.research.seed).to_csv(
        out / "bootstrap_daily.csv", index=False)
    predictions.to_csv(out / "predictions.csv")
    labels.reindex(predictions.index).to_csv(out / "labels_oos.csv")
    fset.values.to_csv(out / "features.csv")
    fset.pivots.to_csv(out / "pivots_confirmed.csv", index=False)
    fset.events.to_csv(out / "patterns_causal.csv", index=False)
    bars.to_csv(out / "bars.csv")
    pd.DataFrame(audits).to_csv(out / "split_audit.csv", index=False)
    daily = pd.concat(curves, axis=1).groupby(evaluated.Session).last()
    daily.to_csv(out / "capital_daily.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for name in ("BUY_HOLD_X1", "INTRADAY_HOLD_X1", "RF_A", "RF_D", "GB_D", "LR_A",
                 "RULE_MOMENTUM", "RULE_VWAP_RECLAIM", "RULE_ORB20", "CASH"):
        if name in daily:
            ax.plot(pd.to_datetime(daily.index), daily[name], label=name)
    synthetic = bool(snapshot.metadata.get("synthetic"))
    ax.set(title=("SYNTHÉTIQUE — vérification logicielle seulement\n" if synthetic else "") +
           f"{snapshot.symbol} · {timeframe} min · {phase}", ylabel="Capital en USD")
    ax.legend(ncol=3, frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(out / "capital.png", dpi=140); plt.close(fig)

    sessions = len(evaluation_days)
    reliability = "INSUFFISANT POUR CONCLUSION STATISTIQUE" if sessions < 100 else "À INTERPRÉTER AVEC PRUDENCE"
    (out / "LIRE_AVANT_RESULTATS.md").write_text(
        ("# DONNÉES SYNTHÉTIQUES : aucune conclusion de trading\n\n" if synthetic else "# Résultats V9\n\n") +
        f"Séances évaluées : {sessions}. Statut : **{reliability}**.\n\n"
        "Les scores RF/GB/LR ne sont pas des probabilités validées. Aucune calibration n'est revendiquée.\n"
        "Signal à la clôture d'une bougie, entrée uniquement à l'ouverture suivante. Aucune position overnight.\n"
        "Si stop et objectif sont touchés dans une même bougie, le stop est prioritaire. Un gap peut dépasser le budget.\n"
        "Les stress de coûts ajoutent 2 puis 5 bps de slippage PAR CÔTÉ aux hypothèses de base.\n"
        "Le bootstrap par séance est descriptif et ne prouve pas l'indépendance des observations.\n"
        "Les labels 30 min se chevauchent ; le nombre de lignes n'est pas un nombre d'observations indépendantes.\n",
        encoding="utf-8")
    return table, stress


def _choose_official(table, stress, evaluation_sessions):
    base = table.set_index("model")
    wide = stress.pivot(index="model", columns="extra_slippage_bps", values="total_return")
    rows = []
    min_trades = max(5, int(np.ceil(evaluation_sessions / 3)))
    for model in OFFICIAL_CANDIDATES:
        if model not in base.index or model not in wide.index:
            continue
        b = base.loc[model]
        s2 = float(wide.loc[model, 2.0]); s5 = float(wide.loc[model, 5.0])
        eligible = (int(b.trades) >= min_trades and float(b.total_return) > 0 and
                    float(b.max_drawdown) > -0.05 and s2 > 0 and s5 > 0)
        rows.append(dict(model=model, trades=int(b.trades), total_return=float(b.total_return),
                         max_drawdown=float(b.max_drawdown), stress_2bps_return=s2,
                         stress_5bps_return=s5, eligible=bool(eligible)))
    audit = pd.DataFrame(rows)
    eligible = audit.loc[audit.eligible]
    if eligible.empty:
        return "CASH", audit
    # Règle de sélection figée : meilleur rendement sous le stress le plus sévère,
    # puis rendement de base, puis drawdown. Ce classement est uniquement pré-test.
    chosen = eligible.sort_values(["stress_5bps_return", "total_return", "max_drawdown"],
                                  ascending=[False, False, False], kind="stable").iloc[0].model
    return str(chosen), audit


def research(snapshot, cfg, timeframe, test_start, test_end, out):
    import re
    if not all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) for value in (test_start, test_end)):
        raise ValueError("Dates au format YYYY-MM-DD.")
    pd.Timestamp(test_start); pd.Timestamp(test_end)
    if test_end <= test_start:
        raise ValueError("La borne test_end exclusive doit suivre test_start.")
    expected_test_sessions = snapshot.schedule.index[(snapshot.schedule.index >= test_start) &
                                                     (snapshot.schedule.index < test_end)].tolist()
    if len(expected_test_sessions) < 5:
        raise ValueError("Prévoir au moins 5 séances finales présentes dans le calendrier du snapshot.")

    past = slice_snapshot(snapshot, test_start)
    bars = completed_bars(past, timeframe, asof=past.schedule.MarketClose.max(), require_complete=True)
    features = FeatureEngine().build(bars)
    labels = build_labels(bars, cfg.research, cfg.risk)
    sessions = bars.Session.unique()
    forecasts, audits = [], []
    for first in range(cfg.research.train_sessions, len(sessions), cfg.research.fold_sessions):
        days = sessions[first:first + cfg.research.fold_sessions]
        boundary = bars.loc[bars.Session == days[0], "SessionOpen"].iloc[0]
        try:
            mask, audit = eligible_train(bars, features.values, labels, boundary, timeframe, cfg.research)
        except ValueError as exc:
            if str(exc).startswith("Entraînement insuffisant"):
                audits.append(dict(prediction_boundary=boundary.isoformat(), skipped=str(exc)))
                continue
            raise
        ix = bars.index[bars.Session.isin(days) & features.values[GROUPS["D"]].notna().all(axis=1)]
        if not len(ix):
            continue
        audit["prediction_last"] = ix[-1].isoformat(); audits.append(audit)
        print(f"Walk-forward {days[0]} → {days[-1]} : {int(mask.sum())} lignes train", flush=True)
        ml = pd.concat([fit_predict(m, features.values, labels, mask, ix, cfg.research) for m in MODEL_IDS], axis=1)
        rules = rule_predictions(bars, features.values, ix)
        forecasts.append(pd.concat([ml, rules], axis=1))
    if not forecasts:
        raise ValueError("Historique insuffisant pour un walk-forward. Fournir davantage de séances ou utiliser 5 min.")

    predictions = pd.concat(forecasts).sort_index()
    path = make_output(out)
    table, stress = export_results(path, past, bars, features, labels, predictions, audits, cfg, timeframe,
                                   "walk-forward avant test")
    official, selection_audit = _choose_official(table, stress, bars.loc[predictions.index, "Session"].nunique())
    selection_audit.to_csv(path / "selection_audit.csv", index=False)
    lock = dict(version=9, symbol=snapshot.symbol, timeframe=timeframe, test_start=test_start,
                test_end=test_end, expected_test_sessions=expected_test_sessions,
                official_model=official,
                selection=("filtre pré-test: >= min trades, rendement net > 0, drawdown > -5%, "
                           "rendement > 0 avec +2/+5 bps de slippage par côté; sinon CASH"),
                config=cfg.to_dict(), feature_groups=GROUPS, model_ids=MODEL_IDS, rule_ids=RULE_IDS,
                code_hashes=code_hashes(), versions=versions(), pretest_fingerprint=fingerprint(snapshot, test_start),
                source=snapshot.metadata, synthetic=bool(snapshot.metadata.get("synthetic")),
                created_at=datetime.now(timezone.utc).isoformat(),
                threshold_policy=f"{cfg.research.threshold:.2f} fixé avant le test; aucune optimisation sur le test",
                caveat="Les scores de modèles ne sont pas des probabilités validées.")
    lock["integrity"] = canonical_hash(lock)
    (path / "protocol_lock.json").write_text(json.dumps(lock, indent=2, ensure_ascii=False), encoding="utf-8")
    return table, lock


def evaluate_final(snapshot, lock_path, out):
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    payload = {k: v for k, v in lock.items() if k != "integrity"}
    if canonical_hash(payload) != lock.get("integrity"):
        raise ValueError("Verrou modifié ou corrompu.")
    if lock.get("version") != 9:
        raise ValueError("Ce runner final exige un verrou V9.")
    if lock["code_hashes"] != code_hashes() or lock["versions"] != versions():
        raise ValueError("Le code ou les versions ont changé depuis la recherche. Refaire un protocole avant test.")
    if lock["symbol"] != snapshot.symbol or lock["pretest_fingerprint"] != fingerprint(snapshot, lock["test_start"]):
        raise ValueError("Le symbole ou les données pré-test ont changé : utiliser le même snapshot.")
    if lock["synthetic"] != bool(snapshot.metadata.get("synthetic")):
        raise ValueError("Métadonnées synthétiques incohérentes.")

    raw = lock["config"]
    cfg = AppConfig(symbols=tuple(raw["symbols"]), calendar=raw["calendar"], currency=raw["currency"],
                    risk=RiskConfig(**raw["risk"]), research=ResearchConfig(**raw["research"]))
    limited = slice_snapshot(snapshot, lock["test_end"])
    bars = completed_bars(limited, lock["timeframe"], asof=limited.schedule.MarketClose.max(), require_complete=True)
    actual = bars.loc[bars.Session >= lock["test_start"], "Session"].unique().tolist()
    if actual != lock["expected_test_sessions"]:
        raise ValueError("Séances finales manquantes : la fenêtre figée doit être entièrement présente.")
    if not (bars.Session >= lock["test_start"]).any():
        raise ValueError("Aucune donnée pour la période finale figée.")

    features = FeatureEngine().build(bars)
    labels = build_labels(bars, cfg.research, cfg.risk)
    boundary = bars.loc[bars.Session >= lock["test_start"], "SessionOpen"].iloc[0]
    mask, audit = eligible_train(bars, features.values, labels, boundary, lock["timeframe"], cfg.research)
    ix = bars.index[(bars.Session >= lock["test_start"]) & features.values[GROUPS["D"]].notna().all(axis=1)]
    ml = pd.concat([fit_predict(m, features.values, labels, mask, ix, cfg.research) for m in MODEL_IDS], axis=1)
    predictions = pd.concat([ml, rule_predictions(bars, features.values, ix)], axis=1)
    path = make_output(out)
    table, _ = export_results(path, limited, bars, features, labels, predictions, [audit], cfg,
                              lock["timeframe"], "test final figé")
    table["official"] = table.model == lock["official_model"]
    table.to_csv(path / "metrics.csv", index=False)
    (path / "protocol_lock_used.json").write_text(json.dumps(lock, indent=2, ensure_ascii=False), encoding="utf-8")
    return table, lock
