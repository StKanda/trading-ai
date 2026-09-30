from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd

from .config import AppConfig
from .data.market import Snapshot, completed_bars, utc
from .data.providers import CSVProvider, YahooProvider, write_snapshot
from .features.engine import FeatureEngine, build_labels, GROUPS, feature_columns_for_variant
from .deep_learning.utility import build_utility_labels
from .deep_learning.economic import build_economic_labels
from .research.models import MODEL_IDS, eligible_train, fit_predict
from .research.rules import RULE_IDS, rule_predictions
from .research.backtest import simulate, reference_curve
from .research.metrics import performance, classification
from .research.walkforward import make_output, cash_curve, bootstrap_daily
from .progress import TerminalProgress


def _banner(text: str) -> None:
    print(f"\n=== {text} ===", flush=True)


def _parse_dl_model(token: str) -> tuple[str, int]:
    raw = token.strip().upper()
    if "_" not in raw:
        raise ValueError(f"Modèle DL invalide: {token}. Utiliser MLP_12, MLP_24, TCN_12 ou TCN_24.")
    architecture, context = raw.split("_", 1)
    if architecture not in {"MLP", "TCN"}:
        raise ValueError(f"Architecture DL inconnue: {architecture}")
    context_i = int(context)
    if context_i not in {12, 24, 48, 78}:
        raise ValueError("Contexte DL supporté: 12, 24, 48 ou 78 bougies.")
    return architecture.lower(), context_i




def _closed_session_snapshot(snapshot: Snapshot) -> Snapshot:
    """Retire uniquement les séances de fin encore ouvertes à l'instant de capture.

    Un snapshot Yahoo pris pendant la séance contient légitimement la séance courante
    partielle. Le backtest, lui, n'accepte que des séances entièrement closes.
    Les éventuels trous dans les séances déjà closes restent des erreurs et ne sont
    jamais masqués.
    """
    captured = snapshot.metadata.get("fetched_at")
    asof = utc(captured) if captured else pd.Timestamp.now(tz="UTC")
    closed_schedule = snapshot.schedule.loc[snapshot.schedule.MarketClose <= asof].copy()
    if closed_schedule.empty:
        raise ValueError("Aucune séance entièrement clôturée n'est disponible dans ce snapshot.")
    last_close = closed_schedule.MarketClose.iloc[-1]
    first_open = closed_schedule.MarketOpen.iloc[0]
    prices = snapshot.prices.loc[(snapshot.prices.index >= first_open) & (snapshot.prices.index < last_close)].copy()
    if prices.empty:
        raise ValueError("Aucun prix appartenant aux séances entièrement clôturées.")
    removed = len(snapshot.schedule) - len(closed_schedule)
    if removed:
        print(f"Snapshot pris en séance: {removed} séance(s) de fin encore ouverte(s) exclue(s) du backtest.", flush=True)
    meta = snapshot.metadata.copy()
    meta["backtest_closed_sessions_only"] = True
    meta["backtest_asof"] = asof.isoformat()
    return Snapshot(snapshot.symbol, prices, closed_schedule, meta)

def _subset_sessions(bars: pd.DataFrame, max_sessions: int | None) -> pd.DataFrame:
    if not max_sessions:
        return bars
    sessions = list(pd.unique(bars.Session))
    keep = sessions[-int(max_sessions):]
    out = bars.loc[bars.Session.isin(keep)].copy()
    out.attrs.update(bars.attrs)
    return out


def _walkforward_fold_starts(session_count: int, train_sessions: int, fold_sessions: int,
                             max_folds: int | None = None):
    if max_folds is not None and max_folds < 1:
        raise ValueError("max_folds doit être supérieur ou égal à 1.")
    starts = range(train_sessions, session_count, fold_sessions)
    return starts if max_folds is None else starts[:max_folds]


def _dl_fold_predict(features, labels, train_mask, valid_mask, model_name, seeds, epochs, patience,
                     checkpoint_root: Path, fold_no: int, feature_columns=None, progress=None,
                     progress_context=""):
    try:
        from .deep_learning.dataset import build_sequence_bundle
        from .deep_learning.train import train_classifier
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise ImportError(
                "PyTorch n'est pas installé. Installer: python -m pip install -r requirements_dl.txt"
            ) from exc
        raise

    architecture, context = _parse_dl_model(model_name)
    bundle = build_sequence_bundle(
        features, labels, train_mask, valid_mask,
        context_bars=context, columns=feature_columns or GROUPS["D"]
    )
    per_seed = []
    training_rows = []
    for seed_number, seed in enumerate(seeds, start=1):
        ckpt = checkpoint_root / model_name / f"fold_{fold_no:02d}_seed_{seed}.pt"
        print(f"      {model_name} · seed {seed} · train={len(bundle.x_train)} valid={len(bundle.x_valid)}", flush=True)
        result = train_classifier(
            bundle,
            architecture=architecture,
            hidden_size=24,
            max_epochs=int(epochs),
            patience=int(patience),
            seed=int(seed),
            checkpoint=ckpt,
            progress_callback=(progress.epoch_callback(
                f"{progress_context}{model_name} seed {seed_number}/{len(seeds)}", epochs
            ) if progress is not None else None),
        )
        per_seed.append(result.validation_scores.astype(float))
        training_rows.append({
            "fold": fold_no,
            "model": model_name,
            "architecture": architecture,
            "context": context,
            "seed": int(seed),
            "best_epoch": int(result.best_epoch),
            "train_loss": float(result.train_loss),
            "valid_loss": float(result.valid_loss),
            "device": result.device,
            "n_parameters": int(result.n_parameters),
            "train_sequences": int(len(bundle.x_train)),
            "valid_sequences": int(len(bundle.x_valid)),
            "checkpoint": result.checkpoint,
        })
    ensemble = np.mean(np.vstack(per_seed), axis=0)
    return pd.Series(ensemble, index=bundle.valid_index, name=model_name), training_rows




def _sigmoid_scaled(x, scale):
    scale = max(float(abs(scale)), 1e-4)
    z = np.clip(np.asarray(x, dtype=float) / scale, -12, 12)
    return 1.0 / (1.0 + np.exp(-z))


def _utility_setup_score(frame: pd.DataFrame, reg_std) -> pd.Series:
    """Score borné [0,1] combinant classification et utilité prédite.

    Les échelles de régression proviennent uniquement du train interne. Ce score
    n'est pas une probabilité calibrée.
    """
    reg_std = np.asarray(reg_std, dtype=float)
    mfe_s, mae_s, _n15_s, n30_s, n60_s = reg_std
    p30 = frame["TargetBeforeStop_30"].clip(0, 1)
    p60 = frame["TargetBeforeStop_60"].clip(0, 1)
    net30 = _sigmoid_scaled(frame["NetReturn_30"], n30_s)
    net60 = _sigmoid_scaled(frame["NetReturn_60"], n60_s)
    mfe = _sigmoid_scaled(frame["MFE_30"], mfe_s)
    mae = _sigmoid_scaled(frame["MAE_30"], mae_s)  # moins négatif => meilleur
    score = 0.25*p30 + 0.30*p60 + 0.10*net30 + 0.20*net60 + 0.10*mfe + 0.05*mae
    return pd.Series(np.clip(score, 0, 1), index=frame.index, name="SetupScore")


def _choose_validation_threshold(score: pd.Series, labels: pd.DataFrame):
    """Choix borné du seuil sur validation interne uniquement.

    Objectif: rendement net moyen 60 min positif, avec au moins 3 observations.
    Si aucun seuil n'a une utilité positive, le fold choisit NO_TRADE.
    """
    joined = pd.DataFrame({"score": score}).join(labels[["TargetBeforeStop_60", "NetReturn_60"]], how="inner").dropna()
    if len(joined) < 5:
        return None, {"validation_rows": int(len(joined)), "reason": "insufficient_validation_rows"}
    grid = np.array([0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75], dtype=float)
    best = None
    for thr in grid:
        sel = joined.score >= thr
        n = int(sel.sum())
        if n < 3:
            continue
        mean_net = float(joined.loc[sel, "NetReturn_60"].mean())
        hit = float(joined.loc[sel, "TargetBeforeStop_60"].mean())
        # Rendement net d'abord; hit-rate sert seulement à départager.
        objective = mean_net + 0.00025 * hit
        row = (objective, mean_net, hit, n, float(thr))
        if best is None or row > best:
            best = row
    if best is None or best[1] <= 0:
        return None, {
            "validation_rows": int(len(joined)), "validation_selected": 0,
            "validation_mean_net60": float("nan"), "validation_hit60": float("nan"),
            "reason": "no_positive_validation_utility",
        }
    _, mean_net, hit, n, thr = best
    return thr, {
        "validation_rows": int(len(joined)), "validation_selected": int(n),
        "validation_mean_net60": mean_net, "validation_hit60": hit,
        "reason": "selected_on_inner_validation",
    }


def _recenter_threshold(score: pd.Series, selected_threshold: float | None, output_threshold: float) -> pd.Series:
    if selected_threshold is None:
        return pd.Series(0.0, index=score.index, name=score.name)
    eps = 1e-6
    x = np.clip(score.astype(float).to_numpy(), eps, 1-eps)
    src = float(np.clip(selected_threshold, eps, 1-eps))
    dst = float(np.clip(output_threshold, eps, 1-eps))
    logit = lambda z: np.log(z/(1-z))
    shifted = 1.0/(1.0 + np.exp(-(logit(x) - logit(src) + logit(dst))))
    return pd.Series(shifted, index=score.index, name=score.name)


def _candidate_setup_mask(bars: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
    """Filtre causal de setups candidats avant ranking DL.

    Le filtre est volontairement simple et figé: règles existantes + contexte
    support/VWAP/W. Il ne regarde aucune cible future et ne remplace pas le DL.
    """
    idx = bars.index
    f = features.reindex(idx)
    rules = rule_predictions(bars, features, idx).max(axis=1).fillna(0).gt(0)
    tradable = f["SessionProgress"].between(0.08, 0.82)
    liquid = f["RelativeVolumeSlot"].fillna(0).ge(0.70)
    support_bounce = f["NearSupport1Pct"].fillna(0).gt(0) & f["Return3"].fillna(0).gt(0)
    w_context = (f["WSignal"].fillna(0).gt(0) | f["WRecent10"].fillna(0).gt(0))
    vwap_reclaim_context = f["DistVWAP"].abs().le(0.0025) & f["Return3"].fillna(0).gt(0)
    momentum_context = f["Return3"].fillna(0).gt(0) & f["Return10"].fillna(0).gt(0) & f["DistVWAP"].fillna(0).gt(0)
    candidate = tradable & liquid & (rules | support_bounce | w_context | vwap_reclaim_context | momentum_context)
    return candidate.astype(bool).rename("CandidateSetup")


def _choose_validation_threshold_economic(score: pd.Series, bars: pd.DataFrame, features: pd.DataFrame, risk, research):
    """Choisit le seuil via le vrai simulateur et conserve tout le grid-search.

    La sélection reste strictement sur validation interne. V11.3.2 ajoute un
    diagnostic exhaustif par seuil afin de distinguer absence d'opportunités,
    mauvais ranking DL et garde-fous de robustesse trop contraignants.
    """
    score = score.dropna().sort_index()
    grid_rows = []
    if len(score) < 5:
        return None, {"validation_rows": int(len(score)), "reason": "insufficient_validation_rows"}, grid_rows
    sessions = pd.unique(bars.loc[bars.index.isin(score.index), "Session"])
    vb = bars.loc[bars.Session.isin(sessions)].copy()
    if vb.empty:
        return None, {"validation_rows": int(len(score)), "reason": "empty_validation_bars"}, grid_rows
    grid = np.array([0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80], dtype=float)
    best = None
    for thr in grid:
        selected = int((score >= thr).sum())
        signal_rate = selected / max(len(score), 1)
        base = {
            "threshold_raw": float(thr), "validation_rows": int(len(score)),
            "selected_signals": selected, "signal_rate": float(signal_rate),
            "trades": 0, "sessions_with_trades": 0, "total_return": np.nan,
            "max_drawdown": np.nan, "mean_trade_pnl": np.nan, "win_rate": np.nan,
            "profit_factor": np.nan, "top1_abs_share": np.nan, "eligible_for_selection": False,
            "rejection_reason": "",
        }
        if selected < 2:
            base["rejection_reason"] = "too_few_selected_signals"
            grid_rows.append(base); continue
        if signal_rate > 0.60:
            base["rejection_reason"] = "signal_rate_above_60pct"
            grid_rows.append(base); continue
        adjusted = _recenter_threshold(score, thr, research.threshold)
        sim = simulate(vb, adjusted, features, risk, research)
        perf = performance(sim, risk.initial_capital)
        ntr = int(perf.get("trades", 0))
        pnl = pd.to_numeric(sim.trades.get("pnl", pd.Series(dtype=float)), errors="coerce").dropna()
        mean_pnl = float(pnl.mean()) if len(pnl) else np.nan
        win_rate = float((pnl > 0).mean()) if len(pnl) else np.nan
        wins = float(pnl[pnl > 0].sum()) if len(pnl) else 0.0
        losses = float(-pnl[pnl < 0].sum()) if len(pnl) else 0.0
        pf = float(wins / losses) if losses > 0 else (np.inf if wins > 0 else np.nan)
        total_return = float(perf.get("total_return", np.nan))
        max_dd = float(perf.get("max_drawdown", np.nan))
        entry_times = pd.to_datetime(sim.trades.get("entry_time", pd.Series(dtype=object)), utc=True, errors="coerce")
        session_by_start = pd.Series(vb.Session.to_numpy(), index=pd.DatetimeIndex(pd.to_datetime(vb.Start, utc=True)))
        trade_sessions = entry_times.map(session_by_start).dropna() if len(entry_times) else pd.Series(dtype=object)
        sessions_with_trades = int(pd.Series(trade_sessions).nunique()) if len(trade_sessions) else 0
        abs_pnl = pnl.abs()
        top1_abs_share = float(abs_pnl.max() / abs_pnl.sum()) if len(abs_pnl) and float(abs_pnl.sum()) > 0 else np.nan
        base.update({
            "trades": ntr, "sessions_with_trades": sessions_with_trades,
            "total_return": total_return, "max_drawdown": max_dd,
            "mean_trade_pnl": mean_pnl, "win_rate": win_rate,
            "profit_factor": pf, "top1_abs_share": top1_abs_share,
        })
        reason = None
        if ntr < int(research.min_validation_trades):
            reason = "insufficient_validation_trades"
        elif not np.isfinite(total_return) or total_return <= 0:
            reason = "non_positive_validation_return"
        elif not np.isfinite(mean_pnl) or mean_pnl <= 0:
            reason = "non_positive_expectancy"
        elif sessions_with_trades < int(research.min_validation_sessions_with_trades):
            reason = "insufficient_validation_sessions"
        elif not np.isfinite(top1_abs_share) or top1_abs_share > float(research.max_validation_top1_abs_share):
            reason = "excessive_top1_concentration"
        elif ntr / max(len(sessions), 1) > 3.0:
            reason = "too_many_trades_per_session"
        elif max_dd <= -0.03:
            reason = "validation_drawdown_below_limit"
        if reason is not None:
            base["rejection_reason"] = reason
            grid_rows.append(base); continue
        base["eligible_for_selection"] = True
        base["rejection_reason"] = "accepted_candidate"
        grid_rows.append(base)
        objective = total_return - 0.20 * abs(max_dd) - 0.001 * top1_abs_share
        row = (objective, total_return, mean_pnl, -(ntr / max(len(sessions), 1)), float(thr), ntr,
               max_dd, win_rate, selected, signal_rate, sessions_with_trades, top1_abs_share)
        if best is None or row > best:
            best = row
    if best is None:
        return None, {
            "validation_rows": int(len(score)), "validation_selected": 0,
            "validation_trades": 0, "validation_sessions_with_trades": 0,
            "validation_total_return": float("nan"), "validation_max_drawdown": float("nan"),
            "validation_mean_trade_pnl": float("nan"), "validation_win_rate": float("nan"),
            "validation_signal_rate": float("nan"), "validation_top1_abs_share": float("nan"),
            "reason": "no_positive_validation_backtest",
        }, grid_rows
    _, total_return, mean_pnl, _neg_tps, thr, ntr, max_dd, win_rate, selected, signal_rate, sessions_with_trades, top1_abs_share = best
    for row in grid_rows:
        row["selected_final"] = bool(row["threshold_raw"] == float(thr))
    return thr, {
        "validation_rows": int(len(score)), "validation_selected": int(selected),
        "validation_trades": int(ntr), "validation_sessions_with_trades": int(sessions_with_trades),
        "validation_total_return": total_return, "validation_max_drawdown": max_dd,
        "validation_mean_trade_pnl": mean_pnl, "validation_win_rate": win_rate,
        "validation_signal_rate": signal_rate, "validation_top1_abs_share": top1_abs_share,
        "reason": "selected_by_validation_backtest_robust",
    }, grid_rows


def _inner_masks_for_utility(bars, features, utility_labels, outer_train_mask, timeframe, research,
                             candidate_mask=None, feature_columns=None):
    sessions = list(pd.unique(bars.loc[outer_train_mask, "Session"]))
    if len(sessions) < 6:
        raise ValueError("Pas assez de séances historiques pour une validation interne DL.")
    n_valid_sessions = max(2, min(4, len(sessions)//4))
    valid_sessions = sessions[-n_valid_sessions:]
    boundary = bars.loc[bars.Session == valid_sessions[0], "SessionOpen"].iloc[0]
    first_pos = bars.index.searchsorted(boundary, side="left")
    gap = 60 // timeframe + 1 + research.extra_embargo_bars
    complete = utility_labels[[
        "TargetBeforeStop_15", "TargetBeforeStop_30", "TargetBeforeStop_60",
        "MFE_30", "MAE_30", "NetReturn_15", "NetReturn_30", "NetReturn_60"
    ]].notna().all(axis=1)
    columns = feature_columns or GROUPS["D"]
    ready = features[columns].notna().all(axis=1) & complete
    if candidate_mask is not None:
        ready = ready & pd.Series(candidate_mask, index=bars.index).astype(bool)
    pos = np.arange(len(bars))
    inner_train = np.asarray(outer_train_mask, dtype=bool) & (pos < first_pos-gap) & ready.to_numpy() & (utility_labels["LabelEnd_60"] < boundary).to_numpy()
    inner_valid = np.asarray(outer_train_mask, dtype=bool) & bars.Session.isin(valid_sessions).to_numpy() & ready.to_numpy()
    if int(inner_train.sum()) < max(100, research.min_train_rows//2):
        raise ValueError(f"Train interne DL insuffisant: {int(inner_train.sum())} lignes.")
    if int(inner_valid.sum()) < 20:
        raise ValueError(f"Validation interne DL insuffisante: {int(inner_valid.sum())} lignes.")
    return inner_train, inner_valid, valid_sessions


def _dl_fold_predict_utility(bars, features, utility_labels, train_mask, test_mask, model_name, seeds, epochs, patience,
                             checkpoint_root: Path, fold_no: int, timeframe: int, research, risk, candidate_mask,
                             feature_columns=None, progress=None, progress_context=""):
    try:
        from .deep_learning.utility import build_utility_sequence_bundle, transform_utility_sequences
        from .deep_learning.multitask import train_multitask
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise ImportError("PyTorch n'est pas installé. Installer: python -m pip install -r requirements_dl.txt") from exc
        raise

    architecture, context = _parse_dl_model(model_name)
    inner_train, inner_valid, inner_valid_sessions = _inner_masks_for_utility(
        bars, features, utility_labels, train_mask, timeframe, research, candidate_mask=candidate_mask,
        feature_columns=feature_columns,
    )
    bundle = build_utility_sequence_bundle(
        features, utility_labels, inner_train, inner_valid,
        context_bars=context, columns=feature_columns or GROUPS["D"]
    )
    candidate_test = np.asarray(test_mask, dtype=bool) & pd.Series(candidate_mask, index=bars.index).to_numpy(dtype=bool)
    x_test, test_index = transform_utility_sequences(features, candidate_test, bundle)
    if len(x_test) < 5:
        raise ValueError("Séquences OOS insuffisantes pour le DL utility.")

    valid_frames, test_frames, training_rows = [], [], []
    for seed_number, seed in enumerate(seeds, start=1):
        ckpt = checkpoint_root / model_name / f"fold_{fold_no:02d}_seed_{seed}.pt"
        print(f"      {model_name} utility · seed {seed} · train={len(bundle.x_train)} inner-valid={len(bundle.x_valid)} test={len(x_test)}", flush=True)
        result = train_multitask(
            bundle, architecture=architecture, hidden_size=24, max_epochs=int(epochs),
            patience=int(patience), seed=int(seed), checkpoint=ckpt,
            x_test=x_test, test_index=test_index,
            progress_callback=(progress.epoch_callback(
                f"{progress_context}{model_name} seed {seed_number}/{len(seeds)}", epochs
            ) if progress is not None else None),
        )
        valid_frames.append(result.validation_frame.astype(float))
        test_frames.append(result.test_frame.astype(float))
        training_rows.append({
            "fold": fold_no, "model": model_name, "objective": "candidate_ranker_v11_3_2",
            "architecture": architecture, "context": context, "seed": int(seed),
            "best_epoch": int(result.best_epoch), "train_loss": float(result.train_loss),
            "valid_loss": float(result.valid_loss), "device": result.device,
            "n_parameters": int(result.n_parameters), "train_sequences": int(len(bundle.x_train)),
            "valid_sequences": int(len(bundle.x_valid)), "test_sequences": int(len(x_test)),
            "inner_valid_sessions": ",".join(map(str, inner_valid_sessions)),
            "pos_weight_15": float(result.pos_weight[0]), "pos_weight_30": float(result.pos_weight[1]),
            "pos_weight_60": float(result.pos_weight[2]), "checkpoint": result.checkpoint,
        })

    valid_avg = sum(valid_frames) / float(len(valid_frames))
    test_avg = sum(test_frames) / float(len(test_frames))
    valid_setup = _utility_setup_score(valid_avg, bundle.reg_std)
    selected_threshold, selection, threshold_grid = _choose_validation_threshold_economic(valid_setup, bars, features, risk, research)
    test_setup_raw = _utility_setup_score(test_avg, bundle.reg_std).rename(model_name)
    score = _recenter_threshold(test_setup_raw, selected_threshold, research.threshold).rename(model_name)
    diag = test_avg.add_prefix(model_name + "__")
    diag[model_name + "__SetupScoreRaw"] = test_setup_raw
    diag[model_name + "__SetupScoreAdjusted"] = score
    selection_row = {
        "fold": fold_no, "model": model_name, "selected_threshold": selected_threshold,
        "applied_oos_threshold_raw": selected_threshold,
        "backtester_threshold_adjusted": float(research.threshold),
        "threshold_recentered": bool(selected_threshold is not None),
        "min_validation_trades_required": int(research.min_validation_trades),
        "min_validation_sessions_required": int(research.min_validation_sessions_with_trades),
        "max_validation_top1_abs_share_allowed": float(research.max_validation_top1_abs_share),
        "inner_valid_sessions": ",".join(map(str, inner_valid_sessions)),
        **selection,
    }
    print(f"      seuil validation {model_name}: {selected_threshold if selected_threshold is not None else 'NO_TRADE'} · {selection['reason']}", flush=True)
    for r in threshold_grid:
        r.update({"fold": fold_no, "model": model_name})
    return score, diag, training_rows, selection_row, threshold_grid



def _economic_score(frame: pd.DataFrame, pnl_scale: float) -> pd.Series:
    """Score [0,1] monotone de la valeur économique prédite.

    0.50 correspond à ExpectedTradeValue = 0. Ce score n'est pas une probabilité
    calibrée; le classement est uniquement piloté par le PnL prédit.
    """
    scale = max(abs(float(pnl_scale)), 1e-6)
    z = np.clip(frame["RealizedPnL"].astype(float).to_numpy() / scale, -12, 12)
    return pd.Series(1.0/(1.0+np.exp(-z)), index=frame.index, name="EconomicScore")


def _inner_masks_for_economic(bars, features, economic_labels, outer_train_mask, timeframe, research,
                               candidate_mask, economic_validation_sessions=None, feature_columns=None):
    sessions = list(pd.unique(bars.loc[outer_train_mask, "Session"]))
    if economic_validation_sessions is None and len(sessions) < 6:
        raise ValueError("Pas assez de séances historiques pour une validation interne DL économique.")
    if economic_validation_sessions is None:
        n_valid_sessions = max(2, min(4, len(sessions)//4))
    else:
        n_valid_sessions = int(economic_validation_sessions)
        if n_valid_sessions < 2:
            raise ValueError("economic_validation_sessions doit être supérieur ou égal à 2.")
        if len(sessions) < n_valid_sessions:
            raise ValueError(
                f"Inner-validation économique demandée: {n_valid_sessions} séances, "
                f"mais seulement {len(sessions)} séances admissibles dans le train externe."
            )
    valid_sessions = sessions[-n_valid_sessions:]
    boundary = bars.loc[bars.Session == valid_sessions[0], "SessionOpen"].iloc[0]
    first_pos = bars.index.searchsorted(boundary, side="left")
    gap = research.horizon_minutes // timeframe + 1 + research.extra_embargo_bars
    cols = ["ProfitableAfterCosts","StopHit","TargetHit","TimeExit","RealizedPnL","RealizedReturn","MFE","MAE"]
    complete = economic_labels[cols].notna().all(axis=1)
    columns = feature_columns or GROUPS["D"]
    ready = features[columns].notna().all(axis=1) & complete & pd.Series(candidate_mask, index=bars.index).astype(bool)
    pos = np.arange(len(bars))
    inner_train = np.asarray(outer_train_mask, dtype=bool) & (pos < first_pos-gap) & ready.to_numpy() & (economic_labels["LabelEnd"] < boundary).to_numpy()
    inner_valid = np.asarray(outer_train_mask, dtype=bool) & bars.Session.isin(valid_sessions).to_numpy() & ready.to_numpy()
    if int(inner_train.sum()) < max(60, research.min_train_rows//4):
        detail = f"Train interne économique insuffisant: {int(inner_train.sum())} lignes."
        if economic_validation_sessions is not None:
            detail += f" Impossible d'entraîner avec --economic-validation-sessions={n_valid_sessions}."
        raise ValueError(detail)
    if int(inner_valid.sum()) < 20:
        detail = f"Validation interne économique insuffisante: {int(inner_valid.sum())} lignes."
        if economic_validation_sessions is not None:
            detail += f" Impossible d'utiliser --economic-validation-sessions={n_valid_sessions}."
        raise ValueError(detail)
    return inner_train, inner_valid, valid_sessions


def _dl_fold_predict_economic(bars, features, economic_labels, train_mask, test_mask, model_name, seeds, epochs, patience,
                              checkpoint_root: Path, fold_no: int, timeframe: int, research, risk, candidate_mask,
                              economic_validation_sessions=None, feature_columns=None, progress=None,
                              progress_context=""):
    try:
        from .deep_learning.economic import build_economic_sequence_bundle, transform_economic_sequences
        from .deep_learning.economic_train import train_economic
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise ImportError("PyTorch n'est pas installé. Installer: python -m pip install -r requirements_dl.txt") from exc
        raise
    architecture, context = _parse_dl_model(model_name)
    inner_train, inner_valid, inner_valid_sessions = _inner_masks_for_economic(
        bars, features, economic_labels, train_mask, timeframe, research, candidate_mask,
        economic_validation_sessions=economic_validation_sessions,
        feature_columns=feature_columns,
    )
    try:
        bundle = build_economic_sequence_bundle(features, economic_labels, inner_train, inner_valid,
                                                context_bars=context, columns=feature_columns or GROUPS["D"])
    except ValueError as exc:
        if economic_validation_sessions is not None:
            raise ValueError(
                f"Données insuffisantes avec --economic-validation-sessions={economic_validation_sessions}: {exc}"
            ) from exc
        raise
    candidate_test = np.asarray(test_mask, dtype=bool) & pd.Series(candidate_mask, index=bars.index).to_numpy(dtype=bool)
    x_test, test_index = transform_economic_sequences(features, candidate_test, bundle)
    if len(x_test) < 5:
        raise ValueError("Séquences OOS insuffisantes pour le DL économique.")
    valid_frames=[]; test_frames=[]; training_rows=[]
    for seed_number, seed in enumerate(seeds, start=1):
        ckpt = checkpoint_root / model_name / f"fold_{fold_no:02d}_seed_{seed}.pt"
        print(
            f"      {model_name} economic · seed {seed} · train={len(bundle.x_train)} "
            f"inner-valid={len(bundle.x_valid)} séquences/{len(inner_valid_sessions)} séances "
            f"test={len(x_test)}",
            flush=True,
        )
        result = train_economic(bundle, architecture=architecture, hidden_size=24, max_epochs=int(epochs),
                                patience=int(patience), seed=int(seed), checkpoint=ckpt,
                                x_test=x_test, test_index=test_index,
                                progress_callback=(progress.epoch_callback(
                                    f"{progress_context}{model_name} seed {seed_number}/{len(seeds)}", epochs
                                ) if progress is not None else None))
        valid_frames.append(result.validation_frame.astype(float)); test_frames.append(result.test_frame.astype(float))
        training_rows.append({
            "fold": fold_no, "model": model_name, "objective": "direct_economic_v11_4",
            "architecture": architecture, "context": context, "seed": int(seed),
            "best_epoch": int(result.best_epoch), "train_loss": float(result.train_loss),
            "valid_loss": float(result.valid_loss), "device": result.device,
            "n_parameters": int(result.n_parameters), "train_sequences": int(len(bundle.x_train)),
            "valid_sequences": int(len(bundle.x_valid)), "test_sequences": int(len(x_test)),
            "inner_valid_session_count": int(len(inner_valid_sessions)),
            "inner_valid_sessions": ",".join(map(str, inner_valid_sessions)),
            "pos_weight_profitable": float(result.pos_weight[0]), "checkpoint": result.checkpoint,
        })
    valid_avg=sum(valid_frames)/float(len(valid_frames)); test_avg=sum(test_frames)/float(len(test_frames))
    pnl_scale=float(bundle.reg_std[0])
    valid_score=_economic_score(valid_avg,pnl_scale)
    selected_threshold, selection, threshold_grid = _choose_validation_threshold_economic(valid_score,bars,features,risk,research)
    test_score_raw=_economic_score(test_avg,pnl_scale).rename(model_name)
    score=_recenter_threshold(test_score_raw,selected_threshold,research.threshold).rename(model_name)
    diag=test_avg.add_prefix(model_name+"__")
    diag[model_name+"__ExpectedTradeValue"] = test_avg["RealizedPnL"]
    diag[model_name+"__EconomicScoreRaw"] = test_score_raw
    diag[model_name+"__EconomicScoreAdjusted"] = score
    selection_row={
        "fold":fold_no,"model":model_name,"selected_threshold":selected_threshold,
        "applied_oos_threshold_raw":selected_threshold,"backtester_threshold_adjusted":float(research.threshold),
        "threshold_recentered":bool(selected_threshold is not None),
        "min_validation_trades_required":int(research.min_validation_trades),
        "min_validation_sessions_required":int(research.min_validation_sessions_with_trades),
        "max_validation_top1_abs_share_allowed":float(research.max_validation_top1_abs_share),
        "inner_valid_session_count": int(len(inner_valid_sessions)),
        "inner_valid_sessions": ",".join(map(str,inner_valid_sessions)), **selection,
    }
    print(f"      seuil validation {model_name}: {selected_threshold if selected_threshold is not None else 'NO_TRADE'} · {selection['reason']}", flush=True)
    for r in threshold_grid: r.update({"fold":fold_no,"model":model_name})
    return score,diag,training_rows,selection_row,threshold_grid


def _economic_oos_metrics(labels: pd.DataFrame, diag: pd.DataFrame, model: str) -> dict:
    from sklearn.metrics import roc_auc_score, average_precision_score
    from scipy.stats import spearmanr
    score_col=model+"__EconomicScoreRaw"; pnl_col=model+"__ExpectedTradeValue"
    joined=pd.DataFrame({
        "y":labels["ProfitableAfterCosts"],"realized_pnl":labels["RealizedPnL"],
        "score":diag[score_col],"pred_pnl":diag[pnl_col],
    }).dropna()
    if joined.empty: return {"model":model,"rows":0}
    yy=joined.y.astype(int); ss=joined.score.astype(float)
    q=float(ss.quantile(.90)); top=joined.loc[ss>=q]
    rho=float(spearmanr(joined.realized_pnl,joined.pred_pnl).statistic) if len(joined)>2 else np.nan
    return {
        "model":model,"rows":int(len(joined)),"profitable_rate":float(yy.mean()),
        "rank_auc":float(roc_auc_score(yy,ss)) if yy.nunique()==2 else np.nan,
        "average_precision":float(average_precision_score(yy,ss)) if yy.nunique()==2 else np.nan,
        "pnl_spearman":rho,"predicted_pnl_mean":float(joined.pred_pnl.mean()),
        "realized_pnl_mean":float(joined.realized_pnl.mean()),"top_decile_rows":int(len(top)),
        "top_decile_profitable_rate":float(top.y.mean()) if len(top) else np.nan,
        "top_decile_realized_pnl_mean":float(top.realized_pnl.mean()) if len(top) else np.nan,
        "note":"ExpectedTradeValue = PnL prédit; score monotone, pas une probabilité calibrée",
    }

def _raw_rank_metrics(y: pd.Series, score: pd.Series, model: str) -> dict:
    """Métriques de ranking sur score brut OOS, indépendantes du gate NO_TRADE.

    Le score est un score de modèle, pas une probabilité calibrée.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score
    joined = pd.concat([y.rename("y"), score.rename("score")], axis=1).dropna()
    if joined.empty:
        return {"model": model, "rows": 0}
    yy = joined.y.astype(int)
    ss = joined.score.astype(float)
    two = yy.nunique() == 2
    q = float(ss.quantile(0.90)) if len(ss) else np.nan
    top = joined.loc[ss >= q] if np.isfinite(q) else joined.iloc[0:0]
    return {
        "model": model, "rows": int(len(joined)), "positive_rate": float(yy.mean()),
        "score_mean": float(ss.mean()), "score_std": float(ss.std(ddof=1)) if len(ss) > 1 else 0.0,
        "rank_auc": float(roc_auc_score(yy, ss)) if two else np.nan,
        "average_precision": float(average_precision_score(yy, ss)) if two else np.nan,
        "top_decile_rows": int(len(top)),
        "top_decile_positive_rate": float(top.y.mean()) if len(top) else np.nan,
        "note": "score brut OOS; pas une probabilité calibrée",
    }


def _candidate_oracle_diagnostic(evaluated: pd.DataFrame, candidate_score: pd.Series, features: pd.DataFrame,
                                 risk, research, progress_callback=None) -> pd.DataFrame:
    """Diagnostic futur-fuyant: résultat isolé de chaque setup candidat OOS.

    Strictement interdit comme signal de trading. Chaque candidat est simulé seul
    avec les mêmes règles d'exécution pour mesurer si l'univers candidat contient
    des trades économiquement positifs après coûts.
    """
    rows = []
    candidates = candidate_score[candidate_score >= research.threshold].index
    minutes = int((evaluated.index[0] - evaluated.Start.iloc[0]).total_seconds() / 60)
    horizon_rows = int(research.horizon_minutes // minutes) + 2
    for candidate_number, signal_time in enumerate(candidates, start=1):
        if progress_callback is not None:
            progress_callback({"completed": candidate_number, "total": len(candidates),
                               "detail": "setups diagnostics"})
        if signal_time not in evaluated.index:
            continue
        session = evaluated.loc[signal_time, "Session"]
        session_bars = evaluated.loc[evaluated.Session == session].copy()
        try:
            pos = int(session_bars.index.get_loc(signal_time))
        except (KeyError, TypeError):
            continue
        # Fenêtre minimale suffisante pour reproduire l'entrée t+1 et la sortie
        # stop/target/time-exit du trade isolé, sans rescanner toute la séance.
        sb = session_bars.iloc[pos:min(len(session_bars), pos + horizon_rows)].copy()
        if len(sb) < 2:
            sim = None
        else:
            isolated = pd.Series([1.0], index=pd.DatetimeIndex([signal_time]), name="CANDIDATE_ORACLE_DIAGNOSTIC")
            sim = simulate(sb, isolated, features, risk, research)
        if sim is None or sim.trades.empty:
            rows.append({
                "signal_time": signal_time, "Session": session, "eligible_trade": False,
                "pnl": np.nan, "profitable_after_costs": np.nan, "exit_reason": "no_entry",
                "diagnostic_only_future_leakage": True,
            })
            continue
        tr = sim.trades.iloc[0]
        pnl = float(tr["pnl"])
        rows.append({
            "signal_time": signal_time, "Session": session, "eligible_trade": True,
            "entry_time": tr["entry_time"], "exit_time": tr["exit_time"],
            "entry_price": float(tr["entry_price"]), "exit_price": float(tr["exit_price"]),
            "pnl": pnl, "return_on_initial_capital": pnl / float(risk.initial_capital),
            "profitable_after_costs": bool(pnl > 0), "exit_reason": tr["exit_reason"],
            "diagnostic_only_future_leakage": True,
        })
    if progress_callback is not None and len(candidates) == 0:
        progress_callback({"completed": 0, "total": 0, "detail": "aucun setup"})
    return pd.DataFrame(rows)

def _parse_date_bounds(start_date=None, end_date=None):
    try:
        start = pd.Timestamp(start_date).date() if start_date is not None else None
        end = pd.Timestamp(end_date).date() if end_date is not None else None
    except Exception as exc:
        raise ValueError("--start-date/--end-date doivent être au format YYYY-MM-DD.") from exc
    if start is not None and end is not None and start > end:
        raise ValueError("--start-date doit être antérieure ou égale à --end-date.")
    return start, end


def _slice_snapshot_date_range(snapshot, start_date=None, end_date=None):
    """V12.1.4: borne le snapshot AVANT la validation des séances.

    Ainsi une recherche 2025 n'est jamais bloquée par une anomalie historique de
    2016/2020 située hors de la période demandée.
    """
    start, end = _parse_date_bounds(start_date, end_date)
    if start is None and end is None:
        return snapshot
    schedule = snapshot.schedule.copy()
    days = pd.to_datetime(schedule.index.astype(str)).date
    mask = pd.Series(True, index=schedule.index)
    if start is not None:
        mask &= days >= start
    if end is not None:
        mask &= days <= end
    schedule = schedule.loc[mask].copy()
    if schedule.empty:
        raise ValueError("Aucune séance dans la période demandée.")
    lo = schedule.MarketOpen.iloc[0]
    hi = schedule.MarketClose.iloc[-1]
    prices = snapshot.prices.loc[(snapshot.prices.index >= lo) & (snapshot.prices.index < hi)].copy()
    if prices.empty:
        raise ValueError("Aucune donnée de prix dans la période demandée.")
    meta = dict(snapshot.metadata)
    meta["requested_start_date"] = str(start) if start is not None else None
    meta["requested_end_date"] = str(end) if end is not None else None
    return type(snapshot)(snapshot.symbol, prices, schedule, meta)


def _filter_date_range(bars: pd.DataFrame, start_date=None, end_date=None) -> pd.DataFrame:
    """Filtre les séances inclusivement selon YYYY-MM-DD, avant features/walk-forward."""
    if start_date is None and end_date is None:
        return bars
    start, end = _parse_date_bounds(start_date, end_date)
    session_dates = pd.to_datetime(bars["Session"]).dt.date
    mask = pd.Series(True, index=bars.index)
    if start is not None:
        mask &= session_dates >= start
    if end is not None:
        mask &= session_dates <= end
    filtered = bars.loc[mask].copy()
    if filtered.empty:
        raise ValueError("Aucune séance dans la période demandée.")
    return filtered


def unified_walkforward(snapshot, cfg: AppConfig, out_dir: Path, timeframe: int = 5,
                        dl_models=("MLP_12", "MLP_24", "TCN_12", "TCN_24"),
                        seeds=(42,), epochs=20, patience=4,
                        include_classic=True, include_rules=True,
                        max_sessions=None, dl_objective="direction", start_date=None, end_date=None,
                        max_folds: int | None = None, economic_validation_sessions: int | None = None,
                        feature_variant="baseline", progress=False):
    if timeframe not in (5, 10, 15):
        raise ValueError("Le moteur de backtest supporte 5, 10 ou 15 minutes.")
    mtf_columns = feature_columns_for_variant(feature_variant)
    if feature_variant != "baseline" and timeframe != 5:
        raise ValueError("Les variantes MTF exigent un timeframe signal de 5 minutes.")
    if economic_validation_sessions is not None:
        if economic_validation_sessions < 2:
            raise ValueError("economic_validation_sessions doit être supérieur ou égal à 2.")
        if dl_objective != "economic":
            raise ValueError("--economic-validation-sessions nécessite --dl-objective economic.")
    out = make_output(out_dir)
    started = time.time()
    reporter = TerminalProgress(enabled=progress)
    _banner(f"PIPELINE {snapshot.symbol} · {timeframe} min")

    snapshot = _closed_session_snapshot(snapshot)
    snapshot = _slice_snapshot_date_range(snapshot, start_date=start_date, end_date=end_date)
    bars = completed_bars(snapshot, timeframe, asof=snapshot.schedule.MarketClose.max(), require_complete=True)
    # Kept as a defensive invariant check; the expensive/session validation above
    # has already been restricted to the requested dates.
    bars = _filter_date_range(bars, start_date=start_date, end_date=end_date)
    bars = _subset_sessions(bars, max_sessions)
    fset = FeatureEngine().build(bars, mtf_variant=feature_variant, snapshot=snapshot)
    dl_feature_columns = GROUPS["D"] + mtf_columns
    dl_ready = fset.values[dl_feature_columns].notna().all(axis=1)
    labels = build_labels(bars, cfg.research, cfg.risk)
    utility_labels = build_utility_labels(bars, fset.values, cfg.research, cfg.risk) if dl_objective == "utility" else None
    candidate_mask = _candidate_setup_mask(bars, fset.values) if dl_objective in {"utility", "economic"} else pd.Series(True, index=bars.index)
    if dl_objective == "economic":
        label_callback = reporter.item_callback("Labels économiques") if progress else None
        economic_labels = build_economic_labels(
            bars, fset.values, cfg.research, cfg.risk, candidate_mask, progress_callback=label_callback
        )
    else:
        economic_labels = None
    sessions = list(pd.unique(bars.Session))
    period_meta = {
        "requested_start_date": start_date,
        "requested_end_date": end_date,
        "actual_first_session": str(sessions[0]) if sessions else None,
        "actual_last_session": str(sessions[-1]) if sessions else None,
        "sessions": int(len(sessions)),
        "bars": int(len(bars)),
        "train_sessions": int(cfg.research.train_sessions),
        "fold_sessions": int(cfg.research.fold_sessions),
    }
    (out / "period_used.json").write_text(json.dumps(period_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    if len(sessions) <= cfg.research.train_sessions:
        raise ValueError(
            f"Historique insuffisant: {len(sessions)} séances, il en faut plus de {cfg.research.train_sessions}."
        )
    print(f"Données: {len(bars)} bougies · {len(sessions)} séances", flush=True)
    print(f"DL: {', '.join(dl_models) if dl_models else 'désactivé'} · seeds={list(seeds)}", flush=True)
    if progress or feature_variant != "baseline":
        print(f"Features: {feature_variant} · DL columns={len(dl_feature_columns)}", flush=True)

    fold_frames = []
    audit_rows = []
    train_rows = []
    utility_prediction_frames = []
    economic_prediction_frames = []
    dl_selection_rows = []
    dl_threshold_grid_rows = []
    checkpoint_root = out / "checkpoints"
    fold_no = 0

    fold_starts = list(_walkforward_fold_starts(
        len(sessions), cfg.research.train_sessions, cfg.research.fold_sessions, max_folds
    ))
    fold_total = len(fold_starts)
    fold_progress = reporter.item_callback("Walk-forward folds") if progress else None
    for fold_position, first in enumerate(fold_starts, start=1):
        days = sessions[first:first + cfg.research.fold_sessions]
        if not len(days):
            continue
        boundary = bars.loc[bars.Session == days[0], "SessionOpen"].iloc[0]
        try:
            train_mask, audit = eligible_train(bars, fset.values, labels, boundary, timeframe, cfg.research)
            classic_train_mask = np.asarray(train_mask, dtype=bool).copy()
            if dl_objective == "economic":
                first_pos = bars.index.searchsorted(boundary, side="left")
                gap = cfg.research.horizon_minutes // timeframe + 1 + cfg.research.extra_embargo_bars
                econ_cols = ["ProfitableAfterCosts","StopHit","TargetHit","TimeExit","RealizedPnL","RealizedReturn","MFE","MAE"]
                ready = dl_ready & economic_labels[econ_cols].notna().all(axis=1) & candidate_mask
                positions = np.arange(len(bars))
                train_mask = (positions < first_pos-gap) & ready.to_numpy() & (economic_labels["LabelEnd"] < boundary).to_numpy()
                if int(train_mask.sum()) < max(60, cfg.research.min_train_rows//4):
                    raise ValueError(f"Entraînement économique insuffisant : {int(train_mask.sum())} lignes.")
                if economic_labels.loc[train_mask, "ProfitableAfterCosts"].nunique() < 2:
                    raise ValueError("Entraînement économique: ProfitableAfterCosts n'a pas deux classes.")
                last = int(np.flatnonzero(train_mask)[-1])
                audit.update({
                    "train_rows": int(train_mask.sum()), "train_last": bars.index[last].isoformat(),
                    "train_label_end_max": economic_labels.loc[train_mask, "LabelEnd"].max().isoformat(),
                    "gap_bars": first_pos-last-1, "required_gap_bars": gap,
                    "leakage_check": bool(economic_labels.loc[train_mask, "LabelEnd"].max() < boundary),
                    "objective": "direct_economic_v11_4",
                })
            elif dl_objective == "utility":
                first_pos = bars.index.searchsorted(boundary, side="left")
                gap = 60 // timeframe + 1 + cfg.research.extra_embargo_bars
                complete_targets = utility_labels[[
                    "TargetBeforeStop_15", "TargetBeforeStop_30", "TargetBeforeStop_60",
                    "MFE_30", "MAE_30", "NetReturn_15", "NetReturn_30", "NetReturn_60"
                ]].notna().all(axis=1)
                ready = dl_ready & complete_targets
                positions = np.arange(len(bars))
                train_mask = (positions < first_pos - gap) & ready & (utility_labels["LabelEnd_60"] < boundary)
                if int(train_mask.sum()) < cfg.research.min_train_rows:
                    raise ValueError(f"Entraînement utility insuffisant : {int(train_mask.sum())} lignes.")
                if utility_labels.loc[train_mask, "TargetBeforeStop_30"].nunique() < 2:
                    raise ValueError("Entraînement utility: target-before-stop 30 min n'a pas deux classes.")
                last = int(np.flatnonzero(train_mask)[-1])
                audit.update({
                    "train_rows": int(train_mask.sum()),
                    "train_last": bars.index[last].isoformat(),
                    "train_label_end_max": utility_labels.loc[train_mask, "LabelEnd_60"].max().isoformat(),
                    "gap_bars": first_pos - last - 1, "required_gap_bars": gap,
                    "leakage_check": bool(utility_labels.loc[train_mask, "LabelEnd_60"].max() < boundary),
                    "objective": "candidate_ranker_v11_3_2",
                })
        except ValueError as exc:
            print(f"Fold ignoré {days[0]}: {exc}", flush=True)
            reporter.update(f"fold {fold_position}/{fold_total} ignoré · {days[0]}", force=True)
            if fold_progress:
                fold_progress({"completed": fold_position, "total": fold_total,
                               "detail": f"fold ignoré {days[0]}"})
            continue
        if dl_objective == "economic":
            econ_cols = ["ProfitableAfterCosts","StopHit","TargetHit","TimeExit","RealizedPnL","RealizedReturn","MFE","MAE"]
            valid_index = bars.index[
                bars.Session.isin(days) & fset.values[GROUPS["D"]].notna().all(axis=1)
                & economic_labels[econ_cols].notna().all(axis=1) & candidate_mask
            ]
        elif dl_objective == "utility":
            valid_index = bars.index[
                bars.Session.isin(days) & fset.values[GROUPS["D"]].notna().all(axis=1)
                & utility_labels[["TargetBeforeStop_15", "TargetBeforeStop_30", "TargetBeforeStop_60",
                                  "MFE_30", "MAE_30", "NetReturn_15", "NetReturn_30", "NetReturn_60"]].notna().all(axis=1)
            ]
        else:
            valid_index = bars.index[
                bars.Session.isin(days) & fset.values[GROUPS["D"]].notna().all(axis=1) & labels.Target.notna()
            ]
        if not len(valid_index):
            if fold_progress:
                fold_progress({"completed": fold_position, "total": fold_total,
                               "detail": f"fold sans validation {days[0]}"})
            continue
        valid_mask = bars.index.isin(valid_index)
        dl_train_mask = (np.asarray(train_mask, dtype=bool) & dl_ready.to_numpy()
                 if dl_objective == "direction" else train_mask)
        dl_valid_mask = valid_mask & dl_ready.to_numpy()
        fold_no += 1
        reporter.update(f"fold {fold_position}/{fold_total} · préparation prédictions", force=True)
        print(
            f"\n[{fold_no}] Walk-forward {days[0]} → {days[-1]} · "
            f"{int(train_mask.sum())} lignes train · {len(valid_index)} lignes test",
            flush=True,
        )
        parts = []
        if include_classic:
            _banner(f"Fold {fold_no}: modèles classiques")
            classic_predictions = []
            classic_progress = reporter.item_callback(f"fold {fold_position}/{fold_total} modèles classiques") if progress else None
            for model_position, model_name in enumerate(MODEL_IDS, start=1):
                reporter.update(
                    f"fold {fold_position}/{fold_total} · modèle classique "
                    f"{model_position}/{len(MODEL_IDS)} {model_name}", force=bool(progress)
                )
                classic_predictions.append(fit_predict(model_name, fset.values, labels, classic_train_mask,
                                                      valid_index, cfg.research))
                if classic_progress:
                    classic_progress({"completed": model_position, "total": len(MODEL_IDS), "detail": model_name})
            parts.append(pd.concat(classic_predictions, axis=1))
        if include_rules:
            parts.append(rule_predictions(bars, fset.values, valid_index))
        for model_position, model_name in enumerate(dl_models, start=1):
            progress_context = f"fold {fold_position}/{fold_total} · modèle {model_position}/{len(dl_models)} · "
            reporter.update(f"{progress_context}{model_name} · seeds={len(seeds)} epochs≤{epochs}", force=True)
            _banner(f"Fold {fold_no}: {model_name} · objectif {dl_objective}")
            if dl_objective == "economic":
                pred, diag, training, selection, threshold_grid = _dl_fold_predict_economic(
                    bars, fset.values, economic_labels, dl_train_mask, dl_valid_mask, model_name,
                    seeds, epochs, patience, checkpoint_root, fold_no, timeframe, cfg.research, cfg.risk,
                    candidate_mask, economic_validation_sessions=economic_validation_sessions,
                    feature_columns=dl_feature_columns, progress=reporter if progress else None,
                    progress_context=progress_context,
                )
                economic_prediction_frames.append(diag)
                dl_selection_rows.append(selection)
                dl_threshold_grid_rows.extend(threshold_grid)
            elif dl_objective == "utility":
                pred, diag, training, selection, threshold_grid = _dl_fold_predict_utility(
                    bars, fset.values, utility_labels, dl_train_mask, dl_valid_mask, model_name,
                    seeds, epochs, patience, checkpoint_root, fold_no, timeframe, cfg.research, cfg.risk, candidate_mask,
                    feature_columns=dl_feature_columns, progress=reporter if progress else None,
                    progress_context=progress_context,
                )
                utility_prediction_frames.append(diag)
                dl_selection_rows.append(selection)
                dl_threshold_grid_rows.extend(threshold_grid)
            else:
                pred, training = _dl_fold_predict(
                    fset.values, labels, dl_train_mask, dl_valid_mask, model_name,
                    seeds, epochs, patience, checkpoint_root, fold_no, feature_columns=dl_feature_columns,
                    progress=reporter if progress else None, progress_context=progress_context,
                )
            parts.append(pred.to_frame())
            train_rows.extend(training)

        fold_pred = pd.concat(parts, axis=1).sort_index()
        fold_frames.append(fold_pred)
        audit.update({
            "fold": fold_no,
            "prediction_first": str(fold_pred.index.min()),
            "prediction_last": str(fold_pred.index.max()),
            "sessions": ",".join(map(str, days)),
        })
        audit_rows.append(audit)
        if fold_progress:
            fold_progress({"completed": fold_position, "total": fold_total,
                           "detail": f"{days[0]} → {days[-1]}"})

    if not fold_frames:
        raise ValueError("Aucun fold walk-forward n'a pu être produit.")

    predictions = pd.concat(fold_frames).sort_index()
    predictions = predictions.loc[~predictions.index.duplicated(keep="first")]
    reporter.update("Exports OOS · predictions et labels", force=True)
    predictions.to_csv(out / "predictions_oos.csv")
    labels.reindex(predictions.index).to_csv(out / "labels_oos.csv")
    if utility_labels is not None:
        utility_labels.reindex(predictions.index).to_csv(out / "utility_labels_oos.csv")
        utility_diag = pd.DataFrame()
        if utility_prediction_frames:
            utility_diag = pd.concat(utility_prediction_frames).sort_index().groupby(level=0).first()
            utility_diag.to_csv(out / "utility_predictions_oos.csv")
            raw_rows = []
            for model_name in dl_models:
                col = model_name + "__SetupScoreRaw"
                if col in utility_diag.columns:
                    raw_rows.append(_raw_rank_metrics(utility_labels["TargetBeforeStop_60"], utility_diag[col], model_name))
            pd.DataFrame(raw_rows).to_csv(out / "dl_raw_oos_metrics.csv", index=False)
        stats = []
        uoos = utility_labels.reindex(predictions.index)
        for h in (15, 30, 60):
            col = f"TargetBeforeStop_{h}"
            stats.append({"target": col, "rows": int(uoos[col].notna().sum()), "positive_rate": float(uoos[col].mean())})
        for col in ("MFE_30", "MAE_30", "NetReturn_15", "NetReturn_30", "NetReturn_60"):
            stats.append({"target": col, "rows": int(uoos[col].notna().sum()), "mean": float(uoos[col].mean()),
                          "median": float(uoos[col].median())})
        pd.DataFrame(stats).to_csv(out / "utility_target_stats.csv", index=False)
        pd.DataFrame(dl_selection_rows).to_csv(out / "dl_threshold_selection.csv", index=False)
        pd.DataFrame(dl_threshold_grid_rows).to_csv(out / "dl_threshold_grid.csv", index=False)
    if economic_labels is not None:
        economic_labels.reindex(predictions.index).to_csv(out / "economic_labels_oos.csv")
        economic_diag = pd.DataFrame()
        if economic_prediction_frames:
            economic_diag = pd.concat(economic_prediction_frames).sort_index().groupby(level=0).first()
            economic_diag.to_csv(out / "economic_predictions_oos.csv")
            rows=[]
            for model_name in dl_models:
                if model_name+"__EconomicScoreRaw" in economic_diag.columns:
                    rows.append(_economic_oos_metrics(economic_labels, economic_diag, model_name))
            pd.DataFrame(rows).to_csv(out / "dl_economic_oos_metrics.csv", index=False)
            # Alias conservé pour les scripts de diagnostic précédents.
            pd.DataFrame(rows).to_csv(out / "dl_raw_oos_metrics.csv", index=False)
        eoos=economic_labels.reindex(predictions.index)
        stats=[]
        for col in ["ProfitableAfterCosts","StopHit","TargetHit","TimeExit"]:
            stats.append({"target":col,"rows":int(eoos[col].notna().sum()),"positive_rate":float(eoos[col].mean())})
        for col in ["RealizedPnL","RealizedReturn","MFE","MAE"]:
            stats.append({"target":col,"rows":int(eoos[col].notna().sum()),"mean":float(eoos[col].mean()),"median":float(eoos[col].median())})
        pd.DataFrame(stats).to_csv(out / "economic_target_stats.csv", index=False)
        pd.DataFrame(dl_selection_rows).to_csv(out / "dl_threshold_selection.csv", index=False)
        pd.DataFrame(dl_threshold_grid_rows).to_csv(out / "dl_threshold_grid.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(out / "folds.csv", index=False)
    pd.DataFrame(train_rows).to_csv(out / "dl_training.csv", index=False)
    reporter.update("Exports OOS · features.csv", force=True)
    fset.values.to_csv(out / "features.csv")
    reporter.update("Exports OOS · bars.csv", force=True)
    bars.to_csv(out / "bars.csv")
    if dl_objective in {"utility", "economic"}:
        candidate_frame = pd.DataFrame({"CandidateSetup": candidate_mask.astype(int), "Session": bars.Session}, index=bars.index)
        candidate_frame.to_csv(out / "candidate_setups.csv")
        candidate_frame.groupby("Session").CandidateSetup.agg(["count","sum","mean"]).rename(columns={"count":"bars","sum":"candidates","mean":"candidate_rate"}).to_csv(out / "candidate_setup_stats.csv")
    reporter.update("Exports OOS · CSV prêts", force=True)

    evaluated_sessions = bars.loc[bars.index.isin(predictions.index), "Session"].unique()
    evaluated = bars.loc[bars.Session.isin(evaluated_sessions)].copy()
    metrics = []
    curves = {}
    stress_rows = []

    # Baseline diagnostique: tous les setups candidats, sans ranking DL.
    # Elle est exclue de toute promotion et sert à tester la qualité de l'univers candidat.
    if dl_objective in {"utility", "economic"}:
        reporter.update("Diagnostic candidat · backtest et stress", force=True)
        candidate_oos = candidate_mask.reindex(predictions.index).fillna(False).astype(float).rename("CANDIDATE_ALL")
        candidate_sim_progress = reporter.bar_callback("CANDIDATE_ALL · backtest") if progress else None
        candidate_all_sim = simulate(evaluated, candidate_oos, fset.values, cfg.risk, cfg.research,
                                     progress_callback=candidate_sim_progress)
        metrics.append({"model": "CANDIDATE_ALL", "kind": "diagnostic", **performance(candidate_all_sim, cfg.risk.initial_capital)})
        curves["CANDIDATE_ALL"] = candidate_all_sim.ledger.Capital
        candidate_all_sim.ledger.to_csv(out / "ledger_CANDIDATE_ALL.csv")
        candidate_all_sim.trades.to_csv(out / "trades_CANDIDATE_ALL.csv", index=False)
        for extra in (2.0, 5.0):
            reporter.update(f"CANDIDATE_ALL · stress +{extra:g} bps", force=True)
            stressed = replace(cfg.risk, slippage_bps=cfg.risk.slippage_bps + extra)
            stress_progress = reporter.bar_callback(f"CANDIDATE_ALL · stress +{extra:g} bps") if progress else None
            stress_sim = simulate(evaluated, candidate_oos, fset.values, stressed, cfg.research,
                                  progress_callback=stress_progress)
            perf = performance(stress_sim, stressed.initial_capital)
            stress_rows.append({
                "model": "CANDIDATE_ALL", "extra_slippage_bps": extra,
                "total_return": perf["total_return"], "final_capital": perf["final_capital"],
                "max_drawdown": perf["max_drawdown"], "trades": perf["trades"],
            })
        oracle_progress = reporter.item_callback("CANDIDATE_ALL · oracle diagnostique") if progress else None
        oracle = _candidate_oracle_diagnostic(evaluated, candidate_oos, fset.values, cfg.risk, cfg.research,
                              progress_callback=oracle_progress)
        oracle.to_csv(out / "candidate_oracle_diagnostic.csv", index=False)
        eligible_oracle = oracle.loc[oracle.get("eligible_trade", pd.Series(dtype=bool)).fillna(False)] if len(oracle) else oracle
        oracle_summary = pd.DataFrame([{
            "candidate_signals_oos": int((candidate_oos >= cfg.research.threshold).sum()),
            "isolated_eligible_trades": int(len(eligible_oracle)),
            "isolated_profitable_trades": int(eligible_oracle.get("profitable_after_costs", pd.Series(dtype=bool)).fillna(False).sum()) if len(eligible_oracle) else 0,
            "isolated_profitable_rate": float(eligible_oracle["profitable_after_costs"].mean()) if len(eligible_oracle) else np.nan,
            "isolated_mean_pnl": float(eligible_oracle["pnl"].mean()) if len(eligible_oracle) else np.nan,
            "isolated_median_pnl": float(eligible_oracle["pnl"].median()) if len(eligible_oracle) else np.nan,
            "warning": "ORACLE DIAGNOSTIC ONLY - future leakage - never use as a trading signal",
        }])
        oracle_summary.to_csv(out / "candidate_oracle_summary.csv", index=False)

    _banner("BACKTEST ÉCONOMIQUE")
    economic_models = list(predictions.columns)
    export_progress = reporter.item_callback("Exports ledger/trades") if progress else None
    for model_position, model in enumerate(economic_models, start=1):
        reporter.update(f"BACKTEST ÉCONOMIQUE · modèle {model_position}/{len(economic_models)} {model}", force=True)
        score = predictions[model].dropna()
        main_callback = reporter.bar_callback(
            f"{model_position}/{len(economic_models)} {model} · backtest principal"
        ) if progress else None
        sim = simulate(evaluated, score, fset.values, cfg.risk, cfg.research,
                       progress_callback=main_callback)
        reporter.update(f"{model} · métriques/classification · {len(sim.trades)} trades", force=True)
        row = {
            "model": model,
            "kind": "dl" if model.startswith(("MLP_", "TCN_")) else ("rule" if model.startswith("RULE_") else "ml"),
            **performance(sim, cfg.risk.initial_capital),
        }
        if not model.startswith("RULE_"):
            if dl_objective == "economic" and model.startswith(("MLP_", "TCN_")):
                cls_labels = pd.DataFrame({"Target": economic_labels["ProfitableAfterCosts"]}, index=economic_labels.index)
                row.update(classification(cls_labels, score, cfg.research.threshold))
            elif dl_objective == "utility" and model.startswith(("MLP_", "TCN_")):
                cls_labels = pd.DataFrame({"Target": utility_labels["TargetBeforeStop_60"]}, index=utility_labels.index)
                row.update(classification(cls_labels, score, cfg.research.threshold))
            else:
                row.update(classification(labels, score, cfg.research.threshold))
        metrics.append(row)
        curves[model] = sim.ledger.Capital
        sim.ledger.to_csv(out / f"ledger_{model}.csv")
        if export_progress:
            export_progress({"completed": (model_position - 1) * 2 + 1,
                             "total": len(economic_models) * 2, "detail": f"ledger_{model}.csv"})
        sim.trades.to_csv(out / f"trades_{model}.csv", index=False)
        if export_progress:
            export_progress({"completed": model_position * 2, "total": len(economic_models) * 2,
                             "detail": f"trades_{model}.csv ({len(sim.trades)} trades)"})
        for extra in (2.0, 5.0):
            reporter.update(f"{model} · stress coûts +{extra:g} bps", force=True)
            stressed = replace(cfg.risk, slippage_bps=cfg.risk.slippage_bps + extra)
            stress_callback = reporter.bar_callback(f"{model} · stress +{extra:g} bps") if progress else None
            stress_sim = simulate(evaluated, score, fset.values, stressed, cfg.research,
                                  progress_callback=stress_callback)
            perf = performance(stress_sim, stressed.initial_capital)
            stress_rows.append({
                "model": model,
                "extra_slippage_bps": extra,
                "total_return": perf["total_return"],
                "final_capital": perf["final_capital"],
                "max_drawdown": perf["max_drawdown"],
                "trades": perf["trades"],
            })

    for name, intraday in (("BUY_HOLD_X1", False), ("INTRADAY_HOLD_X1", True)):
        sim = reference_curve(evaluated, cfg.risk, intraday)
        metrics.append({"model": name, "kind": "benchmark", **performance(sim, cfg.risk.initial_capital)})
        curves[name] = sim.ledger.Capital
        sim.ledger.to_csv(out / f"ledger_{name}.csv")
    cash = cash_curve(evaluated, cfg.risk)
    metrics.append({"model": "CASH", "kind": "benchmark", **performance(cash, cfg.risk.initial_capital)})
    curves["CASH"] = cash.ledger.Capital

    reporter.update("BACKTEST ÉCONOMIQUE · benchmarks et métriques finales", force=True)
    metrics_df = pd.DataFrame(metrics)
    reporter.update("BACKTEST ÉCONOMIQUE · écriture metrics/stress/bootstrap/daily", force=True)
    metrics_df.to_csv(out / "metrics.csv", index=False)
    stress_df = pd.DataFrame(stress_rows)
    stress_df.to_csv(out / "stress_costs.csv", index=False)
    bootstrap_daily(curves, evaluated, cfg.risk.initial_capital, cfg.research.seed).to_csv(
        out / "bootstrap_daily.csv", index=False
    )
    daily = pd.concat(curves, axis=1).groupby(evaluated.Session).last()
    daily.to_csv(out / "capital_daily.csv")

    # Conservative review status, deliberately not an automatic promotion.
    review_rows = []
    stress_wide = stress_df.pivot(index="model", columns="extra_slippage_bps", values="total_return") if len(stress_df) else pd.DataFrame()
    min_trades = max(5, int(np.ceil(len(evaluated_sessions) / 3)))
    for _, row in metrics_df.iterrows():
        model = row["model"]
        if row["kind"] in {"benchmark", "diagnostic"}:
            continue
        s2 = float(stress_wide.loc[model, 2.0]) if model in stress_wide.index else np.nan
        s5 = float(stress_wide.loc[model, 5.0]) if model in stress_wide.index else np.nan
        candidate = (
            int(row.get("trades", 0)) >= min_trades
            and float(row.get("total_return", -1)) > 0
            and float(row.get("max_drawdown", -1)) > -0.05
            and np.isfinite(s2) and s2 > 0
            and np.isfinite(s5) and s5 > 0
        )
        review_rows.append({
            "model": model,
            "trades": int(row.get("trades", 0)),
            "total_return": float(row.get("total_return", np.nan)),
            "max_drawdown": float(row.get("max_drawdown", np.nan)),
            "stress_2bps_return": s2,
            "stress_5bps_return": s5,
            "review_status": "CANDIDATE_FOR_REVIEW" if candidate else "REJECTED",
        })
    review = pd.DataFrame(review_rows)
    review.to_csv(out / "review_gate.csv", index=False)

    # Concentration du PnL : un petit nombre de gros trades ne doit pas être confondu avec un edge robuste.
    concentration_rows = []
    for model in predictions.columns:
        path = out / f"trades_{model}.csv"
        if not path.exists():
            continue
        trades = pd.read_csv(path)
        pnl = pd.to_numeric(trades.get("pnl", pd.Series(dtype=float)), errors="coerce").dropna().sort_values(ascending=False)
        total = float(pnl.sum()) if len(pnl) else 0.0
        top1 = float(pnl.head(1).sum()) if len(pnl) else 0.0
        top3 = float(pnl.head(3).sum()) if len(pnl) else 0.0
        concentration_rows.append({
            "model": model, "trades": int(len(pnl)), "total_pnl": total, "top1_pnl": top1, "top3_pnl": top3,
            "pnl_without_best_1": total - top1, "pnl_without_best_3": total - top3,
            "top1_share_abs": abs(top1) / max(abs(total), 1e-12) if total else np.nan,
            "top3_share_abs": abs(top3) / max(abs(total), 1e-12) if total else np.nan,
        })
    pd.DataFrame(concentration_rows).to_csv(out / "trade_concentration.csv", index=False)
    reporter.update("BACKTEST ÉCONOMIQUE · exports terminés", force=True)

    elapsed = time.time() - started
    report = {
        "version": "11.4-terminal",
        "status": "DEVELOPMENT ONLY",
        "symbol": snapshot.symbol,
        "timeframe_minutes": timeframe,
        "sessions_total": len(sessions),
        "sessions_oos": int(len(evaluated_sessions)),
        "bars": int(len(bars)),
        "folds": int(fold_no),
        "dl_models": list(dl_models),
        "seeds": [int(s) for s in seeds],
        "epochs_max": int(epochs),
        "patience": int(patience),
        "threshold": float(cfg.research.threshold),
        "dl_objective": dl_objective,
        "execution": "candidate setup causal -> DL ExpectedTradeValue -> validation backtest threshold -> OOS signal close t -> entry open t+1; no overnight",
        "automatic_promotion": False,
        "elapsed_seconds": round(elapsed, 2),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            ("Le DL apprend target-before-stop 15/30/60 min + MFE/MAE + rendements multi-horizon; seuil choisi uniquement sur validation interne et score setup combiné; "
             if dl_objective == "utility" else
             "Le target DL est le signe du rendement net futur à horizon fixe; ")
            + "les scores DL/ML ne sont pas des probabilités calibrées et ce pipeline ne prouve pas un edge tradable."
        ),
    }
    if feature_variant != "baseline":
        report["feature_variant"] = feature_variant
        report["dl_feature_columns"] = dl_feature_columns
    (out / "run.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    printable = metrics_df[[c for c in [
        "model", "kind", "trades", "win_rate", "total_return", "final_capital",
        "max_drawdown", "sharpe", "roc_auc"
    ] if c in metrics_df.columns]].copy()
    printable = printable.sort_values("final_capital", ascending=False, na_position="last")
    print("\n" + printable.to_string(index=False), flush=True)
    print(f"\nRésultats: {out}\nDurée: {elapsed:.1f}s", flush=True)
    return metrics_df, report


def fetch_yahoo_snapshot(symbol: str, out_dir: Path):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    snapshot = YahooProvider().fetch(symbol.upper())
    write_snapshot(snapshot, out)
    return snapshot


def run_terminal_pipeline(symbol: str, out_dir: Path, data_dir: Path | None = None, source="local",
                          timeframe=5, dl_models=("MLP_12", "MLP_24", "TCN_12", "TCN_24"),
                          seeds=(42,), epochs=20, patience=4, config_path: Path | None = None,
                          max_sessions=None, dl_objective="direction", train_sessions=None, fold_sessions=None,
                          start_date=None, end_date=None, max_folds: int | None = None,
                          economic_validation_sessions: int | None = None,
                          feature_variant="baseline", progress=False):
    cfg = AppConfig.load(config_path if config_path and Path(config_path).exists() else None)
    if train_sessions is not None or fold_sessions is not None:
        rc = replace(cfg.research,
                     train_sessions=int(train_sessions) if train_sessions is not None else cfg.research.train_sessions,
                     fold_sessions=int(fold_sessions) if fold_sessions is not None else cfg.research.fold_sessions)
        cfg = replace(cfg, research=rc)
    symbol = symbol.upper()
    run_root = Path(out_dir)
    if source == "yahoo":
        snapshot_dir = run_root.parent / f"{run_root.name}_snapshot"
        if snapshot_dir.exists():
            raise ValueError(f"{snapshot_dir} existe déjà; choisir un autre --out pour figer un nouveau snapshot.")
        _banner(f"SNAPSHOT YAHOO {symbol}")
        snapshot = fetch_yahoo_snapshot(symbol, snapshot_dir)
        print(f"Snapshot figé: {snapshot_dir}", flush=True)
    else:
        if data_dir is None:
            raise ValueError("--data est requis avec --source local.")
        snapshot = CSVProvider(data_dir).fetch(symbol)
    return unified_walkforward(
        snapshot, cfg, run_root, timeframe=timeframe, dl_models=tuple(dl_models), seeds=tuple(seeds),
        epochs=epochs, patience=patience, include_classic=True, include_rules=True,
        max_sessions=max_sessions, dl_objective=dl_objective, start_date=start_date, end_date=end_date,
        max_folds=max_folds, economic_validation_sessions=economic_validation_sessions,
        feature_variant=feature_variant, progress=progress,
    )


def doctor() -> dict:
    import platform
    import sklearn
    report = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "torch": None,
        "cuda": False,
        "cuda_device": None,
    }
    try:
        import torch
        report["torch"] = torch.__version__
        report["cuda"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            report["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        report["torch"] = f"unavailable: {exc}"
    return report
