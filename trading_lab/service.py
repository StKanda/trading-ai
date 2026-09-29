"""Cas d'usage partagés par le serveur : scan, graphique, scénario, explication."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import RLock
import time
import numpy as np
import pandas as pd
from .data.market import completed_bars, utc
from .features.engine import FeatureEngine, build_labels, GROUPS
from .research.models import eligible_train, fit_predict
from .scanner import StockScanner
from .assistant import AnalysisAssistant
from .risk import trade_plan


class LabService:
    def __init__(self, provider, config, mode):
        self.provider, self.config, self.mode = provider, config, mode
        self.snapshots, self.analyses = {}, OrderedDict()
        self.lock = RLock()
        self.scanner, self.assistant = StockScanner(), AnalysisAssistant()

    def snapshot(self, symbol):
        if symbol not in self.config.symbols:
            raise ValueError("Symbole hors de l'univers configuré.")
        with self.lock:
            cached = self.snapshots.get(symbol)
            if cached and (self.mode != "yahoo" or time.monotonic() - cached[0] < 60):
                return cached[1]
        fresh = self.provider.fetch(symbol)
        with self.lock:
            self.snapshots[symbol] = (time.monotonic(), fresh)
        return fresh

    def cutoff(self, snapshot, replay_minute=None):
        if self.mode == "demo":
            minute = 270 if replay_minute is None else int(replay_minute)
            if not 30 <= minute <= 390:
                raise ValueError("Rejeu : de 30 à 390 minutes après ouverture.")
            return snapshot.schedule.MarketOpen.iloc[-1] + pd.Timedelta(minutes=minute)
        if self.mode == "csv":
            last_end = snapshot.prices.index[-1] + pd.Timedelta(minutes=5)
            return min(last_end, utc(snapshot.metadata["fetched_at"]))
        return pd.Timestamp.now(tz="UTC")

    def analyze(self, symbol, timeframe=5, replay_minute=None, leverage=None):
        snap = self.snapshot(symbol)
        asof = self.cutoff(snap, replay_minute)
        risk = self.config.risk if leverage is None else replace(self.config.risk, max_leverage=float(leverage))
        key = (symbol, timeframe, asof.floor("5min").isoformat(), risk.max_leverage, snap.metadata.get("fetched_at"))
        with self.lock:
            if key in self.analyses:
                return self.analyses[key]
        bars = completed_bars(snap, timeframe, asof=asof)
        features = FeatureEngine().build(bars, latest_only=True)
        scan = self.scanner.evaluate(snap, bars, features, asof, timeframe, risk)
        current = bars.loc[bars.Session == bars.Session.iloc[-1]]
        f = features.values.loc[current.index]
        plan = trade_plan(current.Close.iloc[-1], f.ATR14Pct.iloc[-1], risk.initial_capital,
                          current.Volume.iloc[-1], risk)
        if plan:
            plan["entry_window_open"] = bool(asof + pd.Timedelta(minutes=self.config.research.horizon_minutes)
                                              <= current.SessionClose.iloc[-1] - pd.Timedelta(minutes=timeframe))
            plan["data_usable"] = scan["selection_score"] is not None
        patterns = []
        pmap = {int(p.pivot_pos): p for p in features.pivots.itertuples()}
        for event in features.events.loc[features.events.accepted == True].tail(8).itertuples():
            patterns.append(dict(pattern=event.pattern, status="cassure confirmée", neckline=event.neckline,
                                 quality_score=event.quality_score, confirmed_at=event.breakout_date,
                                 points=[(pmap[pos].pivot_date, pmap[pos].price)
                                         for pos in (event.start_pos, event.middle_pos, event.end_pos)]))
        patterns += features.pending
        levels = []
        for name, column, color in (("Support confirmé", "NearestSupport", "teal"),
                                     ("Résistance confirmée", "NearestResistance", "coral"),
                                     ("Haut séance précédente", "PreviousHigh", "muted"),
                                     ("Bas séance précédente", "PreviousLow", "muted")):
            value = f[column].iloc[-1]
            if pd.notna(value):
                levels.append(dict(name=name, price=float(value), color=color))
        context = dict(symbol=symbol, timeframe=timeframe, mode=self.mode, scanner=scan,
                       asof=asof.isoformat(), session=str(current.Session.iloc[-1]),
                       currency="USD", plan=plan, patterns=patterns, levels=levels,
                       diagnostics=dict(bars.attrs),
                       bars=[dict(time=t.isoformat(), open=r.Open, high=r.High, low=r.Low,
                                  close=r.Close, volume=r.Volume,
                                  vwap=f.loc[t, "VWAP"], sma20=r.Close / (1 + f.loc[t, "DistSMA20"]))
                             for t, r in current.iterrows()],
                       model=None)
        context["commentary"] = self.assistant.answer("synthèse", context)
        with self.lock:
            self.analyses[key] = context
            while len(self.analyses) > 64:
                self.analyses.popitem(last=False)
        return context

    def scan(self, timeframe=5, replay_minute=None):
        def one(symbol):
            try:
                return self.analyze(symbol, timeframe, replay_minute)["scanner"]
            except Exception as exc:
                return dict(symbol=symbol, selection_score=None, error=str(exc))
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(one, self.config.symbols))
        return sorted(results, key=lambda x: x.get("selection_score") if x.get("selection_score") is not None else -1, reverse=True)

    def model_estimate(self, symbol, timeframe=5, replay_minute=None):
        snap = self.snapshot(symbol)
        bars = completed_bars(snap, timeframe, asof=self.cutoff(snap, replay_minute))
        if bars.index[-1] + pd.Timedelta(minutes=self.config.research.horizon_minutes) > (
                bars.SessionClose.iloc[-1] - pd.Timedelta(minutes=timeframe)):
            raise ValueError("La séance ne permet plus une entrée et un horizon de 30 minutes avant fermeture.")
        features = FeatureEngine().build(bars)
        labels = build_labels(bars, self.config.research, self.config.risk)
        boundary = bars.SessionOpen.iloc[-1]
        mask, audit = eligible_train(bars, features.values, labels, boundary, timeframe, self.config.research)
        last = bars.index[-1:]
        if features.values.loc[last, GROUPS["D"]].isna().any().any():
            raise ValueError("Historique insuffisant pour les indicateurs de la dernière bougie.")
        prediction = fit_predict("RF_A", features.values, labels, mask, last, self.config.research)
        return dict(model="RF_A", score=float(prediction.iloc[0]), calibrated=False,
                    target="Rendement net théorique à 30 min > 0, entrée à l'ouverture suivante",
                    note="Score du modèle non calibré ; aucune probabilité de trade gagnant validée.",
                    audit=audit, synthetic=bool(snap.metadata.get("synthetic")))
