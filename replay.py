#!/usr/bin/env python3
"""Fast historical paper replay.

FeatureEngine is evaluated once on the historical prefix ending at the selected
session. V8/V9 prefix-invariance tests guarantee those feature rows do not depend
on future bars. This avoids recomputing years of history for every 5-minute step.
"""
import argparse, json
from dataclasses import asdict
from pathlib import Path
import pandas as pd
from trading_lab.config import AppConfig
from trading_lab.data.providers import CSVProvider
from trading_lab.data.market import completed_bars
from trading_lab.features.engine import FeatureEngine
from trading_lab.research.rules import rule_predictions, RULE_IDS
from trading_lab.risk import detailed_position_size


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,required=True); p.add_argument("--symbol",default="SPY")
    p.add_argument("--session",required=True); p.add_argument("--champion",default="RULE_MOMENTUM")
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args(); cfg=AppConfig.load(Path("config.json")); snap=CSVProvider(args.data).fetch(args.symbol)
    if args.champion not in RULE_IDS: raise SystemExit("Replay smoke supports RULE_* champion")
    if args.session not in snap.schedule.index: raise SystemExit("Session absent from schedule")
    cutoff=snap.schedule.loc[args.session,"MarketClose"]
    bars=completed_bars(snap,5,asof=cutoff)
    session_ix=bars.index[bars.Session==args.session]
    if len(session_ix)==0: raise SystemExit("Session has no complete bars")
    features=FeatureEngine().build(bars)
    pred=rule_predictions(bars,features.values,session_ix)[args.champion]
    rows=[]
    for end in session_ix:
        row=bars.loc[end]; score=float(pred.loc[end]); atr=features.values.loc[end,"ATR14Pct"]
        if pd.isna(atr):
            rows.append({"market_timestamp":end.isoformat(),"decision":"INSUFFICIENT CONFIDENCE","reason":"ATR unavailable"}); continue
        entry=float(row.Close)*(1+cfg.risk.slippage_bps/10000); stop_pct=max(cfg.risk.min_stop_pct,cfg.risk.stop_atr*float(atr)); stop=entry*(1-stop_pct)
        sizing=detailed_position_size(cfg.risk.initial_capital,entry,stop,float(row.Volume),cfg.risk)
        decision="LONG CANDIDATE" if score>=cfg.research.threshold and sizing.quantity>0 else "NO TRADE"
        rows.append({"market_timestamp":end.isoformat(),"decision":decision,"score":score,"entry_estimate":entry,"stop":stop,
                     "target":entry*(1+cfg.risk.reward_risk*stop_pct),"sizing":asdict(sizing),"mode":"IDEAL_DATA"})
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(rows,indent=2),encoding="utf-8")
    print(f"Replay rows: {len(rows)} -> {args.out}")
if __name__=="__main__": main()
