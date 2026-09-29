#!/usr/bin/env python3
"""Local paper-only inference worker. Never sends orders."""
import argparse, json, time
from pathlib import Path
import pandas as pd
from trading_lab.config import AppConfig
from trading_lab.data.providers import CSVProvider, YahooProvider
from trading_lab.live.decision import decision_from_latest


def write_state(path, state):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description="V10.1 paper-only live worker")
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument("--data", type=Path)
    g.add_argument("--yahoo", action="store_true")
    p.add_argument("--symbol", default="SPY")
    p.add_argument("--champion", default="RULE_MOMENTUM")
    p.add_argument("--state", type=Path, default=None,
                   help="Par défaut: runtime/SYMBOL/runtime_state.json")
    p.add_argument("--poll-seconds", type=int, default=30)
    p.add_argument("--once", action="store_true")
    p.add_argument("--asof", help="UTC/offset-aware timestamp for replay/smoke")
    args=p.parse_args(); cfg=AppConfig.load(Path("config.json"))
    args.symbol=args.symbol.upper().strip()
    state_path=args.state or Path("runtime")/args.symbol/"runtime_state.json"
    provider=CSVProvider(args.data) if args.data else YahooProvider()
    while True:
        try:
            snap=provider.fetch(args.symbol)
            state=decision_from_latest(snap,cfg,args.champion,5,args.asof)
            state.update(live_worker_status="ON", research_worker_status="UNKNOWN", symbol=args.symbol)
        except Exception as exc:
            state={"decision":"NO TRADE","reason":str(exc),"live_worker_status":"ERROR",
                   "symbol":args.symbol,"champion":args.champion,
                   "generated_at":pd.Timestamp.now(tz="UTC").isoformat()}
        write_state(state_path,state)
        # Pointeur de compatibilité: le dernier symbole traité reste lisible par les anciennes vues.
        write_state(Path("runtime/runtime_state.json"),state)
        print(json.dumps(state,ensure_ascii=False),flush=True)
        if args.once: break
        time.sleep(max(5,args.poll_seconds))
if __name__=="__main__": main()
