#!/usr/bin/env python3
"""Bounded local research worker. It never promotes a champion automatically."""
import argparse, json
from pathlib import Path
from trading_lab.research_engine import ExperimentRegistry, ResearchBudget, run_bounded_cycle


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--registry",type=Path,default=Path("runtime/experiments.db"))
    p.add_argument("--dataset-hash",default="SMOKE_DATASET")
    p.add_argument("--max-experiments",type=int,default=3)
    p.add_argument("--out",type=Path,default=Path("runtime/research_cycle_report.json"))
    args=p.parse_args(); registry=ExperimentRegistry(args.registry)
    # Smoke hypotheses validate budget/registry mechanics only. Real hypotheses must be
    # generated from research diagnostics and evaluated on research folds.
    hyps=[{"hypothesis_id":f"SMOKE{i+1:02d}","model":"RULE_MOMENTUM","params":{"threshold":v}}
          for i,v in enumerate((0.54,0.55,0.56,0.57,0.58))]
    def evaluator(h):
        # Intentionally not a performance optimizer: mechanics-only smoke evaluation.
        return {"trades":0,"expectancy":0.0},{"passes":False,"reason":"SMOKE TEST ONLY"}
    report=run_bounded_cycle(registry,args.dataset_hash,"10.0.0",hyps,evaluator,
                             ResearchBudget(max_experiments_per_cycle=args.max_experiments,
                                            max_experiments_total=500,max_generations=1,
                                            stop_after_no_improvement=max(1,args.max_experiments),
                                            max_runtime_minutes=5))
    report["status"]="DEVELOPMENT ONLY"; report["manual_promotion"]=True
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
if __name__=="__main__": main()
