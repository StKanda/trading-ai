from __future__ import annotations
import hashlib, json
from pathlib import Path
import pandas as pd
from .config import AppConfig
from .data.providers import CSVProvider
from .data.market import completed_bars
from .data.freshness import evaluate_freshness
from .features.engine import FeatureEngine, build_labels, GROUPS
from .research.models import eligible_train
from .deep_learning.dataset import build_sequence_bundle


def dataset_manifest(data_dir, symbol="SPY"):
    snap=CSVProvider(data_dir).fetch(symbol)
    path=Path(data_dir)/f"{symbol}_5m.csv"
    h=hashlib.sha256(path.read_bytes()).hexdigest()
    local=snap.prices.index.tz_convert("America/New_York")
    return {"symbol":symbol,"source":snap.metadata.get("source"),"timeframe":5,
            "start":local[0].isoformat(),"end":local[-1].isoformat(),"bars":len(snap.prices),
            "sessions_with_prices":int(len(set(local.strftime('%Y-%m-%d')))),
            "fetched_at":snap.metadata.get("fetched_at"),"synthetic":bool(snap.metadata.get("synthetic")),
            "sha256":h}


def audit_v10(data_dir, symbol, out):
    snap=CSVProvider(data_dir).fetch(symbol)
    bars=completed_bars(snap,5,asof=snap.schedule.MarketClose.max(),require_complete=True)
    manifest=dataset_manifest(data_dir,symbol)
    priced=set(snap.prices.index.tz_convert("America/New_York").strftime("%Y-%m-%d"))
    calendar_without=sorted(set(snap.schedule.index)-priced)
    audit={"status":"OK","manifest":manifest,"completed_bars":len(bars),
           "missing_bars":int(bars.attrs.get("missing_bars",0)),"outside_session_rows":int(bars.attrs.get("outside_session_rows",0)),
           "calendar_sessions_without_prices":calendar_without,
           "statistical_status":"DEVELOPMENT ONLY" if manifest["sessions_with_prices"]<100 else "RESEARCH DATASET"}
    Path(out).write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding="utf-8")
    return audit


def dl_smoke_from_data(data_dir, symbol, out_dir, context=24, architecture="mlp", max_epochs=3, patience=2, seed=42):
    try:
        from .deep_learning.train import smoke_train
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise ImportError(
                "PyTorch n'est pas installe. Pour activer le Deep Learning: "
                "python -m pip install -r requirements_dl.txt"
            ) from exc
        raise
    cfg=AppConfig.load(Path("config.json"))
    snap=CSVProvider(data_dir).fetch(symbol)
    bars=completed_bars(snap,5,asof=snap.schedule.MarketClose.max(),require_complete=True)
    features=FeatureEngine().build(bars)
    labels=build_labels(bars,cfg.research,cfg.risk)
    sessions=list(pd.unique(bars.Session))
    if len(sessions)<15: raise ValueError("Too few sessions even for DL smoke test")
    valid_start=sessions[max(10,int(len(sessions)*0.75))]
    boundary=bars.loc[bars.Session>=valid_start,"SessionOpen"].iloc[0]
    train_mask,_=eligible_train(bars,features.values,labels,boundary,5,cfg.research)
    valid_mask=(bars.Session>=valid_start).to_numpy() & labels.Target.notna().to_numpy()
    bundle=build_sequence_bundle(features.values,labels,train_mask,valid_mask,context_bars=context,
                                 columns=GROUPS["D"])
    if architecture == "mlp" and int(max_epochs) == 3 and int(patience) == 2 and int(seed) == 42:
        result=smoke_train(bundle,out_dir)
    else:
        from .deep_learning.train import train_classifier
        out_path=Path(out_dir); out_path.mkdir(parents=True, exist_ok=True)
        tr=train_classifier(bundle, architecture=architecture, hidden_size=24, max_epochs=int(max_epochs),
                            patience=int(patience), seed=int(seed), checkpoint=out_path / "dl_model.pt")
        result={"status":"DEVELOPMENT ONLY", "best_epoch":tr.best_epoch,
                "train_loss":tr.train_loss, "valid_loss":tr.valid_loss, "device":tr.device,
                "n_parameters":tr.n_parameters, "checkpoint":tr.checkpoint,
                "architecture":architecture, "max_epochs":int(max_epochs), "patience":int(patience),
                "seed":int(seed), "note":"Experimental training; not evidence of a tradable edge."}
    result.update(train_sequences=len(bundle.x_train),valid_sequences=len(bundle.x_valid),
                  validation_start=valid_start,dataset_sessions=len(sessions))
    Path(out_dir,"dl_smoke.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result
