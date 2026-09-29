"""Commandes : interface, capture de données, recherche puis test final."""
import argparse
import json
from pathlib import Path
import sys
from .config import AppConfig
from .data.providers import CSVProvider, YahooProvider, DemoProvider, write_snapshot
from .service import LabService
from .web.server import serve


def _positive_int(value):
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("La valeur doit être un entier positif.") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("La valeur doit être supérieure ou égale à 1.")
    return parsed


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Trading AI V12.1.2 Terminal : historique Alpaca SIP + re-clean local + backtesting/DL, aucune exécution broker.")
    parser.add_argument("--config", type=Path, default=root / "config.json")
    commands = parser.add_subparsers(dest="command", required=True)
    web = commands.add_parser("serve", help="Ouvrir le tableau de bord local")
    source = web.add_mutually_exclusive_group()
    source.add_argument("--demo", action="store_true", help="Données synthétiques, par défaut")
    source.add_argument("--data", type=Path, help="Dossier de snapshots CSV")
    source.add_argument("--yahoo", action="store_true", help="Interroger Yahoo, délai non garanti")
    web.add_argument("--port", type=int, default=8765)
    demo = commands.add_parser("demo-data", help="Écrire des CSV synthétiques pour tester le pipeline")
    demo.add_argument("--out", type=Path, required=True)
    download = commands.add_parser("download", help="Capturer les 59 derniers jours en 5 min")
    download.add_argument("--out", type=Path, required=True)
    download.add_argument("--symbol", help="Un seul symbole, sinon univers config.json")
    alp = commands.add_parser("alpaca-download", help="V12: télécharger/valider l'historique Alpaca SIP 5 min")
    alp.add_argument("--out", type=Path, required=True, help="Racine archive locale")
    alp.add_argument("--symbols", nargs="+", default=["NVDA", "SPY"])
    alp.add_argument("--start", default="2016-01-01")
    alp.add_argument("--end", default=None, help="Borne UTC optionnelle; limitée automatiquement à now-20min")
    alp.add_argument("--feed", default="sip", choices=("sip","iex"))
    alp.add_argument("--adjustments", nargs="+", default=["raw","split"], choices=("raw","split","dividend","all"))
    alp.add_argument("--safety-minutes", type=int, default=20)
    recl = commands.add_parser("alpaca-reclean", help="V12.1.2: re-nettoyer une archive Alpaca locale sans réseau")
    recl.add_argument("--data", type=Path, required=True, help="Racine archive Alpaca existante (source)")
    recl.add_argument("--out", type=Path, required=True, help="Racine archive nettoyée (peut être identique à --data)")
    recl.add_argument("--symbols", nargs="+", default=["NVDA", "SPY"])
    recl.add_argument("--adjustments", nargs="+", default=["raw","split"], choices=("raw","split","dividend","all"))
    research = commands.add_parser("research", help="Walk-forward avant la date du test ; crée un verrou")
    research.add_argument("--data", type=Path, required=True)
    research.add_argument("--symbol", default="SPY")
    research.add_argument("--timeframe", type=int, choices=(5, 10, 15), default=5)
    research.add_argument("--test-start", required=True, help="YYYY-MM-DD, première séance test incluse")
    research.add_argument("--test-end", required=True, help="YYYY-MM-DD, borne exclusive")
    research.add_argument("--out", type=Path, required=True)
    final = commands.add_parser("final", help="Évaluation finale : modèle, seuil et paramètres figés")
    final.add_argument("--data", type=Path, required=True)
    final.add_argument("--lock", type=Path, required=True)
    final.add_argument("--out", type=Path, required=True)
    audit10 = commands.add_parser("audit-v10", help="Audit V10 + manifest SHA-256")
    audit10.add_argument("--data", type=Path, required=True)
    audit10.add_argument("--symbol", default="SPY")
    audit10.add_argument("--out", type=Path, default=Path("runtime/data_audit.json"))
    dl = commands.add_parser("train-dl", help="Smoke test DL causal, DEVELOPMENT ONLY sur petit échantillon")
    dl.add_argument("--data", type=Path, required=True)
    dl.add_argument("--symbol", default="SPY")
    dl.add_argument("--context", type=int, default=24, choices=(12,24,48,78))
    dl.add_argument("--out", type=Path, default=Path("runtime/dl_smoke"))
    dl.add_argument("--architecture", choices=("mlp","tcn"), default="mlp")
    dl.add_argument("--epochs", type=int, default=3)
    dl.add_argument("--patience", type=int, default=2)
    dl.add_argument("--seed", type=int, default=42)

    doctor = commands.add_parser("doctor", help="Vérifier Python, PyTorch et CUDA")

    pipe = commands.add_parser("pipeline", help="Pipeline terminal complet: snapshot/audit logique, walk-forward ML+DL, backtest et stress")
    pipe.add_argument("--symbol", required=True)
    pipe.add_argument("--source", choices=("local","yahoo"), default="local")
    pipe.add_argument("--data", type=Path, help="Dossier CSV si --source local")
    pipe.add_argument("--timeframe", type=int, choices=(5,10,15), default=5)
    pipe.add_argument("--dl-models", nargs="*", default=["MLP_12","MLP_24","TCN_12","TCN_24"])
    pipe.add_argument("--seeds", nargs="+", type=int, default=[42])
    pipe.add_argument("--epochs", type=int, default=20)
    pipe.add_argument("--patience", type=int, default=4)
    pipe.add_argument("--dl-objective", choices=("direction","utility","economic"), default="economic", help="economic = V11.4 Direct Economic Learning: ExpectedTradeValue + backtest validation")
    pipe.add_argument("--max-sessions", type=int, default=None, help="Limiter aux N dernières séances (debug uniquement)")
    pipe.add_argument("--train-sessions", type=int, default=None, help="V12: séances minimales avant le premier fold (ex: 504)")
    pipe.add_argument("--fold-sessions", type=int, default=None, help="V12: taille OOS de chaque fold (ex: 20)")
    pipe.add_argument("--max-folds", type=_positive_int, default=None, help="Limiter le walk-forward aux N premiers folds")
    pipe.add_argument("--start-date", type=str, default=None, help="V12.1.3: première séance incluse (YYYY-MM-DD)")
    pipe.add_argument("--end-date", type=str, default=None, help="V12.1.3: dernière séance incluse (YYYY-MM-DD)")
    pipe.add_argument("--out", type=Path, required=True)

    btdl = commands.add_parser("backtest-dl", help="Walk-forward DL + règles/ML classiques sur un snapshot local")
    btdl.add_argument("--data", type=Path, required=True)
    btdl.add_argument("--symbol", required=True)
    btdl.add_argument("--timeframe", type=int, choices=(5,10,15), default=5)
    btdl.add_argument("--models", nargs="+", default=["MLP_12","MLP_24","TCN_12","TCN_24"])
    btdl.add_argument("--seeds", nargs="+", type=int, default=[42])
    btdl.add_argument("--epochs", type=int, default=20)
    btdl.add_argument("--patience", type=int, default=4)
    btdl.add_argument("--dl-objective", choices=("direction","utility","economic"), default="economic")
    btdl.add_argument("--max-folds", type=_positive_int, default=None, help="Limiter le walk-forward aux N premiers folds")
    btdl.add_argument("--out", type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        cfg = AppConfig.load(args.config if args.config.exists() else None)
        if args.command == "serve":
            if args.data:
                provider, mode = CSVProvider(args.data), "csv"
            elif args.yahoo:
                provider, mode = YahooProvider(), "yahoo"
            else:
                provider, mode = DemoProvider(), "demo"
            serve(LabService(provider, cfg, mode), args.port)
        elif args.command in ("download", "demo-data"):
            from .research.walkforward import make_output
            out = make_output(args.out)
            provider = DemoProvider() if args.command == "demo-data" else YahooProvider()
            symbols = [args.symbol] if getattr(args, "symbol", None) else cfg.symbols
            for symbol in symbols:
                snapshot = provider.fetch(symbol)
                write_snapshot(snapshot, out)
                print(f"{symbol} : {len(snapshot.prices)} bougies → {out}", flush=True)
            print("SYNTHÉTIQUE : aucune conclusion de trading." if args.command == "demo-data" else
                  "Snapshot figé. Réutiliser ce dossier pour la recherche ET le test final.")
        elif args.command == "alpaca-download":
            from .data.alpaca import download_archive
            for symbol in args.symbols:
                results = download_archive(symbol, args.out, start=args.start, end=args.end,
                                           adjustments=tuple(args.adjustments), feed=args.feed,
                                           safety_minutes=args.safety_minutes, calendar=cfg.calendar)
                for result in results:
                    print(json.dumps(result.__dict__, ensure_ascii=False), flush=True)
        elif args.command == "alpaca-reclean":
            from .data.alpaca import reclean_archive
            for symbol in args.symbols:
                results = reclean_archive(symbol, args.data, args.out,
                                          adjustments=tuple(args.adjustments), calendar=cfg.calendar)
                for result in results:
                    print(json.dumps(result.__dict__, ensure_ascii=False), flush=True)
        elif args.command == "research":
            from .research.walkforward import research as run_research
            snapshot = CSVProvider(args.data).fetch(args.symbol)
            table, lock = run_research(snapshot, cfg, args.timeframe, args.test_start, args.test_end, args.out)
            print(table[["model", "roc_auc", "final_capital", "max_drawdown", "sharpe"]].to_string(index=False))
            print(f"Modèle officiel figé : {lock['official_model']}\nVerrou : {args.out / 'protocol_lock.json'}")
        elif args.command == "final":
            from .research.walkforward import evaluate_final
            lock = json.loads(args.lock.read_text(encoding="utf-8"))
            snapshot = CSVProvider(args.data).fetch(lock["symbol"])
            table, lock = evaluate_final(snapshot, args.lock, args.out)
            print(table[["model", "official", "roc_auc", "final_capital", "max_drawdown", "sharpe"]].to_string(index=False))
            print(f"Modèle officiel inchangé : {lock['official_model']}\nRésultats : {args.out}")
        elif args.command == "audit-v10":
            from .v10_commands import audit_v10
            args.out.parent.mkdir(parents=True, exist_ok=True)
            report = audit_v10(args.data, args.symbol, args.out)
            print(json.dumps(report, indent=2, ensure_ascii=False))
        elif args.command == "train-dl":
            from .v10_commands import dl_smoke_from_data
            result = dl_smoke_from_data(args.data, args.symbol, args.out, args.context, args.architecture, args.epochs, args.patience, args.seed)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        elif args.command == "doctor":
            from .terminal_pipeline import doctor as run_doctor
            print(json.dumps(run_doctor(), indent=2, ensure_ascii=False))
        elif args.command == "pipeline":
            from .terminal_pipeline import run_terminal_pipeline
            run_terminal_pipeline(
                args.symbol, args.out, data_dir=args.data, source=args.source, timeframe=args.timeframe,
                dl_models=args.dl_models, seeds=args.seeds, epochs=args.epochs, patience=args.patience,
                config_path=args.config, max_sessions=args.max_sessions, dl_objective=args.dl_objective,
                train_sessions=args.train_sessions, fold_sessions=args.fold_sessions,
                start_date=args.start_date, end_date=args.end_date, max_folds=args.max_folds
            )
        elif args.command == "backtest-dl":
            from .terminal_pipeline import run_terminal_pipeline
            run_terminal_pipeline(
                args.symbol, args.out, data_dir=args.data, source="local", timeframe=args.timeframe,
                dl_models=args.models, seeds=args.seeds, epochs=args.epochs, patience=args.patience,
                config_path=args.config, dl_objective=args.dl_objective, max_folds=args.max_folds
            )
    except (ValueError, FileNotFoundError, RuntimeError, ImportError, OSError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        raise SystemExit(1)
