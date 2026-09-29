# Trading AI V11 — Terminal first

Cette version recentre le projet sur le moteur de recherche/backtest et le Deep Learning. Le dashboard n'est pas requis pour travailler.

## Commandes principales

```bash
./trading-ai doctor
./trading-ai download --symbol NVDA --out data/NVDA
./trading-ai audit-v10 --data data/NVDA --symbol NVDA --out runs/audit_nvda.json
./trading-ai train-dl --data data/NVDA --symbol NVDA --architecture tcn --context 24 --epochs 20 --patience 4 --seed 42 --out runs/train_nvda_tcn24
./trading-ai backtest-dl --data data/NVDA --symbol NVDA --timeframe 5 --models MLP_12 MLP_24 TCN_12 TCN_24 --seeds 42 123 777 --epochs 20 --patience 4 --out runs/nvda_wf
./trading-ai pipeline --symbol NVDA --source yahoo --timeframe 5 --dl-models MLP_12 MLP_24 TCN_12 TCN_24 --seeds 42 123 777 --epochs 20 --patience 4 --out runs/NVDA_run_001
```

`pipeline --source yahoo` fige d'abord un snapshot Yahoo voisin du dossier de résultats, puis réutilise exclusivement ce snapshot pendant le run.

## Ce que fait `pipeline`

1. charge ou fige les données ;
2. construit les features causales ;
3. crée les folds chronologiques avec purge/embargo ;
4. entraîne RF/GB/LR et règles déterministes ;
5. entraîne MLP/TCN fold par fold, scaler fit uniquement sur le train ;
6. moyenne les scores quand plusieurs seeds sont demandées ;
7. backteste tous les scores avec la même mécanique: signal close `t`, entrée open `t+1`, stop prioritaire, aucun overnight ;
8. applique les coûts/slippage et des stress +2/+5 bps par côté ;
9. écrit les métriques, trades, ledgers, folds, checkpoints et rapports.

## Sorties

Un run contient notamment:

- `metrics.csv`
- `stress_costs.csv`
- `review_gate.csv`
- `predictions_oos.csv`
- `labels_oos.csv`
- `folds.csv`
- `dl_training.csv`
- `trades_<MODEL>.csv`
- `ledger_<MODEL>.csv`
- `capital_daily.csv`
- `bootstrap_daily.csv`
- `checkpoints/...`
- `run.json`

## Limites importantes

- Statut **DEVELOPMENT ONLY**.
- Aucun ordre broker, aucune exécution réelle.
- Les scores ML/DL ne sont pas présentés comme des probabilités calibrées.
- La cible DL actuelle reste le signe du rendement net futur à horizon fixe. Une cible multi-horizon / target-before-stop est une étape ultérieure.
- Aucune promotion automatique: `review_gate.csv` ne crée que `CANDIDATE_FOR_REVIEW` ou `REJECTED`.
- Avec quelques dizaines de séances, les résultats ne suffisent pas pour conclure à un edge robuste.


## V11.0.1
Un snapshot Yahoo capturé pendant la séance peut contenir la séance courante partielle. Le pipeline terminal exclut maintenant uniquement les séances de fin qui ne sont pas encore clôturées à `fetched_at`, tout en continuant à refuser les trous dans les séances déjà clôturées.

## V11.1
Voir `V11_1_TERMINAL_README.md`. Le mode DL par défaut devient `utility`: target-before-stop 15/30/60 min, MFE/MAE 30 min et rendements nets 15/30/60 min, avec purge 60 min et diagnostics de concentration du PnL.
