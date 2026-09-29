# Trading AI V11.2 Terminal

Version terminal-only orientée backtesting et Deep Learning.

## Changements V11.2

- Objectif DL multitâche conservé : TargetBeforeStop 15/30/60, MFE/MAE 30, NetReturn 15/30/60.
- Correction du déséquilibre de classes via `pos_weight` calculé uniquement sur le train interne, borné à 20.
- Correction importante du protocole DL : l'early stopping et le choix du seuil n'utilisent plus le fold OOS. Chaque fold externe contient maintenant un sous-train et une validation interne chronologique avec purge 60 min.
- Score de setup combiné : TargetBeforeStop 30/60 + MFE/MAE + NetReturn 30/60.
- Seuil sélectionné uniquement sur validation interne, dans une grille bornée. Si aucun seuil n'a une utilité nette positive, le fold force NO_TRADE.
- Le score est recentré de façon monotone vers le seuil d'exécution global sans regarder le test OOS.
- `dl_threshold_selection.csv` trace les seuils et diagnostics de chaque fold.
- `dl_training.csv` trace les poids de classe, les tailles train/validation/test et les checkpoints.
- Aucune promotion automatique. Les scores ne sont pas des probabilités calibrées.

## Commande recommandée NVDA

```bash
./trading-ai pipeline \
  --symbol NVDA \
  --source yahoo \
  --timeframe 5 \
  --dl-objective utility \
  --dl-models MLP_12 TCN_12 \
  --seeds 42 123 777 \
  --epochs 30 \
  --patience 5 \
  --out runs/NVDA_V112_01
```

## Fichiers à analyser

- `metrics.csv`
- `review_gate.csv`
- `stress_costs.csv`
- `utility_target_stats.csv`
- `dl_threshold_selection.csv`
- `trade_concentration.csv`
- `dl_training.csv`
- `utility_predictions_oos.csv`

`CASH` reste la référence conservatrice. Une stratégie positive sur quelques trades n'est pas considérée comme validée.
