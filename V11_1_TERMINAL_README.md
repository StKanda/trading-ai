# Trading AI V11.1 — Terminal DL utility

V11.1 reste **terminal-first** et se concentre sur le backtesting walk-forward et le Deep Learning. Aucun ordre broker n'est envoyé.

## Changement principal

Le DL peut maintenant être entraîné avec `--dl-objective utility` (valeur par défaut) au lieu de la seule direction future.

Pour chaque signal connu à la clôture `t`, les cibles utilisent une entrée théorique à l'open `t+1`, sans overnight, avec la même priorité pessimiste stop-avant-target que le backtester.

Le réseau multi-tâches apprend simultanément :

- `TargetBeforeStop_15`
- `TargetBeforeStop_30`
- `TargetBeforeStop_60`
- `MFE_30`
- `MAE_30`
- `NetReturn_15`
- `NetReturn_30`
- `NetReturn_60`

Le score utilisé par le backtest DL en mode `utility` est **TargetBeforeStop_30**. C'est un score de modèle, **pas une probabilité calibrée**.

Les horizons 15/30/60 min servent aussi de tâches auxiliaires pour obliger le réseau à apprendre davantage que le simple signe du prochain rendement.

## Purge / anti-leakage

Le mode utility applique une purge basée sur la cible la plus longue, 60 minutes. Une ligne ne peut être utilisée en train que si `LabelEnd_60` est strictement antérieur à la frontière du fold.

Le scaler des features et la normalisation des cibles de régression sont appris uniquement sur le train du fold.

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
  --out runs/NVDA_UTILITY_01
```

Ou :

```bash
./run_pipeline_nvda_v11_1.sh
```

## Sorties nouvelles

En plus des sorties V11 :

- `utility_labels_oos.csv` : vraies cibles hors échantillon ;
- `utility_predictions_oos.csv` : sorties multi-tâches hors échantillon ;
- `utility_target_stats.csv` : fréquence des target-before-stop et distributions MFE/MAE/returns ;
- `trade_concentration.csv` : dépendance du PnL aux 1/3 meilleurs trades ;
- `dl_training.csv` : losses/checkpoints par fold et seed, avec `objective=utility_multitask`.

## Lire les sorties

```bash
cat runs/NVDA_UTILITY_01/metrics.csv
cat runs/NVDA_UTILITY_01/review_gate.csv
cat runs/NVDA_UTILITY_01/stress_costs.csv
cat runs/NVDA_UTILITY_01/utility_target_stats.csv
cat runs/NVDA_UTILITY_01/trade_concentration.csv
cat runs/NVDA_UTILITY_01/dl_training.csv
```

## Important

- **DEVELOPMENT ONLY**.
- `CASH` reste une référence valide.
- Aucune promotion automatique d'un modèle.
- Le score DL n'est pas présenté comme une probabilité validée.
- Le mode utility peut produire très peu de positifs si le target 2R est rarement atteint avant le stop. `utility_target_stats.csv` est justement là pour rendre ce problème visible au lieu de le masquer.
- Un petit nombre de trades ou un résultat concentré sur quelques trades ne constitue pas une preuve d'edge.
