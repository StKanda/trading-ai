# Trading AI V11.4 Terminal — Direct Economic Learning

V11.4 conserve le protocole causal V11.x et remplace le ranking utility par un apprentissage économique direct sur les setups candidats.

## Cible DL

Pour chaque setup candidat causal, le même simulateur que le backtest construit un trade isolé avec : signal à la clôture t, entrée à l'open t+1, stop prioritaire, target, time-exit, frais, slippage et no overnight.

Le réseau apprend :

- `ProfitableAfterCosts`
- `StopHit`
- `TargetHit`
- `TimeExit`
- `RealizedPnL`
- `RealizedReturn`
- `MFE`
- `MAE`

La tête centrale de ranking est `ExpectedTradeValue`, c'est-à-dire le `RealizedPnL` prédit. Il est transformé monotonement en score [0,1] uniquement pour rester compatible avec le moteur de seuil. `0.50` correspond à un PnL prédit nul. Ce score n'est pas une probabilité calibrée.

Les têtes de classification sont auxiliaires et ne sont pas mélangées manuellement au score final.

## Validation

Le scaler, la normalisation des régressions et les `pos_weight` sont calculés uniquement sur le train interne. Le seuil est choisi uniquement par backtest économique sur validation interne, puis figé pour l'OOS. Si aucun seuil ne passe les garde-fous, le fold produit `NO_TRADE`.

## Fichiers V11.4

- `economic_labels_oos.csv`
- `economic_predictions_oos.csv`
- `economic_target_stats.csv`
- `dl_economic_oos_metrics.csv`
- `dl_threshold_grid.csv`
- `dl_threshold_selection.csv`
- `candidate_setups.csv`
- `candidate_oracle_diagnostic.csv` — diagnostic futur-fuyant, jamais signal
- `candidate_oracle_summary.csv`

## NVDA

```bash
./trading-ai pipeline \
  --symbol NVDA \
  --source yahoo \
  --timeframe 5 \
  --dl-objective economic \
  --dl-models MLP_12 TCN_12 \
  --seeds 42 123 777 \
  --epochs 30 \
  --patience 5 \
  --out runs/NVDA_V114_01
```

V11.4 reste un environnement de recherche/backtest. Aucun ordre broker n'est envoyé et aucun résultat ne constitue une preuve d'edge tradable.
