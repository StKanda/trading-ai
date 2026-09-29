# Trading AI V11.3.1 Terminal

Version terminal-only orientée backtesting + Deep Learning. Aucun ordre broker.

## Changement principal

V11.3 ne demande plus au DL de scanner indistinctement toutes les bougies.
Le pipeline est maintenant :

1. détection **causale** de setups candidats (règles existantes + contexte support/VWAP/W/momentum) ;
2. entraînement MLP/TCN multitâches seulement sur ces setups ;
3. ranking DL des setups candidats ;
4. choix du seuil uniquement sur la validation interne via le **même simulateur économique** que le test OOS ;
5. si aucun seuil ne produit PnL et expectancy positifs sur validation : `NO_TRADE` ;
6. application du seuil figé au fold OOS ;
7. stress coûts + concentration du PnL + review gate.

La validation économique inclut entrée open t+1, frais, slippage, stop, target, time-exit, pas d'overnight et limite le nombre de trades par séance.

## Commande NVDA

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
  --out runs/NVDA_V113_01
```

Ou : `./run_pipeline_nvda_v11_3.sh`.

## Nouveaux artefacts

- `candidate_setups.csv`
- `candidate_setup_stats.csv`
- `dl_threshold_selection.csv` avec PnL, drawdown, trades et win rate de validation
- fichiers V11.2 conservés : metrics, stress, concentration, checkpoints, utility predictions.

Les scores DL ne sont pas des probabilités validées/calibrées.
