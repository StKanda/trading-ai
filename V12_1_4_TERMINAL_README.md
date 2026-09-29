# Trading AI V12.1.4 Terminal

V12.1.4 ajoute le découpage temporel explicite du pipeline pour les runs rapides et les validations intermédiaires.

## Nouveau

- `pipeline --start-date YYYY-MM-DD`
- `pipeline --end-date YYYY-MM-DD`
- bornes inclusives sur les séances, appliquées avant features, labels et walk-forward
- validation de l'ordre des dates et des périodes vides
- `period_used.json` dans chaque run pour tracer la période réellement utilisée

## Exemple 2025

```bash
./trading-ai pipeline \
  --symbol NVDA \
  --source local \
  --data data/alpaca/NVDA/split \
  --timeframe 5 \
  --dl-objective economic \
  --dl-models MLP_12 TCN_12 \
  --seeds 42 \
  --epochs 10 \
  --patience 3 \
  --train-sessions 126 \
  --fold-sessions 21 \
  --start-date 2025-01-01 \
  --end-date 2025-12-31 \
  --out runs/NVDA_V1213_2025
```
