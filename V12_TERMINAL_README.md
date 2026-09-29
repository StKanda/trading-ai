# Trading AI V12 — Maximum Historical Data / Robust Walk-Forward

V12 garde le moteur économique V11.4 et remplace la contrainte Yahoo ~60 jours par une archive historique Alpaca SIP 5 min.

## Sécurité

- Recherche/backtest uniquement.
- Aucun endpoint d'ordre/broker dans le downloader V12.
- Les clés Alpaca sont lues uniquement depuis `APCA_API_KEY_ID` et `APCA_API_SECRET_KEY`.
- Ne jamais enregistrer les clés dans `config.json`, les CSV ou les logs.

## 1. Télécharger l'archive maximale

```bash
export APCA_API_KEY_ID="..."
export APCA_API_SECRET_KEY="..."

./trading-ai alpaca-download \
  --symbols NVDA SPY \
  --start 2016-01-01 \
  --feed sip \
  --adjustments raw split \
  --out data/alpaca
```

La borne de fin est automatiquement limitée à `now - 20 min` pour éviter la restriction SIP récente du plan Basic.

Arborescence :

```text
data/alpaca/NVDA/raw/
data/alpaca/NVDA/split/
data/alpaca/NVDA/audit/session_quality.csv
data/alpaca/SPY/raw/
data/alpaca/SPY/split/
data/alpaca/SPY/audit/session_quality.csv
```

Chaque dossier `raw`/`split` contient :
- `SYMBOL_extended_5m.csv` : réponse Alpaca conservée avec extended hours ;
- `SYMBOL_5m.csv` : RTH propre, séances complètes uniquement ;
- `sessions.csv` : calendrier exact des séances retenues, early closes inclus ;
- métadonnées JSON.

Aucune bougie manquante n'est interpolée. Une séance avec un trou est exclue et journalisée.

## 2. Walk-forward long historique

Premier run recommandé : deux années minimum avant le premier fold, folds trimestriels pour garder un coût DL raisonnable.

```bash
./trading-ai pipeline \
  --symbol NVDA \
  --source local \
  --data data/alpaca/NVDA/split \
  --timeframe 5 \
  --dl-objective economic \
  --dl-models MLP_12 TCN_12 \
  --seeds 42 123 777 \
  --epochs 30 \
  --patience 5 \
  --train-sessions 504 \
  --fold-sessions 63 \
  --out runs/NVDA_V12_01
```

Pour SPY, remplacer le symbole et le chemin.

`train_sessions` est la quantité minimale d'historique disponible avant le premier test OOS. L'entraînement reste causal/expanding; `fold_sessions` contrôle seulement la taille de chaque bloc OOS.

## Notes méthodologiques

- Signal close t, exécution open t+1.
- Stop prioritaire si stop et target touchent la même bougie.
- Pas d'overnight.
- Coûts/slippage conservés.
- Scores de modèle non présentés comme probabilités calibrées.
- `split` est destiné à la recherche multi-années; `raw` reste archivé pour audit.
- L'archive ne remplit jamais les trous OHLCV.
