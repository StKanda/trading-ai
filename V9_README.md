# Trading AI V9 — SPY intraday 5 minutes

V9 est une branche expérimentale séparée de V8. Elle ne contient aucune connexion courtier et ne passe aucun ordre.

## Ce que V9 ajoute

- Random Forest et Gradient Boosting V8 conservés.
- Régression logistique (`LR_A`, `LR_D`) comme baseline ML plus simple.
- Trois règles causales : momentum, reclaim VWAP, breakout de l'opening range 20 minutes.
- Walk-forward purgé/embargo ; signal à la clôture, entrée à l'ouverture suivante.
- Aucune position overnight ; stop prioritaire si stop et objectif sont touchés dans la même bougie.
- Frais/slippage de base + stress avec **+2 et +5 bps de slippage par côté**.
- Statistiques de trades, bootstrap descriptif par séance, ablations et courbes de capital.
- Sélection pré-test conservatrice. Si aucun candidat actif ne passe les critères de robustesse, le modèle officiel est `CASH`.
- Verrou cryptographique du protocole et des fichiers Python avant le test final.

Les scores RF/GB/LR sont des **scores de modèle non calibrés**, pas des probabilités validées.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Pour Yahoo/yfinance uniquement :

```bash
python -m pip install -r requirements_market.txt
```

## Audit des données

```bash
python audit_data.py --data /chemin/data_spy --symbol SPY --out audit.json
```

Une ligne de `sessions.csv` sans prix est simplement signalée dans `calendar_sessions_without_prices`. Elle n'est pas transformée en données de marché.

## Lancement recommandé, en deux phases

Choisir d'abord la fenêtre finale **sans regarder ses performances**. `--test-end` est une borne exclusive.

```bash
python main.py research \
  --data /chemin/data_spy \
  --symbol SPY \
  --timeframe 5 \
  --test-start 2026-09-14 \
  --test-end 2026-09-26 \
  --out results_v9_research
```

Puis seulement :

```bash
python main.py final \
  --data /chemin/data_spy \
  --lock results_v9_research/protocol_lock.json \
  --out results_v9_final
```

Commande tout-en-un (pratique, mais garde la discipline de ne pas modifier le protocole après lecture du final) :

```bash
./run_v9_test.sh /chemin/data_spy 2026-09-14 2026-09-26
```

## Fichiers importants produits

- `metrics.csv` : performance et statistiques de trades.
- `cost_stress.csv` : sensibilité au slippage.
- `bootstrap_daily.csv` : incertitude descriptive par resampling de séances.
- `ablation_deltas.csv` : contribution descriptive des groupes de features RF/GB.
- `selection_audit.csv` : critères utilisés **avant** le test final.
- `split_audit.csv` : purge/embargo et frontières de folds.
- `trades_*.csv`, `ledger_*.csv`, `predictions.csv`.
- `capital.png`, `capital_daily.csv`.
- `protocol_lock.json` puis `protocol_lock_used.json`.

## Limite statistique

Quelques dizaines de séances servent à vérifier le pipeline, pas à démontrer un edge. V9 marque explicitement moins de 100 séances évaluées comme insuffisantes pour une conclusion statistique. Pour une étude sérieuse, utiliser idéalement 1–2 ans ou davantage de données 5 minutes couvrant plusieurs régimes de marché.
