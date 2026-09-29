V12.1.2 corrige aussi les fausses séances de frontière UTC lors de `alpaca-reclean`. Voir `V12_1_2_TERMINAL_README.md`.

# Trading AI V12.1.2

V12.1.2 ajoute `alpaca-reclean` pour reconstruire localement les datasets propres et les audits à partir des archives `*_extended_5m.csv`, sans retéléchargement Alpaca. Voir `V12_1_1_TERMINAL_README.md`.

# Trading AI V11.3.2 Terminal

Voir `V11_3_TERMINAL_README.md`. Priorité: backtesting + Deep Learning terminal-only.

# Trading AI V11 — Terminal first

Priorité: backtesting walk-forward + Deep Learning, sans dépendre du dashboard.

Démarrage rapide:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements_market.txt
python -m pip install -r requirements_dl.txt
./trading-ai doctor
```

Pipeline complet NVDA / Yahoo:

```bash
./run_pipeline_nvda.sh
```

Ou commande explicite:

```bash
./trading-ai pipeline --symbol NVDA --source yahoo --timeframe 5 \
  --dl-models MLP_12 MLP_24 TCN_12 TCN_24 \
  --seeds 42 123 777 --epochs 20 --patience 4 \
  --out runs/NVDA_run_001
```

Voir `V11_TERMINAL_README.md` pour les détails.

Aucune connexion broker. Aucun ordre réel. Statut DEVELOPMENT ONLY.

## V11.1 Terminal DL utility
Pour le moteur terminal actuel, voir `V11_1_TERMINAL_README.md`.

## V11.4 Terminal

La version terminal actuelle est décrite dans `V11_4_TERMINAL_README.md`. Le mode recommandé pour le DL est `--dl-objective economic`.
