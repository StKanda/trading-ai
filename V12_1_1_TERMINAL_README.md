# Trading AI V12.1.1 Terminal

Patch data-quality sans nouveau téléchargement Alpaca.

## Nouvelle commande

```bash
./trading-ai alpaca-reclean \
  --symbols NVDA SPY \
  --data /chemin/vers/archive/alpaca/existante \
  --out data/alpaca
```

La commande relit uniquement les fichiers `*_extended_5m.csv`, réapplique le calendrier V12.1 et les règles `MARKET_HALT_COMPLETE`, puis reconstruit les snapshots propres et `session_quality.csv`. Aucun appel réseau et aucune clé API ne sont nécessaires.

`--out` peut être identique à `--data` pour nettoyer sur place. Si la destination est différente, les archives étendues sont copiées afin que la nouvelle archive reste autonome.
