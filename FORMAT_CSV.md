# Format des snapshots

Un dossier contient `SYMBOL_5m.csv`, `SYMBOL_5m.meta.json`, et éventuellement un calendrier partagé `sessions.csv`. La commande `download` ou `demo-data` crée ce format automatiquement.

Le CSV de prix possède ces colonnes :

| Colonne | Convention |
|---|---|
| Timestamp | **Début** de la bougie 5 min, fuseau obligatoire ; exemple `2026-09-25T13:30:00+00:00` |
| Open | Premier prix de la bougie, strictement positif |
| High | Plus haut de la bougie |
| Low | Plus bas de la bougie |
| Close | Dernier prix de la bougie |
| Volume | Volume échangé, positif ou nul |

Dates croissantes et uniques, champs numériques finis, aucune interpolation. La source doit couvrir les heures régulières US. Le calendrier doit définir la séance à laquelle chaque barre appartient ; pas de conversion implicite de dates locales sans fuseau.

Métadonnées minimales pour `SPY_5m.meta.json` :

```json
{
  "symbol": "SPY",
  "source": "Nom du fournisseur réel",
  "synthetic": false,
  "bar_minutes": 5,
  "timestamp_convention": "bar_start",
  "currency": "USD",
  "adjustment": "raw",
  "fetched_at": "2026-09-25T20:05:00+00:00"
}
```

Ne pas déclarer des données synthétiques comme réelles. `fetched_at` doit représenter l'instant de capture, pas une date choisie pour faire passer une validation. Pour des snapshots fusionnés, toutes les bougies déclarées complètes doivent réellement l'avoir été au moment de la capture. Les snapshots constituent un historique observé, sans garantie automatique d'absence de révision ultérieure du fournisseur.

`sessions.csv` comporte `Session` (`YYYY-MM-DD`), `MarketOpen` et `MarketClose` (dates avec fuseau). Il peut inclure des demi-séances. S'il est absent, le fournisseur CSV demande le calendrier NYSE à `pandas_market_calendars`. La bibliothèque de calendrier est alors nécessaire même hors ligne.

Les CSV de la V7 quotidienne ne peuvent pas être renommés en CSV intraday. Ils restent utilisables uniquement avec leur pipeline d'origine. Le backtest V9 refuse des données de séance incomplètes ; conserver les erreurs et traiter les trous à la source plutôt que les masquer.

Les fichiers `trades_*.csv` gardent leur en-tête même lorsqu'aucun trade n'a été exécuté. Un tel résultat n'est pas une erreur : les scores peuvent rester sous le seuil fixé.
