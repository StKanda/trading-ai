# Trading AI V12.1.2 Terminal

Correctif de propreté d'audit par rapport à V12.1.1.

## Changement

`alpaca-reclean` borne désormais le calendrier aux premières/dernières dates locales qui contiennent réellement des barres RTH (09:30–16:00 America/New_York). Les barres extended-hours situées près de la frontière UTC d'une archive ne peuvent donc plus créer une fausse séance précédente avec 78 barres manquantes, comme `2015-12-31` pour une archive démarrant à `2016-01-01T00:00Z`.

Aucun remplissage de bougie n'est effectué. Les règles `MARKET_HALT_COMPLETE`, `EARLY_CLOSE_COMPLETE`, `MISSING_BARS` et `OPEN_OR_FUTURE` restent inchangées.

## Re-clean sans réseau

```bash
./trading-ai alpaca-reclean \
  --symbols NVDA SPY \
  --data /var/home/giron/trading-ai/trading_ai_v12/data/alpaca \
  --out data/alpaca
```
