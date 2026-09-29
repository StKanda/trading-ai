# Trading AI V12.1 — Data Quality Fix

V12.1 conserve V12 et corrige l’audit des séances historiques.

- distingue `COMPLETE`, `EARLY_CLOSE_COMPLETE`, `MARKET_HALT_COMPLETE`, `MISSING_BARS`, `OPEN_OR_FUTURE`;
- reconnaît les quatre interruptions Market-Wide Circuit Breaker de mars 2020 à partir de leurs bar-starts 5 min absents;
- ne crée ni n’interpole aucune bougie;
- conserve les séances de halt dans le dataset propre en retirant uniquement les timestamps officiellement non négociés de la grille attendue;
- les vrais trous Alpaca restent exclus.
