# Trading AI V10.1

Patch interface V10.1 :

- choix du symbole et watchlist configurable ;
- source `Local CSV` ou `Yahoo` ;
- graphique chandeliers SPY/action 5 min ;
- VWAP ;
- support et résistance causaux ;
- Opening Range 20 min ;
- patterns W/M confirmés et breakout ;
- géométrie W/M optionnelle (désactivée par défaut pour éviter une lecture rétroactive) ;
- entrée, stop, target du signal courant ;
- entrées/sorties provenant du ledger de backtest lorsqu'il existe ;
- volume ;
- score du Champion et seuil ;
- fenêtres 1 séance / 2 jours / 5 jours ;
- état runtime isolé par symbole.

Aucun ordre courtier n'est envoyé. Le dashboard est un outil de recherche/paper analysis.

Le `research_worker.py` reste dans cette version le smoke-test borné V10.0.x; il ne doit pas être interprété comme un moteur de découverte de stratégie réel.
