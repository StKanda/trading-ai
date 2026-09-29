# Trading AI V11.3.1 Terminal

Patch méthodologique de V11.3.

- seuil DL choisi uniquement sur validation interne via le backtest économique ;
- minimum 5 trades de validation ;
- minimum 2 séances de validation avec trades ;
- concentration maximale du PnL absolu sur le meilleur trade : 60 % ;
- `dl_threshold_selection.csv` expose `selected_threshold`, `applied_oos_threshold_raw`, `backtester_threshold_adjusted` et les contraintes de validation ;
- `NO_TRADE` si aucune configuration ne satisfait toutes les contraintes ;
- aucune exécution broker.
