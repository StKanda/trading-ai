# Trading AI V11.3.2 Terminal

Version de diagnostic du candidate ranker V11.3.x.

## Nouveaux artefacts

- `dl_threshold_grid.csv`: chaque seuil de validation testé, avec signaux, trades, séances, rendement, expectancy, profit factor, drawdown, concentration et raison de rejet.
- `dl_raw_oos_metrics.csv`: qualité de ranking des scores DL bruts OOS, même si le gate final choisit `NO_TRADE`. Les scores ne sont pas des probabilités calibrées.
- `CANDIDATE_ALL` dans `metrics.csv`: baseline diagnostique qui trade tous les setups candidats sans ranking DL. Elle est exclue du gate de promotion.
- `candidate_oracle_diagnostic.csv`: résultat hypothétique isolé de chaque candidat OOS avec le moteur réel. Cet oracle utilise le futur uniquement pour diagnostic et ne doit jamais être utilisé comme signal.
- `candidate_oracle_summary.csv`: résumé de la densité d'opportunités économiquement positives dans l'univers candidat.

La logique d'exécution reste: signal à la clôture t, entrée à l'open t+1, pas d'overnight, stop prioritaire si stop et target sont touchés dans la même bougie.
