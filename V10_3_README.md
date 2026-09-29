# Trading AI V10.3 — Interface de pilotage

V10.3 transforme le dashboard en cockpit de recherche/paper analysis. Aucune exécution courtier n'est ajoutée.

## Changements principaux

- Le zoom à la molette Plotly est désactivé : la molette fait défiler la page, le zoom se fait avec les contrôles du graphique.
- Supports/résistances causaux remis en avant avec cartes de niveau et zones techniques « probables » autour du dernier niveau confirmé. « Probable » est un terme heuristique, pas une probabilité calibrée.
- Les entrées/sorties peuvent être superposées pour n'importe quel `trades_<MODEL>.csv`, pas uniquement le Champion courant.
- L'onglet Backtest permet de choisir symbole courant, timeframe 5/10/15 min, période de test, puis de lancer le vrai `main.py research` en arrière-plan.
- Le journal du backtest et ses résultats (metrics, capital, trades) sont actualisés dans le dashboard.
- L'onglet Deep Learning permet de lancer MLP ou TCN, choisir contexte, epochs max, patience et seed. CUDA reste automatique.
- Pour Yahoo, chaque calcul capture un snapshot figé dans `runtime/ui_snapshots/` avant le backtest/DL afin de garder la reproductibilité.
- Les jobs sont suivis dans `runtime/ui_jobs/` et peuvent être arrêtés depuis l'interface.

## Limites

- Le DL lancé dans l'interface reste expérimental : il n'est pas encore intégré comme challenger au walk-forward économique RF/GB/LR/règles.
- `research_worker.py` reste un smoke worker borné et n'est pas encore un moteur autonome de découverte de stratégies.
- Les données Yahoo sont du polling avec latence non garantie, pas un flux broker-grade tick-by-tick.
- Les zones support/résistance n'utilisent que des pivots déjà confirmés; aucun look-ahead n'est ajouté.
