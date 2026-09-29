# Validation de construction V9

Validation effectuée dans l'environnement isolé de la conversation :

- `python -m unittest discover -s tests -v` : **34/34 tests OK**.
- Audit du snapshot fourni : **3 198 bougies**, **41 séances de prix**, 78 bougies par séance, 0 doublon, 0 NaN, 0 trou de grille, OHLCV cohérents.
- La séance calendrier du 28 septembre 2026 sans prix est signalée mais n'est pas utilisée comme donnée de marché.
- Smoke test `research` 5 min avec test final 14–25 septembre : pipeline complet jusqu'au verrou, modèle officiel pré-test = `CASH`.
- Smoke test `final` : exécution complète avec le verrou, modèle officiel inchangé = `CASH`.

Ces performances ne constituent pas un nouveau test indépendant : le snapshot fourni a déjà été observé pendant le développement. Utiliser un historique plus long et/ou des données ultérieures réellement inédites pour une conclusion statistique.

## Correctif V10.0.1

- PyTorch est optionnel pour le coeur V10 : son absence ne casse plus l'import de `test_v10` ni `audit-v10`.
- `requirements_dl.txt` installe PyTorch 2.10+ pour activer les commandes Deep Learning.
- Le dashboard désactive le prompt e-mail Streamlit et la télémétrie via `.streamlit/config.toml`.
- 45/45 tests passent après correctif.


## V10.1.0 UI validation

- 48/48 automated tests passed.
- Real SPY snapshot: 3,198 bars loaded; chart builder produced 11 Plotly traces over two sessions.
- Symbol-scoped runtime state and local symbol discovery are covered by tests.
- `research_worker.py` remains a smoke worker in this patch and must not be interpreted as a real automated research engine.

## V11.1 Terminal Utility DL
- 60 tests automatisés réussis, 0 échec.
- Smoke pipeline MLP_12 utility: snapshot local SPY, 30 séances, 3 folds, backtest économique terminé.
- Smoke pipeline TCN_12 utility: snapshot local SPY, 20 séances, 1 fold, backtest économique terminé.
- Cibles utility: target-before-stop 15/30/60 min, MFE/MAE 30 min, rendements nets 15/30/60 min.
- Purge utility basée sur LabelEnd_60 et scaler/normalisation appris sur le train uniquement.
- Les sorties restent DEVELOPMENT ONLY et ne constituent pas une preuve d'edge.

## V11.2 Terminal
- Le fold OOS n'est plus utilisé pour early stopping ni choix de seuil DL.
- Sous-train / validation interne chronologique avec purge fondée sur l'horizon 60 min.
- `pos_weight` calculé exclusivement sur sous-train.
- Seuil choisi sur validation interne dans une grille bornée; absence d'utilité positive => NO_TRADE.
- 63 tests passent et un smoke pipeline utility MLP_12 complet a été exécuté.

## V11.3.1 Terminal
- filtre causal de setups candidats avant DL ;
- ranking MLP/TCN uniquement sur ces candidats ;
- seuil choisi sur validation interne par le simulateur économique exact, jamais par le fold OOS ;
- `NO_TRADE` si aucun seuil n'a PnL/expectancy positifs après coûts ;
- audit `candidate_setup_stats.csv` et `dl_threshold_selection.csv` ;
- 65/65 tests passés dans l'environnement de build.


## V11.3.2 Terminal
- `dl_threshold_grid.csv` conserve tous les seuils de validation et leur motif de rejet.
- `dl_raw_oos_metrics.csv` mesure le ranking brut même si le gate final choisit `NO_TRADE`; score != probabilité calibrée.
- `CANDIDATE_ALL` teste l'univers de setups sans ranking DL et reste exclu de la promotion.
- L'oracle candidat est strictement diagnostique, futur-fuyant, et n'est jamais utilisé comme signal.
- 71/71 tests passés.
- Smoke local SPY: 39 séances, 5 folds, MLP_12, pipeline complet en mode DL-only/classic-off pour valider les nouveaux diagnostics.
