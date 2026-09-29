# Trading AI V10

## Statut

Recherche / backtest / paper analysis uniquement. **Aucun broker, aucun ordre réel.**

Ordre de priorité : causalité > qualité des données > absence de fuite > robustesse > reproductibilité > gestion du risque > simplicité > explicabilité > performance.

## Ce que V10 conserve de V9

- RF, Gradient Boosting, Logistic Regression.
- Momentum, VWAP reclaim, ORB20.
- Buy & Hold, Intraday Hold, CASH.
- Features causales, W/M, supports/résistances, VWAP, volume relatif.
- Walk-forward chronologique, purge/embargo.
- Signal à la clôture t, entrée au plus tôt à l'ouverture t+1.
- Aucune position overnight.
- Stop prioritaire si stop et target touchés dans la même bougie.
- Gaps capables de dépasser le budget de perte.
- Frais/slippage et stress de coûts.
- `protocol_lock.json` avant le final test.
- Sélection conservatrice pouvant retenir CASH.

## Ajouts V10

### Data freshness

`trading_lab/data/freshness.py` mesure la dernière bougie complète attendue et classe le flux FRESH / WARNING / DELAYED / STALE. DELAYED/STALE bloque les signaux live. `market_timestamp` et `generated_at` restent distincts.

`trading_lab/data/provider_chain.py` permet un primary/fallback. Les providers ne sont jamais fusionnés silencieusement; un désaccord important provoque `DATA PROVIDER DISAGREEMENT`.

### Position sizing

`detailed_position_size()` expose : budget de risque, risque/unité, cap risque, cap levier, cap volume, notional, planned risk, daily risk remaining. Un stop n'est jamais présenté comme une garantie : un gap peut augmenter le realized risk.

### Auto-research borné

`trading_lab/research_engine/` fournit :

- registre SQLite de toutes les expériences ;
- détection des doublons ;
- budgets stricts ;
- arrêt après N non-améliorations ;
- absence d'élargissement automatique de l'espace de recherche ;
- aucune promotion silencieuse.

Le worker fourni valide l'infrastructure avec des expériences smoke `DEVELOPMENT ONLY`. Il ne prétend pas constituer un optimiseur rentable.

### Validation Engine

`trading_lab/validation_engine/` contient des primitives de concentration du PnL, stabilité de voisinage/plateau et gate multi-critères. Le rôle du module est d'essayer de casser un challenger, pas de maximiser son backtest.

### Deep Learning

`trading_lab/deep_learning/` fournit un MLP séquentiel et un TCN causal léger sous PyTorch. PyTorch est optionnel pour le coeur V10 et s'installe via `requirements_dl.txt`; les imports non-DL et les tests de causalité ne doivent pas échouer s'il est absent. Le scaler est ajusté uniquement sur le train. Early stopping, seed, AdamW et checkpoint sont pris en charge.

Avec ~41 séances, le résultat doit rester **DEVELOPMENT ONLY**. Le smoke test vérifie l'architecture, pas un edge.

### Live Worker

`worker.py` est paper-only. Il vérifie la fraîcheur, calcule les features, évalue un champion de type règle, calcule le sizing et écrit `runtime/runtime_state.json`. Il n'entraîne pas et ne passe aucun ordre.

L'export live de champions ML/DL n'est pas automatisé dans cette première V10 : il devra charger un artefact explicitement promu/checkpointé. Le worker refuse de fabriquer ce comportement.

### Replay

`replay.py` rejoue une séance complète avec les mêmes features/règles causales. Les tests hérités de prefix-invariance protègent contre l'utilisation du futur. `RECORDED_LATENCY` est prévu dans le cahier des charges mais nécessite un historique réel de latence; V10 fournit actuellement le mode `IDEAL_DATA` et le schéma de `latency_log.csv` via le module freshness.

### Dashboard

`app.py` est une interface Streamlit séparée des calculs. Elle lit l'état runtime, les CSV de résultats et SQLite; elle ne réentraîne pas les modèles à chaque refresh.

## Limites actuelles

- Le snapshot de vérification a 3 198 bougies sur **41 séances**, donc insuffisant pour conclure statistiquement.
- Le 28/09/2026 est présent dans le calendrier sans prix : il est exclu des observations de marché.
- Le final 14–25/09/2026 a déjà été consulté dans les versions précédentes : il ne constitue pas un nouveau final indépendant.
- Streamlit n'était pas disponible dans l'environnement de construction hors réseau; `app.py` a été compilé statiquement mais le serveur Streamlit n'a pas pu être démarré ici.
- Les architectures GRU/LSTM/Transformer sont prévues comme challengers futurs, pas ajoutées automatiquement avec un petit dataset.

## Glossaire court

**Ablation** : retrait volontaire d'une feature/composant pour mesurer son utilité.

**ATR** : mesure de volatilité utilisée notamment pour dimensionner les stops.

**Backtest** : simulation historique respectant les informations/exécutions réellement disponibles à l'époque.

**Baseline** : référence simple comme CASH, Buy & Hold ou Logistic Regression.

**Bootstrap** : rééchantillonnage descriptif pour estimer la variabilité des résultats.

**CASH** : aucune position. Option parfaitement valide.

**Causalité** : calcul utilisant uniquement des informations disponibles au moment de la décision.

**Champion** : stratégie/modèle actuellement retenu après validation.

**Challenger** : nouvelle configuration comparée au Champion.

**Data Drift** : changement de distribution des données d'entrée.

**Data Freshness** : ancienneté du dernier bar reçu par rapport au bar complet attendu.

**Data Leakage** : utilisation directe ou indirecte d'information future.

**Data Snooping** : faux résultat potentiel après trop d'essais sur le même historique.

**Deep Learning** : réseaux neuronaux; ici expérimentaux et comparés à des baselines simples.

**Drawdown** : baisse depuis un sommet de capital.

**Early Stopping** : arrêt d'entraînement quand la validation cesse de s'améliorer.

**Embargo** : espace temporel entre train et validation pour limiter la contamination.

**Expectancy** : gain/perte moyen observé par trade.

**Feature** : variable d'entrée du modèle.

**Feature Parity** : même historique + même instant = mêmes features en backtest/replay/live.

**Final Test** : période verrouillée qui ne sert jamais à optimiser.

**Fold** : couple train/validation d'un walk-forward.

**Fresh / Warning / Delayed / Stale** : états de fraîcheur du flux; les états trop retardés bloquent le signal.

**Gap** : saut de prix pouvant dépasser le niveau de stop.

**Label** : cible future utilisée uniquement pour l'entraînement/évaluation.

**MAE/MFE** : excursion maximale défavorable/favorable d'un trade.

**Model Score** : sortie numérique d'un modèle; pas une probabilité validée par défaut.

**NO TRADE** : décision explicite de ne pas entrer.

**Out-of-Sample** : données non utilisées pour entraîner le modèle évalué.

**Overfitting** : surapprentissage du passé au détriment de la généralisation.

**Planned Risk** : perte planifiée jusqu'au stop; non garantie en présence d'un gap.

**Position Sizing** : calcul du nombre d'actions/contrats à partir du capital, stop, coûts et limites de risque.

**Profit Factor** : gains bruts / pertes brutes.

**Protocol Lock** : configuration figée avant ouverture du final test.

**Purge** : retrait d'observations dont les labels chevaucheraient la validation.

**Replay** : relecture chronologique bougie par bougie.

**Research Budget** : limite stricte du nombre d'expériences/temps/ressources.

**Risk Budget** : montant de capital planifié comme risque maximal selon le stop.

**Shadow Mode** : production prospective de signaux sans ordre réel.

**Sharpe/Sortino** : ratios rendement/risque; fragiles sur petit échantillon.

**Slippage** : écart entre prix théorique et prix d'exécution simulé.

**Target** : niveau théorique de prise de profit.

**TCN** : réseau convolutionnel temporel; la variante incluse est légère et causale.

**Threshold** : seuil transformant un score en signal.

**Walk-Forward** : entraînement sur le passé, validation sur une période future, puis avancée chronologique.

**Win Rate** : part de trades gagnants; insuffisante à elle seule pour juger une stratégie.
