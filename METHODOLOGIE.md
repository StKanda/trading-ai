# Conventions intraday et limites

## Temps et données

La source de base contient une barre de 5 minutes indexée à son début. En sortie de l'agrégation, l'index est sa fin. À 09:39 New York, une bougie 09:30–09:40 n'existe pas encore comme observation complète ; à 09:40 elle devient disponible si les deux sous-bougies sont présentes et la capture suffisamment récente.

Le 10 minutes est assemblé par paires, le 15 minutes par groupes de trois, depuis l'ouverture réelle. Les séances, jours fériés, changements d'heure et demi-séances sont définis par le calendrier. L'affichage se fait en heure de New York, les calculs utilisent UTC. Il n'y a ni forward-fill des prix manquants, ni agrégation traversant la nuit. Un backtest refuse une séance incomplète ; une analyse visuelle signale les trous et suspend le score de sélection.

Le fichier de calendrier peut être fourni avec un CSV. Il est alors une donnée déclarée par son auteur et doit être vérifié ; le code ne peut pas certifier qu'un calendrier externe correspond à une place réelle. Le mode Yahoo utilise le calendrier NYSE de la bibliothèque. Les interruptions exceptionnelles et révisions du fournisseur nécessitent encore un audit de marché.

## Indicateurs et patterns

Les 37 features classiques de V7 sont reprises : rendements, moyennes, pentes, volatilité, ATR, bougies, volume, extrêmes et RSI. **La fenêtre est en bougies** : SMA200 en 5 min ne signifie ni 200 jours ni la même durée qu'en 15 min. Les bougies hors séance sont exclues mais les indicateurs continus peuvent inclure le rendement d'un gap nocturne, connu à l'ouverture suivante.

VWAP, rendement de séance et progression horaire complètent A. Le volume relatif compare le créneau courant aux mêmes créneaux de séances antérieures, avec un décalage avant moyenne ; le volume final de la journée n'intervient pas. Les hauts/bas de la veille sont décalés d'une séance avant d'être utilisés.

La détection W/M et les niveaux de pivots sont réinitialisés chaque séance. Les paramètres intraday initiaux sont fixes : pivot gauche/droite 2 bougies ; tolérance des niveaux 0,15 % ; tolérance des deux extrêmes 0,25 % ; hauteur minimale 0,15 % ; span 6–40 bougies ; attente de cassure 12 bougies ; cooldown 6 ; mémoire 12. Ces valeurs constituent une hypothèse exploratoire, **pas des paramètres validés ou optimisés**. Les seuils NearSupport 1/2/5 % et la normalisation géométrique hérités de V7 sont conservés pour la traçabilité ; certains pourront être peu informatifs en intraday. Tout remplacement doit être évalué dans une nouvelle recherche historique.

Un pivot est connu deux bougies après son extrême. Une formation ne peut être affichée avant la confirmation du second extrême. Une cassure est datée de son observation, jamais de l'extrême historique. Les premiers événements acceptés sont conservés ; un événement plus tardif ne remplace pas un événement antérieur. Des événements au même timestamp suivent l'ordre déterministe du moteur V7, sans classement par rendement ultérieur.

Les formations en attente n'entrent pas comme cassures dans les features. Elles peuvent être invalidées ou expirer. Les scores de forme utilisent seulement la géométrie. Le volume de cassure est séparé pour permettre son ablation.

## Cible et purge

Pour un signal daté de la fin d'une bougie, l'entrée de référence est l'open de la suivante. La sortie de la cible est l'open situé 30 minutes après l'entrée. Le rendement théorique incorpore frais et slippage des deux côtés, mais pas les stops/targets ni les intérêts dépendant de la taille d'une position. C'est une cible de classification de mouvement exploitable à horizon fixe, distincte du P&L réalisé par le simulateur.

Entrée et sortie doivent rester dans la même séance et la sortie doit précéder ou coïncider avec l'ouverture de la dernière bougie. Les labels inconnus restent NaN. Les dernières observations ne sont jamais transformées en classe « baisse » par défaut.

Pour chaque entraînement, la fin du label doit être strictement antérieure à la frontière de prédiction. S'ajoute un intervalle sur la grille brute de `horizon_bars + 1 + extra_embargo_bars` observations, avant filtrage des features manquantes. En 5 min, les paramètres par défaut demandent au moins 8 observations exclues avant la frontière. Les labels de fin de séance sont souvent déjà indisponibles ; la règle supplémentaire reste appliquée et auditée.

Le masque commun exige les mêmes lignes utilisables pour toutes les ablations. Les fenêtres sont temporelles croissantes. HistGradientBoosting n'utilise pas d'early stopping interne avec split aléatoire. Aucune standardisation globale n'est ajustée sur la série entière.

Avant la recherche, toutes les données à partir de `test_start` sont physiquement retirées du snapshot. Le calendrier peut définir la fenêtre finale, mais aucun prix de cette fenêtre n'entre dans features, labels ou sélection historique.

Le verrou fixe dates, timeframe, symbole, groupes, coûts, risque, paramètres et modèle officiel ; il contient l'empreinte des prix pré-test, du calendrier, du code Python et des versions. Le test final effectue un entraînement figé sur le passé purgé, sans adaptation aux résultats finaux. Les empreintes garantissent une cohérence de fichiers, pas l'authenticité économique d'une source ni une preuve cryptographique contre un utilisateur malveillant.

## Exécution et risque

Le signal de clôture est exécuté à l'open de la bougie suivante avec slippage adverse. Cette convention théorique ne garantit pas qu'un ordre réel envoyé après publication de la bougie obtienne ce prix ; une étude de latence sera nécessaire avant paper trading connecté.

Une seule position longue à la fois. Pas de short, de pyramide ni d'allocation simultanée à plusieurs actions. La quantité est entière ; les benchmarks autorisent des fractions afin de rester à 1×. Le simulateur peut renouveler un trade seulement lors d'une nouvelle décision autorisée, pas au milieu d'une bougie après une sortie.

Stop : distance maximale entre minimum fixe et multiple d'ATR connu au signal. Objectif : multiple de cette distance. La taille réserve les pertes au stop et les coûts estimés ; elle est aussi plafonnée par le levier, le volume de la bougie précédente et le budget quotidien restant. Le coût de financement réserve une séance entière lors du sizing, puis est comptabilisé au prorata du cash débiteur et des bougies effectivement simulées. Un intérêt de bougie entière peut être imputé même si la position sort au milieu : hypothèse prudente.

Le levier configuré limite la taille **à l'entrée**. Ce n'est pas un moteur de marge avec exigences de maintenance, appels de marge ou liquidation du courtier. Le levier peut mécaniquement augmenter après une baisse des fonds propres. Ces mécanismes doivent être ajoutés selon les règles du compte avant toute connexion.

La limite quotidienne se mesure depuis le capital de début de séance, sur la valeur de liquidation incluant pertes latentes, frais de sortie estimés et slippage. Une barrière peut fermer la position avant le stop technique si le budget journalier est atteint. L'arrêt reste enclenché pour la séance même si des prix ultérieurs remontent. Il se réinitialise à la séance suivante.

Si un stop et un objectif sont tous deux touchés, l'ordre réel est inconnu : le simulateur prend le stop. Un gap sous le stop est exécuté à l'ouverture, et peut dépasser le budget. L'heure exacte d'une sortie intrabar n'est pas inventée ; le journal indique `intrabar_time_unknown` et enregistre la fin de sa bougie. Les métriques de drawdown observent la valeur aux bougies et les sorties ; elles ne reconstruisent pas chaque tick. L'exposition compte les bougies pendant lesquelles une position était présente, avec une approximation à la borne d'ouverture.

Fermeture à l'open de la dernière bougie de la séance, avec un horaire connu à l'avance. Aucune entrée si l'horizon de 30 min dépasse cet horaire. Les comparaisons Buy & Hold et intraday utilisent la même période évaluée, leurs coûts propres et un levier de référence à 1×.

## Ce qu'il reste à établir

Le scanner utilise seulement le passé disponible, mais il n'est pas encore évalué comme règle de portefeuille. Un classement séduisant en direct ne prouve aucune performance. Pour le valider : univers historique sans biais de survivance, heures fixes de sélection, exécutions des seuls actifs sélectionnés à ces instants, allocation commune et frais réalistes.

Quelques semaines Yahoo donnent beaucoup de lignes mais peu de journées indépendantes. Les labels se chevauchent, les marchés évoluent et les comparaisons de variantes multiplient les essais. Pas de preuve statistique à partir du meilleur AUC observé. Prévoir davantage d'historique, des périodes nouvelles, des intervalles par blocs/journées et une évaluation de calibration avant d'afficher une « confiance » probabiliste.

Les tests logiciels fournis n'établissent pas la qualité d'un flux Yahoo ou la sûreté d'un connecteur de courtier. L'exécution réelle, le paper trading connecté, le stop global de compte persistant, la calibration des probabilités, le scanner portefeuille et le LLM restent des étapes identifiées, non annoncées comme déjà réalisées.
