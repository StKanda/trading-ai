# Architecture

L'application reste locale et s'ouvre dans le navigateur. Pas de framework frontend, de CDN, de clé API, de service distant obligatoire ou de moteur d'ordres caché. Les bibliothèques de données et de ML sont les mêmes familles que dans la V7.

| Fichier / module | Responsabilité |
|---|---|
| `main.py` | Appelle uniquement le point d'entrée |
| `trading_lab/cli.py` | Arguments `serve`, `download`, `demo-data`, `research`, `final` |
| `trading_lab/config.py` | Dataclasses `AppConfig`, `ResearchConfig`, `RiskConfig`, validation |
| `trading_lab/data/providers.py` | Contrat `DataProvider` ; sources `DemoProvider`, `CSVProvider`, `YahooProvider` ; snapshots et empreintes |
| `trading_lab/data/market.py` | `Snapshot`, UTC/New York, calendrier, qualité, bougies closes, agrégation |
| `trading_lab/features/legacy_v7.py` | Copie **inchangée** du moteur de features V7 ; traçabilité |
| `trading_lab/features/engine.py` | `FeatureEngine`, adaptation intraday, VWAP, volume par créneau, groupes, cible, formations en attente |
| `trading_lab/scanner.py` | `StockScanner`, score descriptif de sélection à l'instant observé |
| `trading_lab/risk.py` | `DailyRiskGuard`, budget quotidien, taille et scénario indicatif |
| `trading_lab/research/models.py` | Construction RF/GB, masque de purge, entraînement exclusivement passé |
| `trading_lab/research/backtest.py` | Simulation chronologique, coûts, gaps, stops, horizon et fermeture avant fin de séance |
| `trading_lab/research/metrics.py` | Classification, métriques de portefeuille, deltas d'ablation |
| `trading_lab/research/walkforward.py` | Recherche, protocole, empreintes, test final figé, exports CSV/PNG |
| `trading_lab/assistant.py` | `AnalysisAssistant`, explications factuelles à règles, interface extensible |
| `trading_lab/service.py` | `LabService`, orchestration des cas d'usage et cache en mémoire |
| `trading_lab/web/server.py` | API HTTP, serveur lié exclusivement à `127.0.0.1` |
| `trading_lab/web/static/` | HTML, CSS et JavaScript Canvas ; aucune logique financière dans les dessins |
| `tests/test_v8.py` | Contrôles causaux, d'exécution et HTTP |

## Sens des dépendances

Le navigateur demande une analyse à `LabService`. Celui-ci charge un `Snapshot`, produit des bougies terminées avec `market`, appelle `FeatureEngine`, puis `StockScanner`, `trade_plan` et `AnalysisAssistant`. Le navigateur affiche leurs valeurs : il ne fabrique pas de score financier.

La recherche appelle directement les modules de données, features, modèles, simulation et métriques. Elle n'a pas besoin du serveur web. Les interfaces web et recherche partagent donc leurs conventions de données.

Les modules de risque ne connaissent ni le navigateur ni Yahoo. Leur état est neuf à chaque backtest. Ils ne constituent pas encore un coupe-circuit de compte de trading persistant.

## Ajouter une source

Implémenter `fetch(symbol) -> Snapshot` selon le contrat `DataProvider`. Fournir des OHLCV de 5 min, avec horodatage de **début**, UTC explicite, calendrier et instant de capture. Pour une future API de courtier, utiliser ses statuts de marché et ses données d'exécution ; ne pas remplacer simplement `YahooProvider` en supposant que les garanties de latence sont identiques.

## Ajouter un modèle

Ajouter sa construction dans `research/models.py`, déclarer le protocole et les variantes **avant** le test, puis vérifier la purge. Aucun prétraitement ne doit être ajusté sur toute la série. Pour une calibration future : réserver un segment temporel indépendant, purgé, à l'intérieur du passé d'entraînement ; ne jamais calibrer sur le test final.

## Ajouter un LLM conversationnel

Remplacer ou envelopper `AnalysisAssistant.answer(question, context)`. Le contexte contient les chiffres déjà calculés, leur date et la provenance. Ne pas autoriser le modèle conversationnel à passer des ordres, modifier le budget ou lever un arrêt quotidien. Les scores et niveaux doivent garder leur définition statistique ; une formulation fluide ne transforme pas un score heuristique en probabilité mesurée.

## Ajouter un courtier ultérieurement

Le courtier n'est pas choisi et aucune implémentation d'ordre réel n'est incluse. La prochaine couche devra gérer : compte paper, carnet d'ordres local persistant, identifiants idempotents, synchronisation avec les positions du compte, exécutions partielles, ordres de protection, déconnexions, données périmées et autorisations du compte. Le budget quotidien devra être **commun à tous les actifs** et survivre au redémarrage. L'interface seule ne doit jamais décider si un ordre est autorisé.

## Origine préservée

`legacy_v7.py` a exactement l'empreinte SHA-256 du fichier V7 validé :

`d7d0ce3b9095254240d3eb94856e77cc2e5c3337ce3a6bcad52a5b87f1bca5fd`

Les nouvelles adaptations sont dans d'autres fichiers. Les CSV V5 n'ont pas été réintroduits ; leur déduplication potentiellement rétrospective ne sert pas à la V9. Aucun fichier V6/V7 d'origine n'a été modifié.
