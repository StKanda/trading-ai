# Trading AI V10.2

Patch d'interface et de suivi marché construit sur V10.1.

## Nouveautés

- cockpit de configuration regroupé dans la barre latérale ;
- interface visuelle plus compacte et lisible ;
- sélection symbole + watchlist ;
- source Local CSV ou Yahoo ;
- **mode direct** avec actualisation automatique 15/30/60/120 secondes ;
- intervalles graphiques **5 / 10 / 15 / 30 minutes** ;
- périodes visibles 1 / 2 / 5 / 10 / 20 séances ;
- bouton de rafraîchissement manuel ;
- chandeliers + volume + VWAP + supports/résistances + W/M + trades + stop/target ;
- Opening Range 20 minutes calculée exactement depuis la source 5 minutes, y compris sur une vue agrégée ;
- état de fraîcheur du feed affiché séparément de l'état du worker.

## Limite importante du mode direct

Le mode direct effectue du **polling Yahoo/yfinance**. Il ne s'agit ni d'un flux tick-by-tick ni d'une garantie de données boursières temps réel. Le dashboard continue de n'utiliser que des bougies terminées.

Les Champions de recherche restent associés au timeframe sur lequel ils ont été validés (actuellement 5 minutes). Les vues 10/15/30 minutes sont des vues d'analyse et leur score est explicitement marqué comme descriptif.

Aucune connexion courtier et aucun ordre réel.
