# Références consultées le 27 septembre 2026

Les spécifications peuvent évoluer ; consulter les sources avant de changer un connecteur. Ces références documentent les contraintes, sans valider la performance d'une stratégie.

- yfinance, `download` : intervalles proposés, limite de l'historique intraday, bornes temporelles, ajustement des prix. Le 10 min n'est pas un intervalle natif : https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html
- pandas_market_calendars, calendrier NYSE, horaires et fermetures anticipées : https://pandas-market-calendars.readthedocs.io/en/latest/usage.html
- NYSE, calendrier et horaires officiels : https://www.nyse.com/trade/hours-calendars
- Interactive Brokers, données via API : les titres requièrent généralement un abonnement de marché adapté. Ce lien illustre une contrainte de futur connecteur ; il ne présume pas du choix du courtier : https://www.interactivebrokers.com/docs/general/market-data-subscriptions/introduction
- SEC / Investor.gov, levier et financement sur marge : effets sur pertes et intérêts, indépendamment de la validité du signal : https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-bulletins/leveraged-investing-strategies-know-risks-using-these-advanced-investment-tools
- SEC / Investor.gov, intérêts d'un compte sur marge : https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-bulletins/investor-bulletin-interested-margin-understand-interest

Les règles applicables au levier et au day trading dépendent du courtier, du produit, du compte et du pays. Aucun montant réglementaire ni levier autorisé n'est codé comme une vérité universelle. Le paramètre 1–3× du laboratoire représente uniquement une hypothèse de simulation.
