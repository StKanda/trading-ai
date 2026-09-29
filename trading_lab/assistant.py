"""Assistant explicatif déterministe, testable et sans accès aux ordres.

Une future interface LLM pourra implémenter answer(question, context). Cette
version répond uniquement à partir des chiffres calculés ; elle n'est pas un LLM.
"""


class AnalysisAssistant:
    def answer(self, question, context):
        q = question.casefold()
        item, plan = context["scanner"], context.get("plan")
        symbol = item["symbol"]
        if any(word in q for word in ("confiance", "probab", "score", "gagner", "fiable")):
            score = item["selection_score"]
            return (f"{symbol} a un score de sélection de {score}/100." if score is not None else
                    f"Le score de {symbol} est suspendu car les données sont incomplètes ou en retard.") + (
                " Ce score résume liquidité, activité relative et amplitude face aux coûts. "
                "Il ne mesure pas les chances de gagner. Le score RF optionnel est lui aussi non calibré. "
                "Aucune probabilité de réussite intraday n'est validée pour cette version.")
        if any(word in q for word in ("risque", "levier", "perte", "stop", "taille")):
            if not plan:
                return "Pas de scénario calculable : historique ou données insuffisants."
            return (f"Scénario d'achat {symbol} : {plan['shares']} actions, exposition estimée "
                    f"{plan['leverage']:.2f}× et stop {plan['stop']:.2f} USD. Budget par trade "
                    f"{plan['risk_budget']:.2f} USD ; limite quotidienne {plan['daily_budget']:.2f} USD. "
                    "Le simulateur compte frais et moins-values latentes dans cette limite. "
                    "Un gap peut dépasser le budget. Rien n'est envoyé à un courtier.")
        if any(word in q for word in ("entrée", "sortie", "objectif", "acheter", "vendre", "niveau")):
            if not plan:
                return "Les niveaux d'un scénario d'achat ne sont pas disponibles à cet instant."
            return (f"Repères indicatifs : entrée estimée {plan['entry_estimate']:.2f}, "
                    f"stop {plan['stop']:.2f}, objectif {plan['target']:.2f} USD. "
                    "Ce scénario décrit la gestion d'une éventuelle position longue ; il ne constitue pas "
                    "un signal validé. Le prix et la taille seront recalculés à l'ouverture suivante.")
        if any(word in q for word in ("pattern", "figure", "double", "w/m", "cassure")):
            patterns = context.get("patterns", [])
            if not patterns:
                return "Aucun W/M conforme aux critères n'est visible à cet instant. Les pivots exigent deux bougies de confirmation."
            last = patterns[-1]
            return (f"Dernière figure visible : {last['pattern']}, {last['status']}. "
                    f"Qualité géométrique {last['quality_score']:.0f}/100. "
                    "Cette note n'est pas une probabilité de gain. Une formation en attente de neckline ne déclenche pas de cassure.")
        if any(word in q for word in ("réel", "direct", "donnée", "retard", "heure")):
            return (f"Source : {item['source']}. Dernière bougie terminée : {item['last_bar']}. "
                    "La démonstration contient des prix fabriqués. Yahoo est interrogé périodiquement, "
                    "sans garantie de latence ni flux bid/ask. Un rafraîchissement n'en fait pas un flux temps réel garanti.")
        return (f"À la dernière bougie terminée, {symbol} cote {item['price']:.2f} USD ; "
                f"variation depuis l'ouverture {item['session_change']:+.2%}, tendance {item['trend'].lower()}. "
                "Je peux expliquer les niveaux, les W/M, le score de sélection, le levier et la limite de perte. "
                "Cet assistant local suit des règles explicites ; aucune IA conversationnelle n'est connectée.")
