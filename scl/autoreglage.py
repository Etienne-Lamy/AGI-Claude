"""AUTO-RÉGLAGE (§28.4) : la boucle qui règle les hyperparamètres de l'orchestrateur
SANS intervention humaine, de façon RÉVERSIBLE.

Principe (§28.4, invariant « aucun réglage n'est sacré ») : on ne garde un changement
QUE s'il améliore un OBSERVABLE mesuré ; sinon on revient à l'état d'avant. Asymétrie :
il faut une amélioration STRICTE au-delà d'une marge (sinon on ne bouge pas) — on ne
poursuit pas le bruit. C'est le pendant, au niveau des réglages, du verrouillage
asymétrique des modules (plancher jamais plafond).

Le cœur est GÉNÉRIQUE : il ne connaît ni le monde ni torch. On lui passe, pour un
paramètre, une fonction `appliquer(valeur)` (qui pose la valeur dans le système) et une
fonction `mesurer() → score` (plus grand = mieux : un G, un rappel, −#modules…). Il fait
une recherche locale réversible et rend la meilleure valeur trouvée.

La table SYMPTOMES relie un diagnostic (§28.3) au paramètre à pousser et au sens — c'est
la connaissance qui dit QUOI régler quand tel observable dérape ; le mécanisme ci-dessus
dit COMMENT (réversible, mesuré).
"""
from .logger import log

# Diagnostic (§28.3) → (paramètre à régler, sens du correctif). Sens : +1 augmenter,
# −1 diminuer. C'est de la connaissance déclarative, pas du câblage de valeurs.
SYMPTOMES = {
    "sur_creation_modules":   ("grace_regime", +1),      # trop de naissances → plus de grâce
    "sous_creation_modules":  ("seuil_rappel_inexplique", +1),  # rien ne naît → surprise plus sensible
    "rappel_plafonne":        ("dim_latent_vision", +1),  # goulot trop serré → l'élargir
    "categories_impures":     ("n_categories", +1),       # classes mêlées → plus de prototypes
    "episodes_fragmentes":    ("hysteresis_surprise", +1),  # coupures multiples → plus d'hystérésis
}


class AutoReglage:
    """Recherche locale réversible d'un hyperparamètre contre un observable."""

    def __init__(self, marge=1e-6):
        self.marge = marge          # amélioration minimale pour accepter un changement
        self.historique = []

    def regler(self, nom, valeur, deltas, appliquer, mesurer):
        """UN pas réversible : évalue `valeur` puis chaque voisin `valeur+d`, garde le
        meilleur (revient à `valeur` si aucun voisin ne dépasse la marge). Retourne
        (meilleure_valeur, meilleur_score)."""
        appliquer(valeur)
        meilleur_v, meilleur_s = valeur, mesurer()
        base = meilleur_s
        for d in deltas:
            v = valeur + d
            appliquer(v)
            s = mesurer()
            if s > meilleur_s + self.marge:
                meilleur_v, meilleur_s = v, s
        appliquer(meilleur_v)       # garde le meilleur (= revert si c'était la base)
        change = meilleur_v != valeur
        self.historique.append({"param": nom, "avant": valeur, "apres": meilleur_v,
                                "score": round(float(meilleur_s), 5), "garde": change})
        log("autoreglage", "pas", param=nom, avant=valeur, apres=meilleur_v,
            base=round(float(base), 5), score=round(float(meilleur_s), 5), garde=change)
        return meilleur_v, meilleur_s

    def optimiser(self, nom, valeur, deltas, appliquer, mesurer, iters=10):
        """Enchaîne des pas réversibles jusqu'à stabilité (aucun voisin n'améliore) ou
        `iters` atteint. Descente/montée locale prudente : chaque pas est lui-même
        réversible, donc on ne s'éloigne jamais vers un pire état durable."""
        for _ in range(iters):
            nouvelle, _ = self.regler(nom, valeur, deltas, appliquer, mesurer)
            if nouvelle == valeur:      # plateau : plus rien à gagner localement
                break
            valeur = nouvelle
        return valeur

    def correctif_pour(self, symptome):
        """Traduit un diagnostic §28.3 en (paramètre, sens) à régler. None si inconnu."""
        return SYMPTOMES.get(symptome)
