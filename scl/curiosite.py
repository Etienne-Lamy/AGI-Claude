"""Motivation intrinsèque — curiosité par progrès d'apprentissage (§15.2).

Moteur de l'agent : réduire son incertitude prédictive. Chaque module porte une
incertitude = erreur de prédiction récente (`Module.friction_recente`). Quand un
module cesse de progresser (incertitude basse et stable), il est « maîtrisé » et
n'offre plus de récompense intrinsèque ; l'agent se tourne alors vers ce qui est
encore incertain — d'abord la reconstruction du champ statique, puis, une fois
celle-ci maîtrisée, les CONSÉQUENCES de ses actions (qui ne deviennent
observables qu'en agissant). C'est ce basculement qui fait émerger la découverte
motrice, sans politique d'exploration câblée.

Récompense intrinsèque = progrès d'apprentissage = baisse d'incertitude entre
deux fenêtres (réemploi direct de `decision_action.recompense_intrinseque`,
L_total(t−1)−L_total(t), §15.2). Positive quand un module apprend activement,
~0 quand il est maîtrisé OU pas encore attaqué.
"""
from .config import CONFIG
from .logger import log_verbeux


def incertitude(module, fenetre=None):
    """Incertitude courante d'un module = erreur de prédiction moyenne récente
    (proxy de la variance résiduelle, §4.2). Élevée = mal maîtrisé."""
    fenetre = fenetre if fenetre is not None else CONFIG["fenetre_incertitude"]
    hist = [e for _, e, _ in module.error_history[-fenetre:]]
    if not hist:
        return float(CONFIG["incertitude_initiale"])   # jamais évalué ⇒ max
    return sum(hist) / len(hist)


def progres_apprentissage(module, fenetre=None):
    """Progrès d'apprentissage = incertitude(fenêtre ancienne) −
    incertitude(fenêtre récente) (§15.2). > 0 : le module apprend encore
    (récompense intrinsèque) ; ~0 : maîtrisé ou stagnant (plus d'intérêt)."""
    fenetre = fenetre if fenetre is not None else CONFIG["fenetre_incertitude"]
    hist = [e for _, e, _ in module.error_history]
    if len(hist) < 2 * fenetre:
        return 0.0   # pas assez de recul pour mesurer un progrès
    ancienne = sum(hist[-2 * fenetre:-fenetre]) / fenetre
    recente = sum(hist[-fenetre:]) / fenetre
    return ancienne - recente


def maitrise(module, fenetre=None):
    """True si le module est maîtrisé : incertitude basse ET plus de progrès
    notable (plateau bas, §1.4). C'est la condition qui « sature » la
    réduction d'incertitude sur ce module et pousse l'agent ailleurs."""
    inc = incertitude(module, fenetre)
    prog = abs(progres_apprentissage(module, fenetre))
    hist = [e for _, e, _ in module.error_history]
    assez_de_vecu = len(hist) >= CONFIG["min_vecu_maitrise"]
    return (assez_de_vecu and inc < CONFIG["seuil_incertitude_maitrise"]
            and prog < CONFIG["seuil_progres_maitrise"])


def frontiere(modules, fenetre=None):
    """Renvoie l'id du module à la FRONTIÈRE d'apprentissage : celui qui offre
    le plus de progrès attendu (incertitude haute mais réductible). Sert à
    orienter l'attention/curiosité vers ce qui vaut la peine d'être appris.
    None si tout est maîtrisé (l'agent doit alors générer de la nouveauté en
    agissant)."""
    candidats = {mid: incertitude(m, fenetre) for mid, m in modules.items()
                 if not maitrise(m, fenetre)}
    if not candidats:
        return None
    mid = max(candidats, key=candidats.get)
    log_verbeux("curiosite", "frontiere", module=mid, incertitude=candidats[mid])
    return mid
