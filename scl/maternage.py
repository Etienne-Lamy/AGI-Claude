"""Maternage — placer l'agent PRÈS d'un sucre pour l'éduquer (étape 22, §7 conception).

« Comme une maman chat ou oiseau » : au lieu d'attendre que l'agent tombe sur un sucre par
hasard (récompense trop rare pour apprendre, cf. étapes 19-20), on le POSE à 2-3 cases d'un
sucre existant. Il le trouve alors en 2-3 actions → récompense DENSE → la valeur Q apprend
« sucre visible dans cette direction → accélérer vers lui », puis généralise à le trouver de
plus en plus loin (le crédit remonte, cf. rejeu nocturne).

On ne FABRIQUE pas de sucre (ce serait tricher sur le monde) : on choisit seulement la
POSITION DE DÉPART de l'agent, là où le monde procédural a déjà déposé un sucre. C'est un
échafaudage de curriculum, pas un signal inventé.
"""
import numpy as np


def action_vers_sucre(monde):
    """DÉMONSTRATION de la « maman » (échafaudage de curriculum, §15.3 — dette assumée,
    RETIRÉE à l'évaluation) : l'action qui rapproche du sucre VISIBLE le plus proche.
    Lire l'offset d'un sucre présent dans son propre champ 10×10 est de la PERCEPTION, pas
    une coordonnée vérité-terrain cachée. Sert à donner à la valeur Q des exemples DENSES et
    propres « sucre visible dans cette direction → cette action → récompense » ; la politique
    apprise, elle, ne verra que le champ brut et devra généraliser. None si aucun sucre."""
    import numpy as np
    sucres, _ = monde.objets_visibles()
    if not sucres:
        return None
    dx, dy = min(sucres, key=lambda o: max(abs(o[0]), abs(o[1])))
    if abs(dx) >= abs(dy) and dx != 0:
        return (int(np.sign(dx)), 0)
    if dy != 0:
        return (0, int(np.sign(dy)))
    return (0, 0)


def placer_pres_sucre(monde, rng, dmin=2, dmax=3, portee=400, essais=400):
    """Repositionne l'agent (vitesse nulle) de sorte qu'un sucre soit visible à une
    distance ∈ [dmin, dmax] cases. Retourne l'offset (Δx, Δy) du sucre le plus proche
    ainsi placé, ou None si aucun trouvé après `essais` tentatives."""
    for _ in range(essais):
        monde.agent_pos = np.array([rng.integers(-portee, portee),
                                    rng.integers(-portee, portee)], dtype=np.int64)
        monde.vitesse = np.zeros(2, dtype=np.int64)
        monde.derniere_accel = np.zeros(2, dtype=np.int64)
        monde.historique_vision = [monde._frame() for _ in range(monde.n_frames)]
        sucres, _ = monde.objets_visibles()
        proches = [d for d in sucres if dmin <= max(abs(d[0]), abs(d[1])) <= dmax]
        if proches:
            return min(proches, key=lambda d: max(abs(d[0]), abs(d[1])))
    return None
