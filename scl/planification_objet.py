"""Planification A* sur l'état‑OBJET (étape 25, §3 du plan).

L'état `(E, v)` est minuscule → dérouler l'arbre des ACCÉLÉRATIONS est bon marché. Nœud =
`(objets, vitesse)` ; arête = accélération `a` (coût = 1 pas, +pénalité si on percute un
bâton) ; BUT = un sucre amené sur le corps (centre) — mangé. `g` = coût cumulé, `h` = valeur
apprise du reste à faire (None ⇒ 0 ⇒ Dijkstra borné, §7.3). On récupère la SÉQUENCE D'ACTIONS
qui mange le sucre — l'agent en pousse la première, à chaque pas (temps réel).

Le déroulé unité (décalage d'une case à la fois) reproduit la traversée du monde
(`monde.appliquer_action` teste chaque cellule franchie) → un sucre à 2 cases est mangé même
en un seul pas de vitesse 2.
"""
import heapq
import itertools

import numpy as np

from .monde import ACCELERATIONS_PERMISES, VAL_BATON, VAL_SUCRE


def _cats(po):
    sucre = {k for k in po.cat_objet if abs(float(po.val_cat[k]) - VAL_SUCRE) < 0.1}
    baton = {k for k in po.cat_objet if abs(float(po.val_cat[k]) - VAL_BATON) < 0.1}
    return sucre, baton


def _avancer(po, objets, v2, cat_sucre, cat_baton):
    """Applique la vitesse v2 en décalages UNITÉ ; détecte sucre/bâton franchissant le centre."""
    cen, t = po.centre, po.t
    E = list(objets)
    mange = baton = False
    pas = [(int(np.sign(v2[0])), 0)] * abs(int(v2[0])) \
        + [(0, int(np.sign(v2[1])))] * abs(int(v2[1]))
    for u in pas:
        E2 = []
        for k, (i, j) in E:
            ni, nj = i - u[0], j - u[1]
            if (ni, nj) == (cen, cen):
                if k in cat_sucre:
                    mange = True
                elif k in cat_baton:
                    baton = True
                # objet consommé → retiré
            elif 0 <= ni < t and 0 <= nj < t:
                E2.append((k, (ni, nj)))
        E = E2
    return mange, baton, E


def chercher_sucre(po, dyn, objets, v=(0, 0), budget=3000, profondeur_max=12, v_psi=None):
    """A* : renvoie la séquence d'accélérations qui mange un sucre, ou None. `v_psi(E,v)→coût`
    estimé du reste (heuristique apprise ; None ⇒ recherche non informée bornée)."""
    cat_sucre, cat_baton = _cats(po)
    if not cat_sucre:
        return None
    h = v_psi or (lambda E, vv: 0.0)
    cnt = itertools.count()
    frontiere = [(h(objets, v), next(cnt), list(objets), tuple(v), [])]
    visites = {}
    while frontiere:
        f, _, E, vv, actions = heapq.heappop(frontiere)
        cle = (tuple(sorted(E)), vv)
        if cle in visites and visites[cle] <= len(actions):
            continue
        visites[cle] = len(actions)
        if len(actions) >= profondeur_max or len(visites) > budget:
            continue
        for a in ACCELERATIONS_PERMISES:
            v2 = dyn.accel(vv, a)
            mange, baton, E2 = _avancer(po, E, v2, cat_sucre, cat_baton)
            if mange:
                return actions + [a]                    # plan trouvé
            g2 = len(actions) + 1 + (5 if baton else 0)
            cle2 = (tuple(sorted(E2)), v2)
            if cle2 in visites and visites[cle2] <= g2:
                continue
            heapq.heappush(frontiere, (g2 + h(E2, v2), next(cnt), E2, v2, actions + [a]))
    return None
