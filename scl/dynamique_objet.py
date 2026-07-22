"""Dynamique sur l'état-OBJET (étape 24, §2 du plan) — l'action est une ACCÉLÉRATION.

État dynamique `(E, v)` : E = objets `{(catégorie, position)}`, v = vitesse propre. L'action
`a` est un CHANGEMENT de vitesse (accélération), pas une vitesse :

    v' = clip(v + a, ±v_max)            # `accel` : met à jour la vitesse
    E' = translater(E, v')              # UN opérateur « décaler par v' », réutilisé partout

Compositionnalité NATIVE : `translater(·, (2,0)) = translater(translater(·, (1,0)), (1,0))`.
La vitesse (2,0) n'a PAS de module dédié — elle se **simule par double usage** du décalage
de (1,0). C'est exactement « v=(2,0) se simule par double utilisation du module de (1,0) ».
Prévoir T+h sous une séquence d'actions = itérer la transition (vraies branches, pas un
`(1,0)×10` figé).
"""
import numpy as np


class DynamiqueObjet:
    def __init__(self, champ_objet, v_max=2):
        self.po = champ_objet
        self.v_max = v_max

    def accel(self, v, a):
        """v' = clip(v + a) — l'accélération met à jour la vitesse (le corps)."""
        return (int(np.clip(v[0] + a[0], -self.v_max, self.v_max)),
                int(np.clip(v[1] + a[1], -self.v_max, self.v_max)))

    def transition(self, objets, v, a):
        """(E, v, a) → (E', v') : accélérer puis translater par la nouvelle vitesse."""
        v2 = self.accel(v, a)
        return self.po.decaler(objets, v2), v2

    def derouler(self, objets, v, actions):
        """Exécute une séquence d'accélérations. Retourne la trajectoire [(E, v), …]."""
        E, vv, traj = objets, v, []
        for a in actions:
            E, vv = self.transition(E, vv, a)
            traj.append((E, vv))
        return traj

    def translater_compose(self, objets, v_cible):
        """Simule `translater(·, v_cible)` par applications RÉPÉTÉES du décalage unité
        (±1 par axe) — c.-à-d. la réutilisation compositionnelle : (2,0) = (1,0)∘(1,0)."""
        E = objets
        pas = [(int(np.sign(v_cible[0])), 0)] * abs(int(v_cible[0])) \
            + [(0, int(np.sign(v_cible[1])))] * abs(int(v_cible[1]))
        for u in pas:
            E = self.po.decaler(E, u)          # le MÊME décalage unité, réappliqué
        return E
