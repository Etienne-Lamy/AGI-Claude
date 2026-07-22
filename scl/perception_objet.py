"""Perception OBJET (étape 23, §1 du plan d'upgrade) — raisonner en objets, pas en pixels.

Un champ 10×10 ~90 % vide devient un petit ensemble `{(catégorie, position)}`. Deux gains
décisifs face à la prédiction pixel-brut (qui plafonnait à 57-84 %) :

  - COMPRESSION : ~quelques objets × (catégorie, x, y) au lieu de 100 pixels, et sémantique ;
  - PRÉDICTION T+1 EXACTE par DÉCALAGE : le monde défile à l'opposé du mouvement de l'agent,
    donc chaque position `p` devient `p − v` (v = vitesse propre, proprioception). Régénérer le
    champ depuis les positions décalées. Le seul « défaut » est l'objet qui ENTRE par le bord
    (jamais vu → non prévisible) : plafond honnête, pas une faiblesse du modèle.

On réutilise le VQ ÉMERGENT de l'étape 5 (`ClassifieurEmergent` : encodeur 1×1 → codebook →
catégories pures, sans étiquette). La catégorie « vide » et la catégorie « corps » (toujours
au centre = l'agent) sont identifiées par la VALEUR que décode chaque code — aucune étiquette.
"""
import numpy as np
import torch

from .classification_emergente import ClassifieurEmergent
from .config import CONFIG
from .monde import VAL_CORPS


class ChampObjet:
    def __init__(self, id="perc_objet"):
        self.clf = ClassifieurEmergent(id)
        self.t = self.clf.t
        self.centre = self.t // 2
        self.val_cat = None          # valeur décodée de chaque catégorie
        self.cat_objet = None        # catégories d'OBJETS (sucre/bâton) = valeur > seuil, hors corps
        self.cat_corps = None

    def entrainer(self, champ):
        return self.clf.entrainer(champ)

    def calibrer(self):
        """Décode chaque code du codebook → sa valeur représentative ; en déduit quelles
        catégories sont des objets (valeur > seuil) vs le corps (valeur ≈ VAL_CORPS) vs vide."""
        with torch.no_grad():
            codes = self.clf.codebook.weight.reshape(self.clf.K, self.clf.C, 1, 1)
            vals = self.clf.dec(codes).reshape(self.clf.K).cpu().numpy()
        self.val_cat = vals
        seuil = CONFIG["seuil_objet_vision"]
        self.cat_corps = int(np.argmin(np.abs(vals - VAL_CORPS)))
        self.cat_objet = {k for k in range(self.clf.K)
                          if vals[k] > seuil and k != self.cat_corps}
        return self.cat_objet

    # ------------------------------------------------------------- objets ↔ champ
    def objets(self, champ):
        """Champ → liste d'objets [(catégorie, (i, j))] (hors vide et hors corps central)."""
        if self.cat_objet is None:
            self.calibrer()
        cat = self.clf.categoriser(champ)
        objs = []
        for i in range(self.t):
            for j in range(self.t):
                if (i, j) == (self.centre, self.centre):
                    continue
                k = int(cat[i, j])
                if k in self.cat_objet:
                    objs.append((k, (i, j)))
        return objs

    def regenerer(self, objets):
        """Liste d'objets → champ (corps toujours au centre, propriété du capteur)."""
        f = np.zeros((self.t, self.t), dtype=np.float32)
        f[self.centre, self.centre] = VAL_CORPS
        for k, (i, j) in objets:
            f[i, j] = float(self.val_cat[k])
        return f

    def decaler(self, objets, vitesse):
        """Décale les positions de −vitesse (le monde défile à l'opposé du mouvement)."""
        vx, vy = int(vitesse[0]), int(vitesse[1])
        out = []
        for k, (i, j) in objets:
            ni, nj = i - vx, j - vy
            if 0 <= ni < self.t and 0 <= nj < self.t and (ni, nj) != (self.centre, self.centre):
                out.append((k, (ni, nj)))
        return out

    def predire(self, champ, vitesse):
        """Prédiction T+1 : catégoriser → décaler les objets par la vitesse → régénérer."""
        return self.regenerer(self.decaler(self.objets(champ), vitesse))

    def taille_etat(self, champ):
        """|E| : nombre d'objets suivis (mesure de compression)."""
        return len(self.objets(champ))
