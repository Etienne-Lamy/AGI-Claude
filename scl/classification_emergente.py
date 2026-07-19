"""Outil « classification émergente » de l'orchestrateur (Architecture §27).

AVANT toute reconstruction/segmentation, le système découvre qu'il existe des
SORTES distinctes d'éléments — SANS étiquette, par quantification vectorielle
(VQ-VAE, van den Oord et al. 2017). Un codebook de K_max catégories s'apprend par
reconstruction seule ; les catégories réellement utilisées ÉMERGENT (les autres
sont élaguées → parcimonie). Chaque catégorie devient un élément reconnaissable :
un module qui l'IDENTIFIE (cellule → catégorie la plus proche) et la RÉGÉNÈRE
(catégorie → apparence). C'est sur ces catégories émergentes que la reconstruction
visuelle et la segmentation en objets se construisent ensuite.

Générique (extensible robot) : on classe l'**apparence locale** (encodeur 1×1, pas
de contexte spatial) ; pour une entrée riche (patch/audio) l'encodeur grossit,
le principe est identique. Aucune des catégories (vide/corps/bâton/sucre) n'est
donnée — elles sont découvertes.
"""
import random
from collections import Counter, deque

import torch

from .config import CONFIG
from .logger import log, log_verbeux

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ClassifieurEmergent:
    def __init__(self, id, k_max=None, canaux=None, beta=0.25):
        self.id = id
        self.t = CONFIG["taille_perception"]
        self.K = k_max or CONFIG["k_max_categories"]
        C = canaux or CONFIG["canaux_categorie"]
        self.C, self.beta = C, beta
        # encodeur PER-CELL (1×1) : la catégorie ne dépend que de l'apparence locale
        self.enc = torch.nn.Sequential(
            torch.nn.Conv2d(1, C, 1), torch.nn.ReLU(), torch.nn.Conv2d(C, C, 1)).to(DEVICE)
        self.codebook = torch.nn.Embedding(self.K, C).to(DEVICE)
        self.codebook.weight.data.uniform_(-1, 1)
        self.dec = torch.nn.Sequential(
            torch.nn.Conv2d(C, C, 1), torch.nn.ReLU(), torch.nn.Conv2d(C, 1, 1)).to(DEVICE)
        self.opt = torch.optim.Adam(self._params(), lr=CONFIG["lr_vision_ae"] * 2)
        self.buffer = deque(maxlen=CONFIG["taille_buffer_vision"])
        self.erreurs = []
        log(self.id, "creation_classifieur_emergent", k_max=self.K, device=str(DEVICE))

    def _params(self):
        return list(self.enc.parameters()) + list(self.codebook.parameters()) + list(self.dec.parameters())

    def _img(self, x, n=1):
        return torch.as_tensor(x, dtype=torch.float32, device=DEVICE).reshape(n, 1, self.t, self.t)

    def _quantifier(self, ze):
        B = ze.shape[0]
        flat = ze.permute(0, 2, 3, 1).reshape(-1, self.C)
        d = (flat.pow(2).sum(1, keepdim=True) - 2 * flat @ self.codebook.weight.t()
             + self.codebook.weight.pow(2).sum(1))
        idx = d.argmin(1)
        zq = self.codebook(idx).reshape(B, self.t, self.t, self.C).permute(0, 3, 1, 2)
        return zq, idx

    def entrainer(self, x):
        self.buffer.append(self._img(x).squeeze(0))
        n = min(len(self.buffer), CONFIG["taille_lot_vision"])
        lot = torch.stack(random.sample(list(self.buffer), n))
        ze = self.enc(lot)
        zq, _ = self._quantifier(ze)
        perte_vq = (zq.detach() - ze).pow(2).mean() * self.beta + (zq - ze.detach()).pow(2).mean()
        rec = self.dec(ze + (zq - ze).detach())
        poids = 1.0 + CONFIG["poids_objet_vision"] * (lot > CONFIG["seuil_objet_vision"]).float()
        perte = (poids * (rec - lot) ** 2).mean() + perte_vq
        self.opt.zero_grad(); perte.backward(); self.opt.step()
        e = float(perte.detach()); self.erreurs.append(e)
        if len(self.erreurs) > CONFIG["taille_max_historique_erreurs"]:
            del self.erreurs[: CONFIG["taille_max_historique_erreurs"] // 2]
        log_verbeux(self.id, "entrainement_classifieur", perte=e)
        return e

    # ------------------------------------------------ identifier / régénérer
    def categoriser(self, x):
        """Champ → carte de CATÉGORIES émergentes (indices, t×t)."""
        with torch.no_grad():
            _, idx = self._quantifier(self.enc(self._img(x)))
        return idx.reshape(self.t, self.t).cpu().numpy()

    def regenerer(self, x):
        """Champ régénéré à partir des catégories (valeurs, t×t)."""
        with torch.no_grad():
            ze = self.enc(self._img(x)); zq, _ = self._quantifier(ze)
            rec = self.dec(zq)
        return rec.reshape(self.t, self.t).cpu().numpy()

    def categories_utilisees(self, champs, seuil=3):
        """Ensemble des catégories réellement utilisées (émergentes) sur un jeu de
        champs — les autres codes sont élagués (parcimonie)."""
        c = Counter()
        for f in champs:
            for k in self.categoriser(f).reshape(-1):
                c[int(k)] += 1
        return {k for k, n in c.items() if n > seuil}

    def purete(self, champs):
        """[interprétation/validation, PAS utilisé par le système] : pour chaque
        catégorie émergente, fraction de cellules de la valeur dominante — mesure
        si les catégories correspondent (a posteriori) à des types purs."""
        par_cat = {}
        for f in champs:
            cat = self.categoriser(f); val = f
            for i in range(self.t):
                for j in range(self.t):
                    par_cat.setdefault(int(cat[i, j]), Counter())[round(float(val[i, j]), 2)] += 1
        out = {}
        for k, comptes in par_cat.items():
            if sum(comptes.values()) > 3:
                dom = comptes.most_common(1)[0]
                out[k] = (dom[0], round(dom[1] / sum(comptes.values()), 2))
        return out

    def incertitude(self, fenetre=None):
        fenetre = fenetre or CONFIG["fenetre_incertitude"]
        h = self.erreurs[-fenetre:]
        return sum(h) / len(h) if h else float(CONFIG["incertitude_initiale"])
