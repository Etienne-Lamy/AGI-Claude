"""ÉTAPE 1 — Objet générique détecteur/générateur (§1.3), autoencodeur CONVOLUTIF.

Le module compresse le champ visuel 10×10 en un « champ abstrait » (10×10×k,
k petit) via un encodeur convolutif, puis le régénère. Encodeur (détecteur) et
décodeur (générateur) sont entraînés CONJOINTEMENT, en ligne, à chaque pas, sur
GPU, sur un mini-lot tiré d'une mémoire de rejeu (stable).

Pourquoi CONVOLUTIF et pas un MLP : les objets font 1 pixel à des positions
arbitraires. Un MLP doit apprendre pour chaque cellule le lien position→position
— il n'y arrive pas (plafond mesuré ~44 % de rappel, quelle que soit la taille
du latent : ce n'est PAS un problème de capacité mais d'architecture). Une
convolution est locale et équivariante par translation : elle reproduit une
cellule à sa place NATURELLEMENT → reconstruction ~parfaite (mesuré 100 %).

Pourquoi CLASSIFICATION et pas MSE : le champ est discret (4 classes :
vide/corps/bâton/sucre). L'entropie croisée par cellule, pondérée (objets ≫
vide), donne des décisions NETTES et évite l'effondrement à zéro que produit une
MSE sur un champ ~90 % vide (échec documenté README v2).

Interface générique (réutilisable par le futur orchestrateur) :
  encoder(x) → champ abstrait z        generer(z) → champ reconstruit
  entrainer(x) → erreur                reconstruire(x) → champ reconstruit
  fidelite(x) → {rappel, precision}    incertitude() / fiabilite()
L'objet est autonome, chargeable/déchargeable (state_dict GPU↔CPU), rapide.
"""
import random
from collections import deque

import torch

from .config import CONFIG
from .logger import log, log_verbeux

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VALEURS = [0.0, 0.25, 0.5, 1.0]          # vide, corps, bâton, sucre
_VAL = torch.tensor(VALEURS, device=DEVICE)


def _vers_classes(x):
    """Champ de valeurs → indices de classe (plus proche valeur)."""
    return (x.unsqueeze(-1) - _VAL).abs().argmin(dim=-1)


class ModuleAutoencodeur:
    def __init__(self, id, resolution=None, canaux_latent=None, canaux_cachee=None):
        self.id = id
        t = CONFIG["taille_perception"]
        self.resolution = resolution or (t, t)         # (h, w)
        self.n_classes = len(VALEURS)
        k = canaux_latent or CONFIG["canaux_latent_vision"]
        c = canaux_cachee or CONFIG["canaux_cachee_vision"]
        self.canaux_latent = k
        self.enc = torch.nn.Sequential(
            torch.nn.Conv2d(1, c, 3, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(c, k, 3, padding=1)).to(DEVICE)          # → champ abstrait (k,h,w)
        self.dec = torch.nn.Sequential(
            torch.nn.Conv2d(k, c, 3, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(c, self.n_classes, 3, padding=1)).to(DEVICE)   # → logits (4,h,w)
        self.opt = torch.optim.Adam(
            list(self.enc.parameters()) + list(self.dec.parameters()),
            lr=CONFIG["lr_vision_ae"])
        self.poids_classe = torch.tensor(
            [1.0] + [CONFIG["poids_objet_vision"]] * (self.n_classes - 1), device=DEVICE)
        self.erreurs = []
        self.dernier_latent = None
        self.buffer = deque(maxlen=CONFIG["taille_buffer_vision"])
        log(self.id, "creation_autoencodeur", resolution=list(self.resolution),
            canaux_latent=k, device=str(DEVICE))

    # ---------------------------------------------------------------- forward
    def _img(self, x, n=1):
        h, w = self.resolution
        return torch.as_tensor(x, dtype=torch.float32, device=DEVICE).reshape(n, 1, h, w)

    def encoder(self, x):
        """Champ abstrait (k,h,w) détaché — le « champ compressé » réutilisable."""
        z = self.enc(self._img(x)).squeeze(0)
        self.dernier_latent = z.detach()
        return z

    def generer(self, z):
        """Champ abstrait → champ de valeurs reconstruit (argmax de classe)."""
        k, h, w = self.canaux_latent, *self.resolution
        z = torch.as_tensor(z, dtype=torch.float32, device=DEVICE).reshape(1, k, h, w)
        with torch.no_grad():
            cl = self.dec(z).argmax(1).reshape(-1)
        return _VAL[cl]

    def reconstruire(self, x):
        """Champ de valeurs reconstruit, sur CPU (affichage/comparaison)."""
        return self.generer(self.encoder(x)).cpu()

    # ------------------------------------------------------------ entraînement
    def entrainer(self, x):
        """Un pas conjoint enc+dec sur un mini-lot de rejeu. Entropie croisée
        par cellule, pondérée. Retourne l'erreur cellule (1 − exactitude) sur
        le champ courant."""
        h, w = self.resolution
        self.buffer.append(self._img(x).squeeze(0))          # (1,h,w)
        n = min(len(self.buffer), CONFIG["taille_lot_vision"])
        lot = torch.stack(random.sample(list(self.buffer), n))   # (n,1,h,w)
        classes = _vers_classes(lot.reshape(n, h * w)).reshape(n, h, w)
        logits = self.dec(self.enc(lot))                          # (n,4,h,w)
        perte = torch.nn.functional.cross_entropy(
            logits.permute(0, 2, 3, 1).reshape(-1, self.n_classes),
            classes.reshape(-1), weight=self.poids_classe)
        self.opt.zero_grad()
        perte.backward()
        self.opt.step()
        with torch.no_grad():
            cible = self._img(x)
            pred = self.dec(self.enc(cible)).argmax(1).reshape(-1)
            err = 1.0 - float((pred == _vers_classes(cible.reshape(-1))).float().mean())
        self.erreurs.append(err)
        if len(self.erreurs) > CONFIG["taille_max_historique_erreurs"]:
            del self.erreurs[: CONFIG["taille_max_historique_erreurs"] // 2]
        self.dernier_latent = self.enc(cible).squeeze(0).detach()
        log_verbeux(self.id, "entrainement_autoencodeur", erreur_cellule=err,
                    perte=float(perte.detach()))
        return err

    # ------------------------------------------------ prédiction de transition
    def entrainer_transition(self, x_prec, x_present):
        """Entraîne enc+dec à prédire le champ PRÉSENT à partir du champ
        PRÉCÉDENT : enc(x_prec) → dec → classes de x_present. Même réseau que la
        reconstruction, cible différente. À vitesse fixe, x_present est x_prec
        DÉCALÉ de la vitesse — le module apprend ce décalage, donc sa fiabilité
        est un INDICATEUR DE VITESSE (haute à la vitesse d'entraînement, basse
        ailleurs). Buffer de PAIRES dédié. Retourne l'erreur cellule."""
        if not hasattr(self, "_buffer_trans"):
            self._buffer_trans = deque(maxlen=CONFIG["taille_buffer_vision"])
        h, w = self.resolution
        self._buffer_trans.append((self._img(x_prec).squeeze(0),
                                   self._img(x_present).squeeze(0)))
        n = min(len(self._buffer_trans), CONFIG["taille_lot_vision"])
        paires = random.sample(list(self._buffer_trans), n)
        prec = torch.stack([a for a, _ in paires])                # (n,1,h,w)
        pres = torch.stack([b for _, b in paires])
        classes = _vers_classes(pres.reshape(n, h * w)).reshape(n, h, w)
        logits = self.dec(self.enc(prec))
        perte = torch.nn.functional.cross_entropy(
            logits.permute(0, 2, 3, 1).reshape(-1, self.n_classes),
            classes.reshape(-1), weight=self.poids_classe)
        self.opt.zero_grad()
        perte.backward()
        self.opt.step()
        with torch.no_grad():
            pred = self.dec(self.enc(self._img(x_prec))).argmax(1).reshape(-1)
            err = 1.0 - float((pred == _vers_classes(self._img(x_present).reshape(-1))).float().mean())
        self.erreurs.append(err)
        return err

    def predire(self, x_prec):
        """Champ présent prédit à partir du champ précédent (sur CPU)."""
        return self.generer(self.encoder(x_prec)).cpu()

    def fidelite_transition(self, x_prec, x_present):
        """Rappel/précision de la PRÉDICTION x_prec→x_present (indicateur de
        vitesse : élevé à la vitesse d'entraînement)."""
        with torch.no_grad():
            cl_pred = self.dec(self.enc(self._img(x_prec))).argmax(1).reshape(-1)
        cl_cible = _vers_classes(self._img(x_present).reshape(-1))
        obj_c, obj_p = cl_cible > 0, cl_pred > 0
        n_obj, n_pred = int(obj_c.sum()), int(obj_p.sum())
        rappel = int(((cl_pred == cl_cible) & obj_c).sum()) / n_obj if n_obj else 1.0
        precision = int(((cl_pred == cl_cible) & obj_p).sum()) / n_pred if n_pred else 0.0
        exactitude = float((cl_pred == cl_cible).float().mean())
        return {"rappel": round(rappel, 3), "precision": round(precision, 3),
                "exactitude": round(exactitude, 3)}

    # -------------------------------------------------------------- métriques
    def fidelite(self, x):
        """Sur les cellules-objets de la cible, fraction dont la CLASSE est
        correctement reconstruite = RAPPEL ; sur les cellules prédites objets,
        fraction correcte = PRÉCISION."""
        cible = self._img(x)
        cl_cible = _vers_classes(cible.reshape(-1))
        with torch.no_grad():
            cl_pred = self.dec(self.enc(cible)).argmax(1).reshape(-1)
        obj_c, obj_p = cl_cible > 0, cl_pred > 0
        n_obj, n_pred = int(obj_c.sum()), int(obj_p.sum())
        rappel = int(((cl_pred == cl_cible) & obj_c).sum()) / n_obj if n_obj else 1.0
        precision = int(((cl_pred == cl_cible) & obj_p).sum()) / n_pred if n_pred else 0.0
        return {"rappel": round(rappel, 3), "precision": round(precision, 3),
                "n_objets_cible": n_obj, "n_objets_reconstruits": n_pred}

    def incertitude(self, fenetre=None):
        fenetre = fenetre or CONFIG["fenetre_incertitude"]
        h = self.erreurs[-fenetre:]
        return sum(h) / len(h) if h else float(CONFIG["incertitude_initiale"])

    def fiabilite(self):
        """Score ∈ [0,1] (indicateur pour le futur orchestrateur) = exactitude
        cellule récente."""
        return 1.0 - self.incertitude()

    # ------------------------------------------------ charge/décharge mémoire
    def etat(self):
        """state_dict CPU (sauvegarde/déchargement)."""
        return {"enc": {k: v.cpu() for k, v in self.enc.state_dict().items()},
                "dec": {k: v.cpu() for k, v in self.dec.state_dict().items()}}

    def charger_etat(self, etat):
        self.enc.load_state_dict({k: v.to(DEVICE) for k, v in etat["enc"].items()})
        self.dec.load_state_dict({k: v.to(DEVICE) for k, v in etat["dec"].items()})
