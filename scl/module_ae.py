"""ÉTAPE 1 — Objet générique détecteur/générateur (§1.3), autoencodeur COMPRESSANT.

Principe §5 (parcimonie/MDL) : un module DOIT réduire — sa sortie (le champ
abstrait) est PLUS PETITE que son entrée. Ici : champ visuel 10×10 = 100 valeurs
→ champ abstrait de `dim_latent` = 64 flottants (< 100) → champ reconstruit.
Encodeur (détecteur) et décodeur (générateur) entraînés CONJOINTEMENT, en ligne,
sur GPU, mini-lot de rejeu.

Architecture : encodeur CONVOLUTIF (extrait les objets, locaux) puis GOULOT dense
(force la compression) ; décodeur dense puis déconvolution. Un MLP pur échoue
(~44 % de rappel) ; la convolution avant le goulot est ce qui permet de comprimer
en gardant les positions (mesuré ~85 % de rappel à latent 48-64, vs 100 valeurs).

Reconstruction par CLASSIFICATION par cellule (4 classes discrètes : vide/corps/
bâton/sucre), entropie croisée PONDÉRÉE (objets ≫ vide) → pas d'effondrement à
zéro (échec MSE documenté README v2).

Interface générique (réutilisable par l'orchestrateur) : encoder / generer /
entrainer / reconstruire / fidelite / incertitude / fiabilite ; prédiction de
transition (entrainer_transition / predire) ; etat()/charger_etat() (GPU↔CPU).
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
    return (x.unsqueeze(-1) - _VAL).abs().argmin(dim=-1)


class ModuleAutoencodeur:
    def __init__(self, id, resolution=None, dim_latent=None, canaux_cachee=None):
        self.id = id
        t = CONFIG["taille_perception"]
        self.resolution = resolution or (t, t)
        h, w = self.resolution
        self.n_classes = len(VALEURS)
        self.dim_latent = dim_latent or CONFIG["dim_latent_vision"]
        c = canaux_cachee or CONFIG["canaux_cachee_vision"]
        self._hb, self._wb = (h + 1) // 2, (w + 1) // 2      # résolution après stride 2
        self._c = c
        self.enc = torch.nn.Sequential(
            torch.nn.Conv2d(1, c, 3, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(c, c, 3, padding=1, stride=2), torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(c * self._hb * self._wb, self.dim_latent)).to(DEVICE)   # → GOULOT
        self.dec = torch.nn.Sequential(
            torch.nn.Linear(self.dim_latent, c * self._hb * self._wb), torch.nn.ReLU(),
            torch.nn.Unflatten(1, (c, self._hb, self._wb)),
            torch.nn.ConvTranspose2d(c, c, 3, stride=2, padding=1, output_padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(c, self.n_classes, 3, padding=1)).to(DEVICE)
        self.opt = torch.optim.Adam(
            list(self.enc.parameters()) + list(self.dec.parameters()), lr=CONFIG["lr_vision_ae"])
        self.poids_classe = torch.tensor(
            [1.0] + [CONFIG["poids_objet_vision"]] * (self.n_classes - 1), device=DEVICE)
        self.erreurs = []
        self.dernier_latent = None
        self.buffer = deque(maxlen=CONFIG["taille_buffer_vision"])
        log(self.id, "creation_autoencodeur", resolution=list(self.resolution),
            dim_latent=self.dim_latent, dim_entree=h * w, device=str(DEVICE))

    # ---------------------------------------------------------------- forward
    def _img(self, x, n=1):
        h, w = self.resolution
        return torch.as_tensor(x, dtype=torch.float32, device=DEVICE).reshape(n, 1, h, w)

    def _logits(self, champs):                          # (n,1,h,w) → (n,4,h,w)
        return self.dec(self.enc(champs))

    def encoder(self, x):
        """Champ abstrait COMPRESSÉ (vecteur de dim_latent) détaché."""
        z = self.enc(self._img(x)).squeeze(0)
        self.dernier_latent = z.detach()
        return z

    def generer(self, z):
        """Champ abstrait → champ de valeurs reconstruit (argmax de classe)."""
        z = torch.as_tensor(z, dtype=torch.float32, device=DEVICE).reshape(1, self.dim_latent)
        with torch.no_grad():
            cl = self.dec(z).argmax(1).reshape(-1)
        return _VAL[cl]

    def reconstruire(self, x):
        return self.generer(self.encoder(x)).cpu()

    # ------------------------------------------------------------ entraînement
    def _pas_lot(self, entrees, cibles):
        """Un pas de descente : logits(enc(entrees)) vs classes(cibles), CE
        pondérée. entrees/cibles : (n,1,h,w)."""
        h, w = self.resolution
        n = entrees.shape[0]
        classes = _vers_classes(cibles.reshape(n, h * w)).reshape(n, h, w)
        logits = self._logits(entrees)
        perte = torch.nn.functional.cross_entropy(
            logits.permute(0, 2, 3, 1).reshape(-1, self.n_classes),
            classes.reshape(-1), weight=self.poids_classe)
        self.opt.zero_grad()
        perte.backward()
        self.opt.step()
        return float(perte.detach())

    def _err_cellule(self, x_entree, x_cible):
        with torch.no_grad():
            pred = self._logits(self._img(x_entree)).argmax(1).reshape(-1)
        return 1.0 - float((pred == _vers_classes(self._img(x_cible).reshape(-1))).float().mean())

    def entrainer(self, x):
        """Reconstruction : champ → champ (mini-lot de rejeu)."""
        self.buffer.append(self._img(x).squeeze(0))
        n = min(len(self.buffer), CONFIG["taille_lot_vision"])
        lot = torch.stack(random.sample(list(self.buffer), n))
        self._pas_lot(lot, lot)
        err = self._err_cellule(x, x)
        self._noter(err)
        self.dernier_latent = self.enc(self._img(x)).squeeze(0).detach()
        return err

    def _noter(self, err):
        self.erreurs.append(err)
        if len(self.erreurs) > CONFIG["taille_max_historique_erreurs"]:
            del self.erreurs[: CONFIG["taille_max_historique_erreurs"] // 2]

    # ------------------------------------------------ prédiction de transition
    def entrainer_transition(self, x_prec, x_present):
        """Prédit le champ PRÉSENT à partir du PRÉCÉDENT (enc(prec)→dec→présent).
        À vitesse fixe, un décalage ; la fiabilité est un INDICATEUR DE VITESSE."""
        if not hasattr(self, "_buf_trans"):
            self._buf_trans = deque(maxlen=CONFIG["taille_buffer_vision"])
        self._buf_trans.append((self._img(x_prec).squeeze(0), self._img(x_present).squeeze(0)))
        n = min(len(self._buf_trans), CONFIG["taille_lot_vision"])
        paires = random.sample(list(self._buf_trans), n)
        prec = torch.stack([a for a, _ in paires])
        pres = torch.stack([b for _, b in paires])
        self._pas_lot(prec, pres)
        err = self._err_cellule(x_prec, x_present)
        self._noter(err)
        return err

    def predire(self, x_prec):
        return self.generer(self.encoder(x_prec)).cpu()

    def fidelite_transition(self, x_prec, x_present):
        return self._fidelite(self._img(x_prec), x_present)

    # -------------------------------------------------------------- métriques
    def fidelite(self, x):
        return self._fidelite(self._img(x), x)

    def _fidelite(self, entree_img, x_cible):
        cl_cible = _vers_classes(self._img(x_cible).reshape(-1))
        with torch.no_grad():
            cl_pred = self._logits(entree_img).argmax(1).reshape(-1)
        obj_c, obj_p = cl_cible > 0, cl_pred > 0
        n_obj, n_pred = int(obj_c.sum()), int(obj_p.sum())
        rappel = int(((cl_pred == cl_cible) & obj_c).sum()) / n_obj if n_obj else 1.0
        precision = int(((cl_pred == cl_cible) & obj_p).sum()) / n_pred if n_pred else 0.0
        exactitude = float((cl_pred == cl_cible).float().mean())
        return {"rappel": round(rappel, 3), "precision": round(precision, 3),
                "exactitude": round(exactitude, 3),
                "n_objets_cible": n_obj, "n_objets_reconstruits": n_pred}

    def incertitude(self, fenetre=None):
        fenetre = fenetre or CONFIG["fenetre_incertitude"]
        h = self.erreurs[-fenetre:]
        return sum(h) / len(h) if h else float(CONFIG["incertitude_initiale"])

    def fiabilite(self):
        return 1.0 - self.incertitude()

    # ------------------------------------------------ charge/décharge mémoire
    def etat(self):
        return {"enc": {k: v.cpu() for k, v in self.enc.state_dict().items()},
                "dec": {k: v.cpu() for k, v in self.dec.state_dict().items()}}

    def charger_etat(self, etat):
        self.enc.load_state_dict({k: v.to(DEVICE) for k, v in etat["enc"].items()})
        self.dec.load_state_dict({k: v.to(DEVICE) for k, v in etat["dec"].items()})


class PredicteurAbstrait:
    """Module 2 — prédit le CHAMP ABSTRAIT suivant à partir du précédent, DANS
    l'espace latent COMPRESSÉ du module 1 (vecteur dim_latent). MLP (le latent
    est dense, plus spatial). À vitesse fixe, apprend le « décalage » en latent."""

    def __init__(self, id, dim_latent=None, dim_cachee=128):
        self.id = id
        d = dim_latent or CONFIG["dim_latent_vision"]
        self.d = d
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d, dim_cachee), torch.nn.ReLU(),
            torch.nn.Linear(dim_cachee, dim_cachee), torch.nn.ReLU(),
            torch.nn.Linear(dim_cachee, d)).to(DEVICE)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=CONFIG["lr_vision_ae"])
        self.buffer = deque(maxlen=CONFIG["taille_buffer_vision"])
        self.erreurs = []
        log(self.id, "creation_predicteur_abstrait", dim_latent=d, device=str(DEVICE))

    def predire(self, z_prec):
        z = torch.as_tensor(z_prec, dtype=torch.float32, device=DEVICE).reshape(1, self.d)
        with torch.no_grad():
            return self.net(z).squeeze(0)

    def entrainer(self, z_prec, z_present):
        self.buffer.append((torch.as_tensor(z_prec, dtype=torch.float32, device=DEVICE).reshape(self.d),
                            torch.as_tensor(z_present, dtype=torch.float32, device=DEVICE).reshape(self.d)))
        n = min(len(self.buffer), CONFIG["taille_lot_vision"])
        paires = random.sample(list(self.buffer), n)
        prec = torch.stack([a for a, _ in paires])
        pres = torch.stack([b for _, b in paires])
        perte = torch.mean((self.net(prec) - pres) ** 2)
        self.opt.zero_grad()
        perte.backward()
        self.opt.step()
        e = float(perte.detach())
        self.erreurs.append(e)
        return e

    def incertitude(self, fenetre=None):
        fenetre = fenetre or CONFIG["fenetre_incertitude"]
        h = self.erreurs[-fenetre:]
        return sum(h) / len(h) if h else float(CONFIG["incertitude_initiale"])
