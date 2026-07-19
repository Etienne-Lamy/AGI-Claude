"""Outil « attention/masquage » de l'orchestrateur — décompose le champ visuel en
OBJETS par Slot Attention (Locatello et al. 2020), au lieu d'une compression brute
globale (opaque, prédiction difficile). Chaque slot se lie à un objet ; son masque
alpha SÉLECTIONNE (masque) sa région → représentation OBJET-CENTRÉE, structurée.

C'est l'incarnation du principe « plusieurs modules + attention » (Architecture §27) :
au lieu d'un latent monolithique, une liste d'objets. De cette liste on extrait le
code compact (x, y, type) par slot — position = centroïde du masque — ce qui rend la
prédiction triviale (à vitesse v : position → position + v).

Compétition = softmax SUR LES SLOTS (chaque cellule se répartit entre slots), itératif
(GRU). GPU.
"""
import random
from collections import deque

import torch

from .config import CONFIG
from .logger import log, log_verbeux

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VALEURS = [0.0, 0.25, 0.5, 1.0]
_VAL = torch.tensor(VALEURS, device=DEVICE)


def _classes(x):
    return (x.unsqueeze(-1) - _VAL).abs().argmin(dim=-1)


def _grille(t):
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, t), torch.linspace(-1, 1, t), indexing="ij")
    return torch.stack([xs, ys], 0).to(DEVICE)          # (2,t,t) coords normalisées


class _SlotAttention(torch.nn.Module):
    def __init__(self, n, D, iters):
        super().__init__()
        self.n, self.D, self.iters = n, D, iters
        self.mu = torch.nn.Parameter(torch.randn(1, 1, D))
        self.logsigma = torch.nn.Parameter(torch.zeros(1, 1, D))
        self.q = torch.nn.Linear(D, D); self.k = torch.nn.Linear(D, D); self.v = torch.nn.Linear(D, D)
        self.gru = torch.nn.GRUCell(D, D)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(D, 2 * D), torch.nn.ReLU(), torch.nn.Linear(2 * D, D))
        self.ln_in = torch.nn.LayerNorm(D); self.ln_s = torch.nn.LayerNorm(D); self.ln_m = torch.nn.LayerNorm(D)

    def forward(self, inp):                              # (B,HW,D)
        B = inp.shape[0]
        inp = self.ln_in(inp); k = self.k(inp); v = self.v(inp)
        slots = self.mu + torch.exp(self.logsigma) * torch.randn(B, self.n, self.D, device=DEVICE)
        for _ in range(self.iters):
            prev = slots
            q = self.q(self.ln_s(slots))
            logits = torch.einsum("bid,bjd->bij", k, q) * (self.D ** -0.5)     # (B,HW,N)
            attn = torch.softmax(logits, dim=-1) + 1e-8                        # softmax sur les slots
            attn = attn / attn.sum(dim=1, keepdim=True)
            updates = torch.einsum("bij,bid->bjd", attn, v)
            slots = self.gru(updates.reshape(-1, self.D), prev.reshape(-1, self.D)).reshape(B, self.n, self.D)
            slots = slots + self.mlp(self.ln_m(slots))
        return slots


class ModuleAttentionSlots:
    """Autoencodeur objet-centré. Interface générique proche de ModuleAutoencodeur :
    encoder(champ)→slots ; reconstruire(champ)→champ ; entrainer(champ)→erreur ;
    fidelite(champ) ; liste_objets(champ)→[(x,y,type,presence)] (code compact)."""

    def __init__(self, id, n_slots=8, D=64, iters=3):
        self.id = id
        self.t = CONFIG["taille_perception"]
        self.n_slots, self.D = n_slots, D
        self.enc = torch.nn.Sequential(
            torch.nn.Conv2d(1, D, 3, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(D, D, 3, padding=1), torch.nn.ReLU()).to(DEVICE)
        self.pos_enc = torch.nn.Conv2d(2, D, 1).to(DEVICE)
        self.sa = _SlotAttention(n_slots, D, iters).to(DEVICE)
        self.pos_dec = torch.nn.Linear(2, D).to(DEVICE)
        self.dec = torch.nn.Sequential(
            torch.nn.Linear(D, D), torch.nn.ReLU(),
            torch.nn.Linear(D, D), torch.nn.ReLU(),
            torch.nn.Linear(D, len(VALEURS) + 1)).to(DEVICE)   # 4 classes + alpha
        self.opt = torch.optim.Adam(self._params(), lr=CONFIG["lr_vision_ae"])
        self.poids = torch.tensor([1.0] + [CONFIG["poids_objet_vision"]] * (len(VALEURS) - 1), device=DEVICE)
        self.grille = _grille(self.t)
        self.buffer = deque(maxlen=CONFIG["taille_buffer_vision"])
        self.erreurs = []
        log(self.id, "creation_attention_slots", n_slots=n_slots, D=D, device=str(DEVICE))

    def _params(self):
        return (list(self.enc.parameters()) + list(self.pos_enc.parameters())
                + list(self.sa.parameters()) + list(self.pos_dec.parameters())
                + list(self.dec.parameters()))

    def _img(self, x, n=1):
        return torch.as_tensor(x, dtype=torch.float32, device=DEVICE).reshape(n, 1, self.t, self.t)

    def _slots_et_sortie(self, champs):                  # (B,1,t,t) → slots, logits, masks
        B = champs.shape[0]; t = self.t
        f = self.enc(champs) + self.pos_enc(self.grille.unsqueeze(0))
        tokens = f.reshape(B, self.D, t * t).permute(0, 2, 1)
        slots = self.sa(tokens)                                          # (B,N,D)
        g = self.grille.reshape(2, t * t).permute(1, 0)
        s = slots.unsqueeze(2) + self.pos_dec(g).unsqueeze(0).unsqueeze(0)
        out = self.dec(s)                                               # (B,N,HW,5)
        logits, alpha = out[..., :len(VALEURS)], out[..., len(VALEURS):]
        masks = torch.softmax(alpha, dim=1)                            # (B,N,HW,1) masque sur slots
        recon = (logits * masks).sum(1)                                 # (B,HW,4)
        return slots, recon.permute(0, 2, 1).reshape(B, len(VALEURS), t, t), masks

    def encoder(self, x):
        with torch.no_grad():
            slots, _, _ = self._slots_et_sortie(self._img(x))
        return slots.squeeze(0)                                         # (N,D)

    def reconstruire(self, x):
        with torch.no_grad():
            _, rec, _ = self._slots_et_sortie(self._img(x))
        return _VAL[rec.argmax(1).reshape(-1)].cpu()

    def entrainer(self, x):
        self.buffer.append(self._img(x).squeeze(0))
        n = min(len(self.buffer), CONFIG["taille_lot_vision"])
        lot = torch.stack(random.sample(list(self.buffer), n))
        _, rec, _ = self._slots_et_sortie(lot)
        cl = _classes(lot.reshape(n, -1)).reshape(n, self.t, self.t)
        perte = torch.nn.functional.cross_entropy(
            rec.permute(0, 2, 3, 1).reshape(-1, len(VALEURS)), cl.reshape(-1), weight=self.poids)
        self.opt.zero_grad(); perte.backward()
        torch.nn.utils.clip_grad_norm_(self._params(), 1.0)
        self.opt.step()
        with torch.no_grad():
            _, rec1, _ = self._slots_et_sortie(self._img(x))
            err = 1.0 - float((rec1.argmax(1).reshape(-1) == _classes(self._img(x).reshape(-1))).float().mean())
        self.erreurs.append(err)
        if len(self.erreurs) > CONFIG["taille_max_historique_erreurs"]:
            del self.erreurs[: CONFIG["taille_max_historique_erreurs"] // 2]
        log_verbeux(self.id, "entrainement_attention", erreur_cellule=err)
        return err

    def fidelite(self, x):
        cl = _classes(self._img(x).reshape(-1))
        with torch.no_grad():
            _, rec, _ = self._slots_et_sortie(self._img(x))
        pred = rec.argmax(1).reshape(-1)
        oc, op = cl > 0, pred > 0
        no, npr = int(oc.sum()), int(op.sum())
        rappel = int(((pred == cl) & oc).sum()) / no if no else 1.0
        precision = int(((pred == cl) & op).sum()) / npr if npr else 0.0
        return {"rappel": round(rappel, 3), "precision": round(precision, 3)}

    def liste_objets(self, x):
        """Code COMPACT structuré : liste des objets de la RECONSTRUCTION
        objet-centrée, chacun (row, col, type). C'est la liste (x,y,lettre) —
        prédire le champ suivant = DÉCALER ces positions de la vitesse (aucun
        réseau de prédiction). Le corps central (0.25) est exclu (repère fixe)."""
        t = self.t
        with torch.no_grad():
            _, rec, _ = self._slots_et_sortie(self._img(x))
            cl = rec.squeeze(0).argmax(0).cpu().numpy()          # (t,t) classes
        objets = []
        for i in range(t):
            for j in range(t):
                typ = int(cl[i, j])
                if typ > 0 and not (i == t // 2 and j == t // 2):
                    objets.append((i, j, typ))                   # (row, col, type)
        return objets

    def incertitude(self, fenetre=None):
        fenetre = fenetre or CONFIG["fenetre_incertitude"]
        h = self.erreurs[-fenetre:]
        return sum(h) / len(h) if h else float(CONFIG["incertitude_initiale"])
