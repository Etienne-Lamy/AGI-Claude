"""Composition de modules : délai (T-1), module-vitesse (verrouillage asymétrique),
et naissance d'un module quand un RÉGIME NOUVEAU apparaît. Compresseur simulé
(stub) pour rester rapide et déterministe."""
import numpy as np
import torch

from scl.composition import DetecteurVitesse, ModuleDelai, ModuleVitesse
from scl.config import CONFIG
from scl.module_ae import DEVICE


def test_delai_rend_la_valeur_du_pas_precedent():
    d = ModuleDelai()
    assert d.sortie is None
    a = torch.ones(4, device=DEVICE)
    d.pousser(a)
    assert torch.allclose(d.sortie, a)
    b = torch.zeros(4, device=DEVICE)
    d.pousser(b)
    assert torch.allclose(d.sortie, b)


def test_module_vitesse_apprend_puis_se_verrouille():
    """Il apprend une transformation fixe, puis — compétent — se VERROUILLE et
    n'apprend plus (c'est ce verrou qui empêche l'oubli du régime, §1.4)."""
    torch.manual_seed(0)
    dim = 8
    mv = ModuleVitesse("t", dim)
    z = torch.randn(dim, device=DEVICE)
    cible = torch.roll(z, 1)
    for _ in range(CONFIG["maturite_module_vitesse"] + 50):
        mv.entrainer(z, cible, res_rel=0.1)          # résidu bas ⇒ compétent
    assert mv.verrouille, "un module compétent doit se verrouiller"
    assert mv.entrainer(z, cible, res_rel=0.1) is None, "verrouillé ⇒ n'apprend plus"


class _CompresseurStub:
    """Compresseur simulé : latent = champ aplati (dim fixe), générateur = identité."""
    dim_latent = 6

    def encoder(self, champ):
        return torch.as_tensor(champ, dtype=torch.float32, device=DEVICE).reshape(-1)[: self.dim_latent]

    def generer(self, z):
        return torch.as_tensor(z, dtype=torch.float32, device=DEVICE).reshape(-1)


def test_naissance_quand_un_specialiste_verrouille_ne_peut_plus_expliquer():
    """Mécanisme visé : un spécialiste VERROUILLÉ ne peut pas s'adapter à un régime
    nouveau → le résidu reste haut → surprise → NAISSANCE d'un module. (Le harnais
    `etape6_composition` le montre en conditions réelles : 3 régimes → 3 couverts.)"""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    det = DetecteurVitesse(_CompresseurStub())

    def champ(i, k=1.0):
        b = np.zeros(6, dtype=np.float32)
        b[i % 6] = k
        return b + 0.01 * rng.standard_normal(6).astype(np.float32)

    for i in range(50):                              # amorce : un module existe
        det.etape(champ(i))
    assert len(det.vitesses) >= 1
    # on le rend SPÉCIALISTE (verrouillé) : il ne peut plus absorber un autre régime
    for mv in det.vitesses.values():
        mv.verrouille = True
    det._grace_restante = 0
    det._ema_res = None
    n_avant = len(det.vitesses)

    for i in range(600):                             # régime franchement différent
        det.etape(champ(i * 5, k=8.0))
    assert len(det.vitesses) > n_avant, \
        "régime inexpliqué par un spécialiste verrouillé ⇒ un module doit naître"
