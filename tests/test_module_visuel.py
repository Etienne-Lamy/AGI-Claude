"""Phase 2 — porte d'isolement du module visuel (§25 étape 3 : "valider en
isolation avant intégration à la boucle temps réel"). Aucune dépendance au
monde réel : frames synthétiques uniquement. Ce test DOIT passer avant que
module_visuel.py ne soit jamais câblé dans inne.py/boucle.py (Phase 11)."""
import torch

from scl.module_visuel import ModuleVisuel
from scl.utils import pente


def test_formes_forward():
    torch.manual_seed(0)
    m = ModuleVisuel("vision_t")
    champ = torch.rand(3, 10, 10)
    latent = m.forward_reconnaissance(champ)
    assert latent.shape == (m.n_latent,)
    sortie = m.forward_generation(latent)
    assert sortie.shape == (m.n_outputs_gen + m.dim_reinjection,)
    assert m.n_outputs_gen == 3 * 10 * 10


def test_aucun_gradient_ne_traverse_la_frontiere():
    torch.manual_seed(0)
    m = ModuleVisuel("vision_t2")
    champ_amont = torch.rand(3, 10, 10, requires_grad=True)
    m.entrainer_masque(champ_amont, fraction_masque=0.5)
    assert champ_amont.grad is None


def test_latent_predictif_multi_init_forme():
    torch.manual_seed(0)
    m = ModuleVisuel("vision_t3")
    z = m.chercher_latent_predictif(torch.rand(3, 10, 10), torch.rand(3, 10, 10),
                                    n_iterations=2, n_inits=2)
    assert z.shape == (m.n_latent,)


def test_croissance_non_supportee_explicite():
    m = ModuleVisuel("vision_t4")
    assert m.grandir("reco") is False
    assert m.grandir("gen") is False


# --------------------------- porte d'isolement : convergence sur frames synthétiques

def test_module_visuel_reconstruction_masquee_converge():
    """Entraînement isolé (pas de monde réel) : reconstruction masquée sur un
    petit jeu de frames synthétiques fixes. L'erreur doit baisser nettement
    puis se stabiliser (pas de divergence) — condition d'acceptation avant
    toute intégration au graphe/boucle réelle."""
    torch.manual_seed(0)
    m = ModuleVisuel("vision_conv")
    champs = [torch.rand(3, 10, 10) for _ in range(4)]

    erreurs = []
    for i in range(300):
        champ = champs[i % len(champs)]
        e = m.entrainer_masque(champ, fraction_masque=0.5, t=i)
        erreurs.append(e)

    debut = sum(erreurs[:20]) / 20
    fin = sum(erreurs[-20:]) / 20
    assert fin < debut, f"pas de baisse : début={debut:.4f} fin={fin:.4f}"
    assert fin < 0.5 * debut, f"baisse insuffisante : début={debut:.4f} fin={fin:.4f}"
    # quasi-plateau en fin de course : la pente reste petite devant l'échelle de l'erreur
    p = pente(erreurs[-100:])
    assert abs(p) < 0.01 * max(debut, 1e-6), f"pas de plateau, pente={p:.6f}"
