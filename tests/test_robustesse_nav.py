"""Tests des correctifs de robustesse de navigation : le réflexe de douleur
rend la main à v=0 (pas de deadlock), et le décodeur de pointeurs ne crashe
jamais même si ses paramètres divergent (inf/nan)."""
import torch

from scl.inne import reflexe_frein
from scl.attention import PointerNetwork
from scl.config import CONFIG


def test_reflexe_rend_la_main_a_vitesse_nulle():
    """À v=0 le garde-fou n'a plus d'élan à freiner : il doit rendre la main
    (None) au lieu de renvoyer (0,0) et paralyser l'agent (deadlock corrigé)."""
    douleur_forte = CONFIG["seuil_reflexe_douleur"] + 0.5
    assert reflexe_frein([0, 0], douleur_forte) is None


def test_reflexe_freine_l_elan_dominant():
    """Avec de l'élan et de la douleur, le réflexe freine bien la composante
    dominante."""
    douleur_forte = CONFIG["seuil_reflexe_douleur"] + 0.5
    assert reflexe_frein([2, 0], douleur_forte) == (-1, 0)
    assert reflexe_frein([0, -2], douleur_forte) == (0, 1)


def test_reflexe_inactif_sous_seuil():
    assert reflexe_frein([2, 0], 0.0) is None


def test_decodeur_ne_crashe_pas_sur_params_diverges():
    """Si les pointeurs divergent (inf/nan), le décodeur doit échantillonner
    sans lever (repli uniforme) — sinon crash multinomial sur run long."""
    d = PointerNetwork()
    with torch.no_grad():
        d.u_src.mul_(float("inf"))      # force des logits non finis
        d.W_cib.fill_(float("nan"))
    n = 4
    representation = torch.randn(n, d.d_model)
    elements = [{"id": f"m{i}", "type": "spatial-x"} for i in range(n)]
    triplet, log_prob = d.decoder(representation, elements)   # ne doit pas lever
    assert triplet["src"] in {e["id"] for e in elements}
    assert triplet["cib"] in {e["id"] for e in elements}
    assert torch.isfinite(log_prob)
