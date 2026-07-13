"""Phase 8 — tests de credit.py : regret exact sur cas synthétiques, rejeu
contrefactuel qui ne mute AUCUN poids, amorçage à la création."""
import torch

from scl.credit import (
    regret_composition, approx_regret_jour, rejeu_contrefactuel_nocturne,
    amorcage_creation,
)
from scl.module import Module


def _module(id_):
    return Module(id_, n_inputs_reco=4, n_latent=3, n_outputs_gen=4)


# --------------------------------------------------------------- regret_composition

def test_regret_composition_exact():
    assert regret_composition(0.5, [0.5, 0.3, 0.9]) == 0.5 - 0.3


def test_regret_composition_zero_si_le_choix_etait_optimal():
    assert regret_composition(0.1, [0.5, 0.1, 0.9]) == 0.0


def test_regret_composition_sans_candidats():
    assert regret_composition(0.5, []) == 0.0


def test_approx_regret_jour_exact():
    assert approx_regret_jour(0.2, [0.5, 0.8, 0.1]) == 0.8 - 0.2


def test_approx_regret_jour_sans_candidats():
    assert approx_regret_jour(0.2, []) == 0.0


# ------------------------------------------------------- rejeu_contrefactuel_nocturne

def test_rejeu_contrefactuel_nocturne_ne_mute_aucun_poids():
    torch.manual_seed(0)
    candidats = {"m1": _module("m1"), "m2": _module("m2")}
    echantillon = [{"input": torch.randn(4), "cible": torch.randn(3)} for _ in range(5)]
    avant = {cid: [p.detach().clone() for p in m.parametres()]
            for cid, m in candidats.items()}

    residus = rejeu_contrefactuel_nocturne(candidats, echantillon)

    assert set(residus.keys()) == {"m1", "m2"}
    assert all(r >= 0.0 for r in residus.values())
    for cid, m in candidats.items():
        for a, p in zip(avant[cid], m.parametres()):
            assert torch.equal(a, p.detach())


def test_rejeu_contrefactuel_nocturne_voie_gen():
    candidats = {"m1": _module("m1")}
    echantillon = [{"input": torch.randn(3), "cible": torch.randn(4)} for _ in range(3)]
    residus = rejeu_contrefactuel_nocturne(candidats, echantillon, voie="gen")
    assert "m1" in residus


# ----------------------------------------------------------------------- amorcage_creation

def test_amorcage_creation_injecte_exemple_positif():
    m = _module("m")
    jeu = []
    ex = amorcage_creation(m, torch.randn(4), jeu)
    assert jeu == [ex]
    assert ex["label"] == "positif"
    assert ex["module_id"] == "m"
