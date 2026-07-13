"""Phase 10 — tests de decision_action.py : fusion pondérée continue (perception
vs prédiction), et le test mécanisé DIRECT de la contrainte que l'ancien
orchestrateur.py violait — un seul besoin dominant, jamais un mélange."""
import torch

from scl.decision_action import (
    fusion_ponderee, recompense_intrinseque, reflexe_cable,
    priorisation_besoin_dominant, generer_actions_candidates,
)
from scl.memoires import TableBesoins


# ------------------------------------------------------------------- fusion_ponderee

def test_fusion_ponderee_confiance_zero_donne_perception():
    perception, prediction = torch.tensor([1.0, 2.0]), torch.tensor([10.0, 20.0])
    assert torch.allclose(fusion_ponderee(perception, prediction, confiance=0.0), perception)


def test_fusion_ponderee_confiance_un_donne_prediction():
    perception, prediction = torch.tensor([1.0, 2.0]), torch.tensor([10.0, 20.0])
    assert torch.allclose(fusion_ponderee(perception, prediction, confiance=1.0), prediction)


def test_fusion_ponderee_continue_entre_les_deux():
    perception, prediction = torch.tensor([0.0, 0.0]), torch.tensor([10.0, 10.0])
    resultat = fusion_ponderee(perception, prediction, confiance=0.3)
    assert torch.allclose(resultat, torch.tensor([3.0, 3.0]))


def test_recompense_intrinseque():
    assert abs(recompense_intrinseque(1.0, 0.6) - 0.4) < 1e-9


# --------------------------------------------------------------------- reflexe_cable

def test_reflexe_cable_se_declenche_au_dela_du_seuil():
    assert reflexe_cable(0.9, seuil=0.4) == "freiner"


def test_reflexe_cable_rien_en_dessous_du_seuil():
    assert reflexe_cable(0.1, seuil=0.4) is None


# --------------- contrainte structurelle §0/§15.3 : le cœur de cette phase ---------------

def test_priorisation_besoin_dominant_choisit_un_seul_besoin_jamais_un_melange():
    """Test mécanisé direct de la contrainte que l'ancien orchestrateur.py
    violait (souhaitabilite_torch mélangeait tous les besoins en continu) :
    avec deux besoins quasi égaux, la décision doit être EXACTEMENT l'action
    d'un seul des deux, jamais une combinaison pondérée des deux."""
    tb = TableBesoins()
    tb.etats["faim"], tb.etats["ennui"] = 0.5, 0.48   # quasi égaux
    actions = {"faim": {"manger": 1.0, "explorer": 0.1},
              "ennui": {"manger": 0.1, "explorer": 1.0}}
    action = priorisation_besoin_dominant(tb, actions)
    assert action in ("manger", "explorer")   # jamais une structure composite/mélangée
    assert action == "manger"                  # argmax(0.5, 0.48) = faim


def test_priorisation_besoin_dominant_bascule_avec_hysteresis():
    tb = TableBesoins()
    tb.etats["faim"], tb.etats["ennui"] = 0.5, 0.1
    actions = {"faim": {"manger": 1.0}, "ennui": {"explorer": 1.0}}
    assert priorisation_besoin_dominant(tb, actions) == "manger"
    tb.etats["ennui"] = 0.5 + tb.delta + 0.01   # dépasse nettement la marge δ
    assert priorisation_besoin_dominant(tb, actions) == "explorer"


def test_reflexe_cable_toujours_prioritaire_sur_le_besoin_dominant():
    tb = TableBesoins()
    tb.etats["faim"] = 0.9   # besoin dominant très fort
    actions = {"faim": {"manger": 1.0}, "ennui": {"explorer": 1.0}}
    action = priorisation_besoin_dominant(tb, actions, reflexe="freiner")
    assert action == "freiner"


def test_priorisation_sans_candidate_pour_le_besoin_dominant():
    tb = TableBesoins()
    tb.etats["faim"], tb.etats["ennui"] = 0.9, 0.0
    assert priorisation_besoin_dominant(tb, {"ennui": {"explorer": 1.0}}) is None


# --------------------------------------------------------- generer_actions_candidates

def test_generer_actions_candidates_structure():
    tb = TableBesoins()
    scores = generer_actions_candidates(["a", "b"], tb)
    assert set(scores.keys()) == set(tb.etats.keys())
    assert all(set(v.keys()) == {"a", "b"} for v in scores.values())
