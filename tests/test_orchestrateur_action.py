"""Orchestrateur d'action à attention (étape 26) : apprend par imitation à émettre une
action depuis l'ensemble de jetons‑objets."""
import torch

from scl.module_ae import DEVICE
from scl.orchestrateur_action import OrchestrateurAction


def test_imitation_apprend_une_action():
    torch.manual_seed(0)
    orch = OrchestrateurAction(k_categories=4)
    tok = torch.zeros((1, orch.f_in), device=DEVICE)
    tok[0, 3] = 1.0; tok[0, 4] = 0.4                 # un objet (catégorie 3, position)
    for _ in range(200):
        orch.imiter([(tok, [0, 0], 1)])               # cible : action 1
    assert orch.choisir(tok, [0, 0]) == 1


def test_gere_ensemble_vide_et_taille_variable():
    orch = OrchestrateurAction(k_categories=4)
    vide = torch.zeros((0, orch.f_in), device=DEVICE)
    assert 0 <= orch.choisir(vide, [1, 0]) < 5        # aucun objet → décide quand même
    deux = torch.zeros((2, orch.f_in), device=DEVICE)
    assert orch.logits(deux, [0, 0]).shape[0] == 5    # invariant à la taille de l'ensemble
