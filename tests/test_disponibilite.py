"""Phase 4 — tests de disponibilite.py : plateau de progrès + stabilité du
bruit (disponibilite_anticipee), accept/reject sur variation de π
(logique_acceptation, restauration des poids en cas de rejet)."""
import torch

from scl.disponibilite import disponibilite_anticipee, logique_acceptation
from scl.memoires import RegistreDisponibilite
from scl.module import Module


def _module(**kw):
    return Module("m", n_inputs_reco=4, n_latent=3, n_outputs_gen=4, **kw)


def _peupler(rd, module_id, erreurs):
    for i, e in enumerate(erreurs):
        rd.ajouter(module_id, torch.ones(4) * i * 10, erreur=e)   # contextes distincts


# ------------------------------------------------------------ disponibilite_anticipee

def test_disponible_si_plateau_stable():
    rd = RegistreDisponibilite()
    # série palindrome : pente de régression exactement nulle par construction
    _peupler(rd, "m", [0.05, 0.06, 0.04, 0.04, 0.06, 0.05])
    assert disponibilite_anticipee(_module(), rd) is True


def test_non_disponible_si_encore_en_progres():
    rd = RegistreDisponibilite()
    _peupler(rd, "m", [0.5, 0.4, 0.3, 0.2, 0.1, 0.05])   # nette tendance à la baisse
    assert disponibilite_anticipee(_module(), rd) is False


def test_non_disponible_si_bruit_instable():
    rd = RegistreDisponibilite()
    _peupler(rd, "m", [0.05, 0.5, 0.02, 0.6, 0.01, 0.55])   # moyenne ~flat, variance énorme
    assert disponibilite_anticipee(_module(), rd) is False


def test_non_disponible_si_echantillon_insuffisant():
    rd = RegistreDisponibilite()
    _peupler(rd, "m", [0.05, 0.05])
    assert disponibilite_anticipee(_module(), rd) is False


# -------------------------------------------------------------- logique_acceptation

def test_logique_acceptation_accepte_si_pi_augmente():
    m = _module()
    valeurs = iter([0.3, 0.7])
    m.fiabilite_contextuelle = lambda x: next(valeurs)
    decision = logique_acceptation(m, torch.randn(4), torch.randn(3))
    assert decision == "acceptee"


def test_logique_acceptation_rejette_et_restaure_les_poids_si_pi_diminue():
    m = _module()
    poids_avant = [p.detach().clone() for p in m.parametres_reco()]
    valeurs = iter([0.7, 0.3])
    m.fiabilite_contextuelle = lambda x: next(valeurs)
    decision = logique_acceptation(m, torch.randn(4), torch.randn(3))
    assert decision == "rejetee_contexte_signale"
    for avant, apres in zip(poids_avant, m.parametres_reco()):
        assert torch.equal(avant, apres.detach())


def test_logique_acceptation_ne_met_pas_a_jour_condensateur_si_rejetee():
    m = _module()
    valeurs = iter([0.7, 0.3])
    m.fiabilite_contextuelle = lambda x: next(valeurs)
    logique_acceptation(m, torch.randn(4), torch.randn(3))
    assert m.condensateur_reco == 0.0   # jamais touché : mise à jour rejetée
