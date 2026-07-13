"""Phase 10 — tests d'allocation_attention.py : proportionnalité WFQ, budget
jamais dépassé, partage jour/nuit de la création."""
from scl.allocation_attention import allouer_capacite, role_creation, urgence_fil
from scl.memoires import TableBesoins


def test_allouer_capacite_proportionnelle():
    allocation = allouer_capacite({"a": 1.0, "b": 3.0}, W=16)
    assert allocation["b"] >= 3 * allocation["a"]


def test_allouer_capacite_budget_jamais_depasse():
    urgences = {f"f{i}": float(i + 1) for i in range(10)}
    allocation = allouer_capacite(urgences, W=16)
    assert sum(allocation.values()) <= 16


def test_allouer_capacite_urgence_totale_nulle_repartition_egale():
    allocation = allouer_capacite({"a": 0.0, "b": 0.0}, W=10)
    assert allocation["a"] == allocation["b"]


def test_allouer_capacite_vide():
    assert allouer_capacite({}) == {}


def test_urgence_fil_jamais_nulle():
    tb = TableBesoins()
    u = urgence_fil("test", tb, residu_surprise=0.0, poids_besoin=0.0, poids_surprise=0.0)
    assert u > 0.0


def test_role_creation_jour_minimum_viable():
    tb = TableBesoins()
    assert role_creation(tb, residu_surprise=10.0, phase="jour", W=16) <= 1


def test_role_creation_nuit_proportionnelle_a_la_surprise():
    tb = TableBesoins()
    part_faible = role_creation(tb, residu_surprise=0.01, phase="nuit", W=16)
    part_forte = role_creation(tb, residu_surprise=100.0, phase="nuit", W=16)
    assert part_forte >= part_faible
