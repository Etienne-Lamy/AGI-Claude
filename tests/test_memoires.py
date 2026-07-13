"""Phase 1 — tests des mémoires de base (§8) : TableBesoins/besoin_dominant,
TableContexte, tampon/exceptions, registres (câblage, rupture, disponibilité,
provenance)."""
import pickle

import torch

from scl.memoires import (
    TableBesoins, TableContexte, MemoireTampon, MemoireExceptions,
    RegistreCablage, RegistreRupture, RegistreDisponibilite, RegistreProvenance,
)


# ------------------------------------------------------------ TableBesoins

def test_besoin_dominant_pas_d_oscillation_sur_quasi_egalite():
    tb = TableBesoins()
    tb.etats["faim"], tb.etats["ennui"] = 0.50, 0.49
    premier = tb.besoin_dominant()
    assert premier == "faim"
    # ennui grimpe très légèrement mais reste sous la marge δ : pas de bascule
    tb.etats["ennui"] = 0.50 + tb.delta / 2
    assert tb.besoin_dominant() == "faim"


def test_besoin_dominant_bascule_au_dela_de_la_marge():
    tb = TableBesoins()
    tb.etats["faim"], tb.etats["ennui"] = 0.50, 0.10
    assert tb.besoin_dominant() == "faim"
    tb.etats["ennui"] = 0.50 + tb.delta + 0.01
    assert tb.besoin_dominant() == "ennui"


def test_besoin_dominant_jamais_un_melange():
    # contrat structurel : besoin_dominant renvoie toujours une seule clé,
    # jamais une combinaison — c'est exactement ce que l'ancien orchestrateur
    # violait (mélange pondéré continu via souhaitabilite_torch).
    tb = TableBesoins()
    tb.etats["faim"], tb.etats["ennui"] = 0.5, 0.5
    d = tb.besoin_dominant()
    assert d in ("faim", "ennui")
    assert isinstance(d, str)


def test_ennui_croit_et_plafonne_a_0_5():
    tb = TableBesoins()
    for _ in range(2000):
        tb.mettre_a_jour()
    assert tb.etats["ennui"] == 0.5


def test_surprise_validee_reinitialise_ennui():
    tb = TableBesoins()
    for _ in range(500):
        tb.mettre_a_jour()
    assert tb.etats["ennui"] > 0.0
    tb.noter_surprise_validee()
    tb.mettre_a_jour()
    assert tb.etats["ennui"] < 0.01   # repart quasiment de zéro


def test_faim_augmente_puis_sucre_la_reduit():
    tb = TableBesoins()
    f0 = tb.etats["faim"]
    tb.mettre_a_jour(evenements=[])
    assert tb.etats["faim"] > f0
    tb.mettre_a_jour(evenements=["sucre"])
    assert tb.etats["faim"] < f0 + 0.01


def test_besoins_picklable_pour_checkpoint():
    tb = TableBesoins()
    tb.mettre_a_jour(evenements=["sucre"])
    tb.besoin_dominant()
    tb2 = pickle.loads(pickle.dumps(tb))
    assert tb2.etats == tb.etats


# ------------------------------------------------------------ TableContexte

def test_contexte_choc_sur_collision_baton():
    tc = TableContexte()
    assert tc.etat == "normal"
    for _ in range(3):
        tc.mettre_a_jour(evenements=["baton"])
    assert tc.etat == "choc"


def test_contexte_retour_normal_apres_decroissance():
    tc = TableContexte()
    tc.mettre_a_jour(evenements=["baton", "baton", "baton"])
    assert tc.etat == "choc"
    for _ in range(200):
        tc.mettre_a_jour(evenements=[])
    assert tc.etat == "normal"


def test_contexte_independant_de_table_besoins():
    # douleur ne doit plus exister dans TableBesoins (v6 : b_t = faim, ennui)
    tb = TableBesoins()
    assert "douleur" not in tb.etats


# --------------------------------------------------------------- MemoireTampon

def test_memoire_tampon_ajout_et_filtrage():
    mt = MemoireTampon()
    mt.ajouter_reco("m1", torch.zeros(3), torch.ones(3), 0.1, t=0)
    mt.ajouter_reco("m2", torch.zeros(3), torch.ones(3), 0.2, t=1)
    mt.ajouter_gen("m1", torch.zeros(2), torch.ones(2), 0.3, t=2)
    r, g = mt.pour_point("m1")
    assert len(r) == 1 and len(g) == 1
    assert len(mt.pour_point("m2")[0]) == 1
    mt.clear()
    assert mt.tentatives_reco == [] and mt.tentatives_gen == []


def test_memoire_exceptions_non_resolues():
    me = MemoireExceptions()
    me.ajouter(contexte="x", erreur=0.5, t=0)
    assert len(me.non_resolues()) == 1
    me.entrees[0]["resolved"] = True
    assert me.non_resolues() == []


# --------------------------------------------------------- registres câblage/rupture

def test_registre_cablage_append():
    rc = RegistreCablage()
    e = rc.append("m1", point_injection="capteur", contexte="x",
                  signature_anomalie="anomalie", t=0, type_="rupture")
    assert rc.entrees == [e]
    assert e["module_id"] == "m1"


def test_registre_rupture_cooldown():
    rr = RegistreRupture()
    assert rr.peut_creer("p1", t=0)
    rr.marquer_abandon("p1", t=0)
    assert not rr.peut_creer("p1", t=1)          # dans la fenêtre de cooldown
    assert rr.peut_creer("p1", t=100_000)        # bien après : autorisé


# --------------------------------------------------------- RegistreDisponibilite

def test_disponibilite_dedoublonne_par_diversite():
    rd = RegistreDisponibilite()
    base = torch.zeros(4)
    for _ in range(10):
        rd.ajouter("m1", base)   # dix fois le même contexte : reste un seul
    assert len(rd.echantillon("m1")) == 1
    rd.ajouter("m1", torch.ones(4) * 10)   # nettement différent : accepté
    assert len(rd.echantillon("m1")) == 2


def test_disponibilite_bornee_en_taille():
    rd = RegistreDisponibilite()
    for i in range(50):
        rd.ajouter("m1", torch.ones(4) * i * 5)   # tous suffisamment distincts
    assert len(rd.echantillon("m1")) <= 20   # taille_echantillon_disponibilite


# ------------------------------------------------------------ RegistreProvenance

def test_provenance_marquage_et_purge():
    rp = RegistreProvenance()
    ex1 = {"module_id": "m1", "valeur": 1}
    ex2 = {"module_id": "m1", "valeur": 2}
    rp.marquer(ex1, "reel")
    rp.marquer(ex2, "imagine")
    assert ex1["provenance"] == "reel"
    assert ex2["provenance"] == "imagine"
    n = rp.purger("m1")
    assert n == 2
    assert rp.purger("m1") == 0   # déjà purgé


def test_provenance_ratio_lot_plafonne():
    rp = RegistreProvenance()
    lot = [{"provenance": "reel"} for _ in range(2)] + \
          [{"provenance": "imagine"} for _ in range(20)]
    plafonne = rp.ratio_lot(lot)
    n_reels = sum(1 for e in plafonne if e["provenance"] == "reel")
    n_imagines = sum(1 for e in plafonne if e["provenance"] == "imagine")
    assert n_reels == 2
    assert n_imagines <= 2 * 3   # plafond_ratio_imagine_reel = 3.0


def test_provenance_ratio_lot_sans_reel_inchange():
    rp = RegistreProvenance()
    lot = [{"provenance": "imagine"} for _ in range(10)]
    assert rp.ratio_lot(lot) == lot
