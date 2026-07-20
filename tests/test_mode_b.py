"""Mode B : émetteur de programmes typé, appris par imitation. Doit reproduire des
programmes cibles DIFFÉRENTS selon l'objectif, et n'émettre que du bien typé."""
import torch

from scl.mode_b import (ModeB, entrainer_par_imitation, entrainer_par_renforcement,
                        _masque_type, VOCAB)
from scl.orchestrateur import OPERATEURS


def test_emission_toujours_bien_typee():
    m = ModeB(n_objectifs=2)
    for obj in (0, 1):
        chaine = m.emettre(obj)
        t = "champ"
        for op in chaine:
            assert OPERATEURS[op][0] == t          # type d'entrée compatible
            t = OPERATEURS[op][1]
        assert t == "champ"                         # programme terminal valide


def test_imitation_apprend_des_programmes_differents_par_objectif():
    torch.manual_seed(0)
    m = ModeB(n_objectifs=2)
    exemples = [(0, ["predire_champ"]),
                (1, ["compresser", "generer"])]
    entrainer_par_imitation(m, exemples, pas=300)
    assert m.emettre(0) == ["predire_champ"]
    assert m.emettre(1) == ["compresser", "generer"]


def test_renforcement_decouvre_le_meilleur_programme_sans_professeur():
    """Sans imitation : par la SEULE récompense R(objectif, chaine), Mode B doit
    converger vers le meilleur programme PAR objectif (Principe 2 : appris par
    renforcement selon le contexte)."""
    torch.manual_seed(0)
    cibles = {0: ["predire_champ"], 1: ["compresser", "generer"]}
    def recompense(obj, chaine):
        return 1.0 if chaine == cibles[obj] else 0.1
    m = ModeB(n_objectifs=2)
    entrainer_par_renforcement(m, recompense, objectifs=(0, 1), pas=300)
    assert m.emettre(0) == cibles[0]
    assert m.emettre(1) == cibles[1]


def test_echantillonnage_respecte_le_typage_par_pas():
    # l'échantillonneur explore (peut finir non terminal à max_len, la récompense le
    # pénalise) mais ne fait JAMAIS de transition invalide, et log_prob est dérivable.
    m = ModeB(n_objectifs=2)
    torch.manual_seed(1)
    for _ in range(20):
        chaine, logp, ent = m.echantillonner(0)
        t = "champ"
        for op in chaine:
            assert OPERATEURS[op][0] == t          # transition toujours bien typée
            t = OPERATEURS[op][1]
        assert logp.requires_grad and ent.requires_grad


def test_masque_type_interdit_les_transitions_invalides():
    # après 'compresser' on est en 'latent' : 'predire_champ' (champ→champ) interdit
    masque = _masque_type(["compresser"])
    assert not masque[VOCAB.index("predire_champ")]
    assert masque[VOCAB.index("generer")]          # latent→champ autorisé
