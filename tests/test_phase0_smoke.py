"""Phase 0 — script de fumée : monde + logger + checkpoint sur 100 pas, en
isolation (aucune dépendance à Graphe/TableBesoins, qui arrivent en Phase 1+
et Phase 4 — seule la mécanique de persistance/journalisation est vérifiée
ici, via des objets minimaux)."""
import json
import random
import types

from scl import checkpoint, logger
from scl.monde import Monde, ACCELERATIONS_PERMISES


def test_monde_logger_checkpoint_100_pas(tmp_path):
    log_path = tmp_path / "smoke.jsonl"
    audit = logger.configurer(chemin=str(log_path), console=False, verbeux=False)

    monde = Monde(graine=123)
    rng = random.Random(123)
    for step in range(100):
        audit.set_temps(jour=0, step=step)
        monde.percevoir()
        accel = rng.choice(ACCELERATIONS_PERMISES)
        monde.appliquer_action(accel)

    audit.fermer()

    # le fichier JSONL est bien formé, ligne par ligne, champs standard présents
    lignes = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lignes) > 0
    for ligne in lignes:
        rec = json.loads(ligne)
        assert {"n", "jour", "step", "acteur", "action"} <= rec.keys()

    assert monde.compteurs["steps"] == 100

    # checkpoint : round-trip sur des objets minimaux (Graphe/TableBesoins
    # réels arrivent en Phase 1/4 ; seule la mécanique de persistance compte ici)
    graphe_stub = types.SimpleNamespace(modules={"m0": "dummy"})
    besoins_stub = types.SimpleNamespace(faim=0.3, ennui=0.1)

    chemin_ckpt = tmp_path / "etat.pkl"
    assert not checkpoint.existe(str(chemin_ckpt))
    checkpoint.sauvegarder(str(chemin_ckpt), graphe=graphe_stub, monde=monde, besoins=besoins_stub)
    assert checkpoint.existe(str(chemin_ckpt))

    composants = checkpoint.charger(str(chemin_ckpt))
    assert list(composants["graphe"].modules.keys()) == ["m0"]
    assert composants["monde"].compteurs["steps"] == monde.compteurs["steps"]
    assert composants["besoins"].faim == 0.3
    assert composants["besoins"].ennui == 0.1
