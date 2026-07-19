"""ÉTAPE 5 — classification ÉMERGENTE des éléments (avant toute reconstruction).

Le système découvre, SANS étiquette, qu'il existe des sortes distinctes d'éléments
(VQ). On observe : combien de catégories émergent (les inutiles sont élaguées =
parcimonie), et — pour NOTRE interprétation seulement — si elles correspondent à
des types purs (vide/corps/bâton/sucre). Chaque catégorie émergente = un module
identifie+régénère (« un sucre »).

    python3 -m scl.etape5_classification --pas 4000
"""
import argparse

import numpy as np

from .classification_emergente import DEVICE, ClassifieurEmergent
from .config import CONFIG
from .logger import set_temps
from .monde import Monde

_NOM = {0.0: "vide", 0.25: "corps", 0.5: "bâton", 1.0: "sucre"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pas", type=int, default=4000)
    p.add_argument("--vitesse", type=int, nargs=2, default=[1, 1])
    args = p.parse_args()
    v = tuple(args.vitesse)

    clf = ClassifieurEmergent("classif_vision")
    m = Monde(graine=1); m.vitesse = np.array(v, dtype=np.int64)
    print(f"Device : {DEVICE} — découverte des catégories (K_max={clf.K}, aucune étiquette)…")
    for s in range(args.pas):
        set_temps(step=s)
        clf.entrainer(m.percevoir()["vision"][-1])
        m.appliquer_action((0, 0))

    ev = []
    for _ in range(40):
        for _ in range(5): m.appliquer_action((0, 0))
        ev.append(np.asarray(m.percevoir()["vision"][-1]).copy())

    cats = clf.categories_utilisees(ev)
    pur = clf.purete(ev)
    # reconstruction à partir des catégories
    rap = []
    for f in ev:
        rec = clf.regenerer(f); v_ = f.reshape(-1); p_ = rec.reshape(-1)
        obj = v_ > 0.1; no = int(obj.sum())
        if no: rap.append(int((np.abs(p_ - v_) < 0.2)[obj].sum()) / no)

    print(f"\n{len(cats)} catégories ÉMERGENTES (sur {clf.K} possibles ; le reste élagué).")
    print(f"Reconstruction à partir des catégories : rappel={np.mean(rap):.0%}")
    print("\nCorrespondance A POSTERIORI (pour notre interprétation, pas utilisée par le système) :")
    for k in sorted(pur):
        val, p_ = pur[k]
        print(f"   catégorie émergente #{k} → « {_NOM.get(val, val)} » (pureté {p_:.0%})")
    print("\nChaque catégorie est un module identifie (cellule→catégorie) + régénère "
          "(catégorie→apparence). C'est l'outil sur lequel objets & reconstruction se construisent.")


if __name__ == "__main__":
    main()
