"""ÉTAPE 17 — les PULSIONS et l'objectif dominant (§4 conception).

L'agent bouge dans le monde ; un module vision RÉEL s'entraîne à reconstruire le champ et
fournit les pulsions cognitives (incertitude → curiosité, progrès → apprentissage, maîtrise
→ bullage). Les événements du corps (sucre/bâton) alimentent faim et douleur. On observe :

  - la faim monte, chute quand on mange ; un bâton déclenche le réflexe douleur ;
  - l'objectif DOMINANT bascule : d'abord cognitif (la vision n'est pas maîtrisée → curiosité/
    apprentissage), puis, une fois la vision maîtrisée, corps/bullage — sans politique câblée.

    python3 -m scl.etape17_pulsions
"""
import argparse

import numpy as np

from .config import CONFIG
from .logger import set_temps
from .module_ae import DEVICE, ModuleAutoencodeur
from .monde import ACCELERATIONS_PERMISES, Monde
from .pulsions import Pulsions


def _cognitives(erreurs, w):
    """(curiosité, apprentissage, bullage) depuis la courbe d'erreur d'un module."""
    if not erreurs:
        return 1.0, 0.0, 0.0
    inc = float(np.mean(erreurs[-w:]))
    prog = 0.0
    if len(erreurs) >= 2 * w:
        prog = float(np.mean(erreurs[-2 * w:-w]) - np.mean(erreurs[-w:]))
    maitrise = (len(erreurs) >= CONFIG["min_vecu_maitrise"]
                and inc < CONFIG["seuil_incertitude_maitrise"]
                and abs(prog) < CONFIG["seuil_progres_maitrise"])
    return inc, max(0.0, prog) * 20.0, (0.35 if maitrise else 0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pas", type=int, default=3000)
    args = p.parse_args()
    print(f"Device : {DEVICE} — {args.pas} pas ; pulsions corps + cognitives (module vision réel)")

    monde = Monde(graine=3)
    vision = ModuleAutoencodeur("vision_pulsions")
    puls = Pulsions()
    rng = np.random.default_rng(0)
    w = CONFIG["fenetre_incertitude"]

    dominants, reflexes, sucres = [], 0, 0
    for s in range(args.pas):
        set_temps(step=s)
        champ = np.asarray(monde.percevoir()["vision"][-1]).copy()
        vision.entrainer(champ)                                  # un module à la fois
        cur, app, bul = _cognitives(vision.erreurs, w)
        a = ACCELERATIONS_PERMISES[int(rng.integers(len(ACCELERATIONS_PERMISES)))]
        ev = monde.appliquer_action(a)
        sucres += sum(1 for e in ev if e == "sucre")
        vitn = float(np.linalg.norm(monde.vitesse))
        puls.maj(evenements=ev, vitesse_norme=vitn, curiosite=cur, apprentissage=app, bullage=bul)
        obj = puls.objectif_dominant()
        if obj == "douleur":
            reflexes += 1
        dominants.append(obj)

    # bilan. La douleur (réflexe câblé prioritaire) masque le reste : on la met en
    # manchette PUIS on regarde la dynamique des pulsions sur les pas HORS-douleur.
    n = len(dominants)
    from collections import Counter
    taux_douleur = reflexes / n
    print(f"\nRéflexe douleur : {taux_douleur:.0%} des pas (agent ALÉATOIRE → percute sans cesse ; "
          f"{sucres} sucres mangés au hasard) — c'est ce qui motive la PLANIFICATION (étape 18-19).")

    hors = [x for x in dominants if x != "douleur"]
    def part_cog(seq):
        return sum(1 for x in seq if x in ("curiosite", "apprentissage")) / max(1, len(seq))
    d, f = hors[:len(hors) // 2], hors[len(hors) // 2:]
    print("\nObjectif dominant HORS douleur — début vs fin :")
    for lbl, seq in (("début", d), ("fin", f)):
        c = Counter(seq)
        print(f"   {lbl:>5} : " + ", ".join(f"{k} {v * 100 // len(seq)}%" for k, v in c.most_common(3)))
    switches = sum(1 for i in range(1, n) if dominants[i] != dominants[i - 1])
    print(f"\nCognitif (curiosité+apprentissage) hors douleur : {part_cog(d):.0%} au début → "
          f"{part_cog(f):.0%} à la fin  (la vision se maîtrise → la curiosité se sature)")
    print(f"Bascules d'objectif : {switches} sur {n} pas ({switches * 100 // n}%) → hystérésis effective")
    ok = (reflexes >= 1 and part_cog(d) > part_cog(f) and switches < n // 3)
    print("\nOK — un seul objectif dominant (argmax+hystérésis), la douleur court-circuite, "
          "et la curiosité se sature à mesure que la vision s'apprend" if ok else
          "\nà affiner")


if __name__ == "__main__":
    main()
