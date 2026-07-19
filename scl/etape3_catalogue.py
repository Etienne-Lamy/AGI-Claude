"""ÉTAPE 3 — l'orchestrateur naïf choisit la TAILLE du module par MDL (§5).

Démonstration : au lieu que JE fixe la dimension du goulot, on donne un catalogue
de tailles, on entraîne un candidat pour chacune, et on garde celui de plus petit
MDL (code compressé + résidu). On observe que le MDL ne choisit ni le plus petit
(reconstruction insuffisante) ni le plus grand (code trop cher), mais une taille
qui EXTRAIT l'information à valeur.

    python3 -m scl.etape3_catalogue --pas 1500
"""
import argparse

import numpy as np

from .config import CONFIG
from .logger import set_temps
from .module_ae import DEVICE
from .monde import Monde
from .orchestrateur_naif import essayer_catalogue


def _flux_train_fn(vitesse, graine):
    def gen(pas):
        m = Monde(graine=graine)
        m.vitesse = np.array(vitesse, dtype=np.int64)
        for s in range(pas):
            set_temps(step=s)
            yield m.percevoir()["vision"][-1]
            m.appliquer_action((0, 0))
    return gen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pas", type=int, default=1500)
    p.add_argument("--vitesse", type=int, nargs=2, default=[1, 1])
    args = p.parse_args()
    v = tuple(args.vitesse)

    # jeu d'évaluation FIXE (tenu à l'écart de l'entraînement)
    me = Monde(graine=99); me.vitesse = np.array(v, dtype=np.int64)
    champs_eval = []
    for _ in range(40):
        for _ in range(5): me.appliquer_action((0, 0))
        champs_eval.append(np.asarray(me.percevoir()["vision"][-1]).copy())

    print(f"Device : {DEVICE} — catalogue {CONFIG['catalogue_dims_module']}, "
          f"{args.pas} pas/candidat, coût={CONFIG['bits_par_dim_mdl']} bits/dim")
    meilleur, res = essayer_catalogue(
        _flux_train_fn(v, graine=1), champs_eval, pas=args.pas)

    print(f"\n{'dim':>4} {'rappel':>8} {'code(bits)':>11} {'residu(bits)':>13} {'MDL':>8}")
    for r in res:
        etoile = "  ← CHOISI (MDL min)" if r["dim"] == meilleur["dim"] else ""
        print(f"{r['dim']:>4} {r['rappel']:>8.0%} {r['code']:>11.0f} "
              f"{r['residuel']:>13.1f} {r['mdl']:>8.1f}{etoile}")
    print(f"\nL'orchestrateur naïf choisit dim={meilleur['dim']} "
          f"(entrée = {CONFIG['taille_perception']**2} valeurs → goulot {meilleur['dim']}, "
          f"rappel {meilleur['rappel']:.0%}). C'est le compromis parcimonie/fidélité (MDL).")


if __name__ == "__main__":
    main()
