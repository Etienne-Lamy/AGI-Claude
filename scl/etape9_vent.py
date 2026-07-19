"""ÉTAPE 9 — le VENT : reconnaître un régime nouveau, et savoir OÙ ça rate.

Test d'émergence prévu par l'auteur. Un « vent » se lève : le monde défile
autrement alors que l'agent n'a rien changé à son état interne. Prédictions
attendues par la théorie (§29.1, §29.4) :

- **N1 (perception) doit rester INTACT** : le champ a la même apparence, le
  compresseur le reconstruit toujours aussi bien ;
- **N2 (dynamique) doit S'EFFONDRER** : aucun module-vitesse connu n'explique plus
  la transition → la familiarité chute ;
- le diagnostic doit donc localiser la panne **au niveau N2, pas N1** — et c'est là
  qu'il faut créer ;
- laissé libre, le système doit faire **NAÎTRE un module** pour ce régime, puis
  la familiarité doit **remonter** (le vent est un régime apprenable, pas du bruit).

    python3 -m scl.etape9_vent
"""
import argparse

import numpy as np

from .composition import _residu_champ
from .config import CONFIG
from .logger import set_temps
from .module_ae import DEVICE
from .monde import Monde
from .pipeline import construire


def _mesures(comp, det, monde, n, apprendre=False):
    """Retourne (résidu N1 = 1−rappel de reconstruction, familiarité N2, nb births)."""
    n1, n2 = [], []
    avant = len(det.vitesses)
    for s in range(n):
        set_temps(step=s)
        champ = monde.percevoir()["vision"][-1]
        # --- N1 : le compresseur reconstruit-il toujours le champ ?
        d = comp.fidelite(champ)
        n1.append(1.0 - d["rappel"])
        # --- N2 : un module-vitesse explique-t-il la transition ?
        if apprendre:
            det.etape(champ)
            actif, res, fam = None, {}, None
            # familiarité relue juste après (sans repousser le délai)
            n2.append(None)
        else:
            _, _, fam = det.identifier(champ)
            n2.append(fam)
        monde.appliquer_action((0, 0))
    n2v = [x for x in n2 if x is not None]
    return (float(np.mean(n1)), float(np.mean(n2v)) if n2v else None,
            len(det.vitesses) - avant)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pas_comp", type=int, default=2000)
    p.add_argument("--pas_regime", type=int, default=900)
    p.add_argument("--pas_mesure", type=int, default=300)
    p.add_argument("--pas_adaptation", type=int, default=1500)
    p.add_argument("--vent", type=int, nargs=2, default=[1, 0])
    args = p.parse_args()

    print(f"Device : {DEVICE} — construction du pipeline (sans vent)…")
    comp, det, n3, idx, noms, m, assoc = construire(
        args.pas_comp, args.pas_regime, pas_action=600)

    # --- référence : pas de vent, vitesse connue
    m.vent = np.array([0, 0], dtype=np.int64)
    m.vitesse = np.array([1, 0], dtype=np.int64)
    det.delai.pousser(None)
    n1_ref, n2_ref, _ = _mesures(comp, det, m, args.pas_mesure)

    # --- le vent se lève (l'agent n'a rien changé)
    m.vent = np.array(args.vent, dtype=np.int64)
    det.delai.pousser(None)
    n1_vent, n2_vent, _ = _mesures(comp, det, m, args.pas_mesure)

    print("\nDIAGNOSTIC PAR NIVEAU (§29.4) — où ça rate ?")
    print(f"{'':<28}{'sans vent':>12}{'avec vent':>12}")
    print(f"{'N1 résidu perception':<28}{n1_ref:>12.3f}{n1_vent:>12.3f}")
    print(f"{'N2 familiarité dynamique':<28}{n2_ref:>12.3f}{n2_vent:>12.3f}")
    d1 = n1_vent - n1_ref
    d2 = n2_ref - n2_vent
    print(f"\n   dégradation N1 : {d1:+.3f}   |   chute de familiarité N2 : {d2:+.3f}")
    if d2 > 0.15 and d1 < 0.15:
        print("   → panne localisée en **N2 (dynamique)**, N1 intact :")
        print("     régime nouveau, PAS une perception nouvelle → créer au niveau transition.")
    elif d1 >= 0.15:
        print("   → N1 dégradé aussi : perception nouvelle (créer plus bas).")
    else:
        print("   → aucune dégradation nette (le vent n'a pas changé le régime perçu).")

    # --- laissé libre, un module doit NAÎTRE pour ce régime, puis la familiarité remonter
    print(f"\nAdaptation libre sous vent {tuple(args.vent)} ({args.pas_adaptation} pas)…")
    for mv in det.vitesses.values():
        mv.verrouille = True          # les spécialistes connus restent protégés
    det.delai.pousser(None)
    avant = len(det.vitesses)
    for s in range(args.pas_adaptation):
        set_temps(step=s); det.etape(m.percevoir()["vision"][-1]); m.appliquer_action((0, 0))
    nes = len(det.vitesses) - avant
    det.delai.pousser(None)
    _, n2_apres, _ = _mesures(comp, det, m, args.pas_mesure)
    print(f"   modules nés : {nes}   |   familiarité N2 : {n2_vent:.3f} → {n2_apres:.3f}")
    ok = nes >= 1 and n2_apres > n2_vent + 0.1
    print("\nOK — le vent a été reconnu comme un RÉGIME NOUVEAU (module né, familiarité restaurée)"
          if ok else "\nà affiner — pas de reconnaissance nette du nouveau régime")


if __name__ == "__main__":
    main()
