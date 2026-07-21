"""DÉMO VIEWER v7 — produit un log JSONL riche pour voir, en direct :

  1. les MODULES qui naissent et s'entraînent (rappel par module au fil du temps) ;
  2. la RECONSTRUCTION du champ (VU vs PRÉVU par le module actif) ;
  3. le GRAPHE DE BRANCHEMENT que l'orchestrateur compose (programmes typés + valeur).

Deux phases écrites dans le même log :
  — phase « modules » : détection de régime en espace-champ sur des régimes qui changent
    (naissances, courbes de rappel, instantanés de champ) ;
  — phase « orchestrateur » : Mode A énumère et évalue les programmes typés (le graphe).

    python3 -m scl.demo_viewer --log demo.jsonl
    python3 viewer.py --log demo.jsonl --port 8400     # dans un 2e terminal

Vocabulaire v7 (acteur « viewer ») : meta, modules_etat, champ, phase, programme_choisi ;
+ réutilise regime/naissance_module_regime et orchestrateur/programme_evalue.
"""
import argparse
import time

import numpy as np

from .logger import configurer, log, set_temps
from .module_ae import DEVICE
from .monde import Monde
from .orchestrateur import mode_A
from .regime import DetecteurRegimeChamp

REGIMES = [(1, 0), (2, 0), (0, 1)]


def _champ(monde):
    return np.asarray(monde.percevoir()["vision"][-1]).copy()


def _grille(x, h, w):
    """Champ (array/tenseur plat ou 2D) → liste de listes arrondies pour le log."""
    a = np.asarray(x, dtype=float).reshape(h, w)
    return [[round(float(v), 3) for v in ligne] for ligne in a]


def _etats_modules(det, actif):
    return [{"id": mid, "rappel": round(float(e.ema_rappel), 3) if e.ema_rappel is not None else None,
             "verrouille": bool(e.verrouille), "actif": (mid == actif)}
            for mid, e in det.regimes.items()]


def phase_modules(det, pas_par_regime, periode):
    h = w = None
    step = 0
    for v in REGIMES:
        m = Monde(graine=1); m.vitesse = np.array(v, dtype=np.int64)
        det.champ_prec = None
        prec = None
        for _ in range(pas_par_regime):
            set_temps(step=step)
            champ = _champ(m)
            if h is None:
                h, w = champ.shape
                log("viewer", "meta", h=h, w=w, regimes=[str(r) for r in REGIMES])
            actif, _ = det.etape(champ, apprendre=True)
            # instantané périodique : champ VU, champ PRÉVU par le module actif, état modules
            if prec is not None and actif is not None and step % periode == 0:
                mod = det.regimes[actif].module
                prevu = mod.predire(prec)
                rap = mod.fidelite_transition(prec, champ)["rappel"]
                log("viewer", "champ", regime=str(v), module=actif, rappel=round(float(rap), 3),
                    vu=_grille(champ, h, w), prevu=_grille(prevu, h, w))
                log("viewer", "modules_etat", regime=str(v), etats=_etats_modules(det, actif))
            prec = champ
            m.appliquer_action((0, 0))
            step += 1
        log("viewer", "phase", nom="modules", regime_fini=str(v), n_modules=len(det.regimes))
    det.verrouiller_tout()
    return h, w


def phase_orchestrateur(pas):
    """Mode A évalue les programmes typés → le graphe de branchement (programme_evalue)."""
    v = (1, 0)
    me = Monde(graine=555); me.vitesse = np.array(v, dtype=np.int64)
    paires, prec = [], None
    for _ in range(80):
        c = _champ(me)
        if prec is not None:
            paires.append((prec, c))
        prec = c
        me.appliquer_action((0, 0))

    def flux(n):
        m = Monde(graine=1); m.vitesse = np.array(v, dtype=np.int64)
        p = None
        for s in range(n):
            set_temps(step=s)
            c = _champ(m)
            yield p, c
            p = c
            m.appliquer_action((0, 0))

    log("viewer", "phase", nom="orchestrateur", info="Mode A évalue les programmes typés")
    classement = mode_A(flux, paires, pas=pas)
    best = classement[0]
    log("viewer", "programme_choisi", chaine=best["chaine"], G=best["G"],
        cout=best["cout"], valeur=best["valeur"])
    return classement


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="demo.jsonl")
    p.add_argument("--pas_regime", type=int, default=1500)
    p.add_argument("--periode", type=int, default=25)
    p.add_argument("--pas_orchestrateur", type=int, default=500)
    args = p.parse_args()

    configurer(chemin=args.log, verbeux=False)
    log("viewer", "phase", nom="demarrage", device=str(DEVICE), horodatage=time.time())
    print(f"Log : {args.log}  (device {DEVICE}) — lance `python3 viewer.py --log {args.log}` en parallèle")

    det = DetecteurRegimeChamp()
    phase_modules(det, args.pas_regime, args.periode)
    classement = phase_orchestrateur(args.pas_orchestrateur)

    print("Phase modules :", len(det.regimes), "modules ;",
          "meilleur programme :", classement[0]["chaine"],
          f"(valeur={classement[0]['valeur']})")
    print("Démo terminée — le log est complet, le viewer peut le rejouer.")


if __name__ == "__main__":
    main()
