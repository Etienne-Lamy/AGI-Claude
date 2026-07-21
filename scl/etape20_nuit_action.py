"""ÉTAPE 20 — le rejeu NOCTURNE AMONT fait émerger la visée du sucre (§7 conception).

Cycles JOUR/NUIT. Le jour : l'agent-valeur agit (ε-glouton), apprend en ligne (TD) et
ENREGISTRE ses épisodes. La nuit : `RejeuNocturne` rejoue en priorité les rares épisodes
récompensés, avec des retours n-pas → le crédit du sucre remonte en amont. Attendu : le
taux de sucre CROÎT de cycle en cycle et dépasse le hasard — là où le jour seul (étape 19)
restait au niveau du hasard. C'est la preuve de la navigation apprise par crédit amont.

    python3 -m scl.etape20_nuit_action
"""
import argparse
import random

import numpy as np

from .etape18_boucle_action import _agent_aleatoire
from .logger import set_temps
from .module_ae import DEVICE
from .monde import ACCELERATIONS_PERMISES, Monde
from .nuit_action import RejeuNocturne
from .planification import ModeleValeurQ
from .pulsions import Pulsions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cycles", type=int, default=4)
    p.add_argument("--jour", type=int, default=1500)
    p.add_argument("--nuit", type=int, default=1500)
    p.add_argument("--fenetre", type=int, default=12)     # longueur d'un épisode enregistré
    p.add_argument("--epsilon", type=float, default=0.15)
    args = p.parse_args()
    print(f"Device : {DEVICE} — {args.cycles} cycles jour({args.jour})/nuit({args.nuit})")

    random.seed(0)
    q = ModeleValeurQ(n_actions=len(ACCELERATIONS_PERMISES), gamma=0.95)
    rej = RejeuNocturne(n_pas=args.fenetre // 2)
    puls = Pulsions()
    monde = Monde(graine=7)
    s = np.asarray(monde.percevoir()["vision"][-1]).copy()

    taux_sucre = []
    for c in range(args.cycles):
        sucres, episode = 0, []
        for step in range(args.jour):
            set_temps(step=c * args.jour + step)
            a = q.choisir(s, epsilon=args.epsilon)
            ev = monde.appliquer_action(ACCELERATIONS_PERMISES[a])
            s2 = np.asarray(monde.percevoir()["vision"][-1]).copy()
            r = puls.recompense(evenements=ev)
            q.observer(s, a, r, s2)                 # apprentissage de JOUR (en ligne)
            episode.append((s, a, r))
            sucres += sum(1 for e in ev if e == "sucre")
            if len(episode) >= args.fenetre:
                rej.enregistrer(episode); episode = []
            s = s2
        tx = 1000 * sucres / args.jour
        taux_sucre.append(tx)
        perte = rej.nuit(q, passes=args.nuit)      # NUIT : rejeu amont priorisé
        print(f"   cycle {c + 1} : sucre {tx:.1f}/1000 (jour) ; nuit perte={perte:.3f}", flush=True)

    print(f"\nSucre/1000 par cycle (jour, monde d'entraînement) : {[round(t, 1) for t in taux_sucre]}")
    print("  (⚠ non comparable entre cycles : le sucre est consommé DÉFINITIVEMENT → déplétion)")

    # Évaluation PROPRE : politique GELÉE (ε=0, aucun apprentissage) sur un monde FRAIS
    # jamais vu (pas de déplétion héritée), comparée au hasard sur LE MÊME monde frais.
    def _eval(politique, graine, pas=2500):
        m = Monde(graine=graine); su = ba_ = 0
        s = np.asarray(m.percevoir()["vision"][-1]).copy()
        for _ in range(pas):
            a = politique(s)
            ev = m.appliquer_action(ACCELERATIONS_PERMISES[a])
            s = np.asarray(m.percevoir()["vision"][-1]).copy()
            su += sum(1 for e in ev if e == "sucre"); ba_ += sum(1 for e in ev if e == "baton")
        return 1000 * su / pas, 1000 * ba_ / pas

    GR = 314159                       # monde frais, absent de l'entraînement (graine=7)
    su_q, ba_q = _eval(lambda s: q.choisir(s, epsilon=0.0), GR)
    su_r, ba_r = _eval(lambda s: random.randrange(len(ACCELERATIONS_PERMISES)), GR)
    print(f"\nÉVALUATION monde frais (gelée) — sucre/1000 : agent {su_q:.1f} vs hasard {su_r:.1f} "
          f"→ {su_q - su_r:+.1f} ({100 * (su_q - su_r) / max(1e-9, su_r):+.0f}%)")
    print(f"                              bâtons/1000 : agent {ba_q:.1f} vs hasard {ba_r:.1f} "
          f"→ {ba_q - ba_r:+.1f}")
    gain = su_q - su_r
    degenere = ba_q < 5.0 and su_q < 10.0
    ok = gain > 5.0
    if ok:
        print("\nOK — après rejeu nocturne AMONT, la politique GELÉE vise le sucre sur un monde "
              "FRAIS, nettement au-dessus du hasard (généralisation) — là où le jour seul "
              "(étape 19) restait au hasard : c'est le NUIT qui fait la différence.")
    elif degenere:
        print("\nRÉSULTAT NÉGATIF HONNÊTE — la politique gelée DÉGÉNÈRE en VOL RECTILIGNE "
              f"(≈{ba_q:.0f} bâton, ≈{su_q:.0f} sucre /1000 : elle fuit TOUT, sucre compris). "
              "La récompense-sucre (contact rare, +0.4) est dominée par la pénalité de temps et "
              "l'évitement de douleur → le greedy converge vers « accélérer tout droit ». Le "
              "MÉCANISME de crédit amont est bon (test unitaire), mais sur CHAMP BRUT la valeur "
              "ne peut pas relier « sucre visible au loin » à « aller vers lui ». Prescription "
              "(du design doc §6) : PLANIFIER SUR L'ÉTAT COMPACT OBJET (slot-attention, étape 4), "
              "pas le champ brut — c'est la frontière de la prochaine session.")
    else:
        print(f"\nMÉCANISME EN PLACE (test unitaire) mais gain de navigation dans le bruit sur "
              f"monde frais ({gain:+.1f}/1000) — leviers : état compact objet, n-pas plus long.")


if __name__ == "__main__":
    main()
