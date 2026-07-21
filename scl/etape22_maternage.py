"""ÉTAPE 22 — placement maternel : cracker la visée du sucre par récompense DENSE (§7).

L'étape 20 a échoué à faire viser le sucre : trop rare au hasard, la politique dégénère en
vol rectiligne. Ici on ÉDUQUE l'agent : chaque leçon le pose à 2-3 cases d'un sucre (monde
frais → configurations variées). Il le trouve en 2-3 actions → récompense dense → la valeur
Q apprend « sucre visible → aller vers lui ». On rejoue les leçons la nuit (crédit amont).
Puis on ÉVALUE la politique GELÉE sur un monde FRAIS SANS placement : la visée doit
maintenant généraliser (sucre > hasard) — la preuve que l'éducation a pris.

    python3 -m scl.etape22_maternage --graine_eval 314159
"""
import argparse
import random

import numpy as np

from .logger import set_temps
from .maternage import action_vers_sucre, placer_pres_sucre
from .module_ae import DEVICE
from .monde import ACCELERATIONS_PERMISES, Monde
from .nuit_action import RejeuNocturne
from .planification import ModeleValeurQ
from .pulsions import Pulsions


def _lecon(q, rej, puls, graine, rng, epsilon, max_pas):
    """Une leçon : placer près d'un sucre ; la MAMAN démontre l'action vers le sucre
    (sauf exploration ε) → l'agent l'atteint en 2-3 pas → récompense DENSE et propre. Q
    apprend « champ avec sucre visible → cette action → récompense » ; à l'éval, Q seul
    (champ brut) devra généraliser."""
    monde = Monde(graine=graine)
    if placer_pres_sucre(monde, rng) is None:
        return 0
    s = np.asarray(monde.percevoir()["vision"][-1]).copy()
    episode, mange = [], 0
    for _ in range(max_pas):
        demo = action_vers_sucre(monde)                       # démonstration de la maman
        if demo is None or random.random() < epsilon:
            a = random.randrange(len(ACCELERATIONS_PERMISES))  # exploration (contraste pour Q)
        else:
            a = ACCELERATIONS_PERMISES.index(demo)
        ev = monde.appliquer_action(ACCELERATIONS_PERMISES[a])
        s2 = np.asarray(monde.percevoir()["vision"][-1]).copy()
        r = puls.recompense(evenements=ev)
        q.observer(s, a, r, s2)
        episode.append((s, a, r))
        s = s2
        if any(e == "sucre" for e in ev):
            mange += 1
            break
    rej.enregistrer(episode)
    return mange


def _evaluer(politique, graine, pas=2500):
    m = Monde(graine=graine); su = ba = 0
    s = np.asarray(m.percevoir()["vision"][-1]).copy()
    for _ in range(pas):
        a = politique(s)
        ev = m.appliquer_action(ACCELERATIONS_PERMISES[a])
        s = np.asarray(m.percevoir()["vision"][-1]).copy()
        su += sum(1 for e in ev if e == "sucre"); ba += sum(1 for e in ev if e == "baton")
    return 1000 * su / pas, 1000 * ba / pas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lecons", type=int, default=2500)
    p.add_argument("--max_pas", type=int, default=6)
    p.add_argument("--epsilon", type=float, default=0.25)
    p.add_argument("--nuit_tous", type=int, default=250)
    p.add_argument("--graine_eval", type=int, default=314159)
    args = p.parse_args()
    print(f"Device : {DEVICE} — {args.lecons} leçons de placement maternel")

    random.seed(0)
    rng = np.random.default_rng(0)
    q = ModeleValeurQ(n_actions=len(ACCELERATIONS_PERMISES), gamma=0.95)
    rej = RejeuNocturne(n_pas=args.max_pas)
    puls = Pulsions()

    manges = 0
    for l in range(args.lecons):
        set_temps(step=l)
        manges += _lecon(q, rej, puls, graine=1000 + l, rng=rng,
                         epsilon=args.epsilon, max_pas=args.max_pas)
        if (l + 1) % args.nuit_tous == 0:
            rej.nuit(q, passes=1000)
            print(f"   {l + 1} leçons : sucres trouvés {manges} ({100 * manges // (l + 1)}%) ; nuit", flush=True)

    print("\nÉVALUATION monde FRAIS (gelée, sans placement) :")
    su_q, ba_q = _evaluer(lambda s: q.choisir(s, epsilon=0.0), args.graine_eval)
    su_r, ba_r = _evaluer(lambda s: random.randrange(len(ACCELERATIONS_PERMISES)), args.graine_eval)
    print(f"   sucre /1000 : agent {su_q:.1f} vs hasard {su_r:.1f} → {su_q - su_r:+.1f} "
          f"({100 * (su_q - su_r) / max(1e-9, su_r):+.0f}%)")
    print(f"   bâtons/1000 : agent {ba_q:.1f} vs hasard {ba_r:.1f} → {ba_q - ba_r:+.1f}")
    ok = su_q > su_r + 5.0
    if ok:
        print("\nOK — l'éducation a PRIS : la politique gelée VISE le sucre sur un monde frais, "
              "au-dessus du hasard, sans aucun placement — la visée du sucre a émergé.")
    else:
        deg = ba_q < 5 and su_q < 10
        print("\nÉCHEC" + (" (politique dégénérée en vol rectiligne)" if deg else "") +
              f" — visée non généralisée (sucre {su_q:.1f} vs hasard {su_r:.1f}). "
              "Leviers : plus de leçons, distance croissante, ε plus haut, état compact objet.")


if __name__ == "__main__":
    main()
