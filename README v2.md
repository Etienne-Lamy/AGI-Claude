# SCL (Structural Continual Learning) — État du projet

## Ce que c'est

Tentative d'architecture d'IA où la structure de calcul (quels modules existent, comment ils sont connectés) émerge par l'expérience plutôt que d'être fixée à l'avance. Testée sur un petit monde simulé 2D (un point mobile qui perçoit un champ visuel 10×10, cherche des sucres, évite des bâtons).

**État actuel : le rez-de-chaussée (perception visuelle de base) n'a jamais convergé. Tout l'étage construit au-dessus (gestion de graphe, cycle nocturne, orchestrateur, pilote) est fonctionnellement correct mais n'a jamais pu être validé en pratique faute d'un module de vision qui marche.**

Ce document explique ce qui a été codé, ce qui marche, ce qui ne marche pas, et pourquoi.

---

## Le monde de simulation

```
Grille infinie 2D. Le corps est un point fixe au centre du champ visuel — c'est le monde qui défile.
Vitesse (vx, vy) ∈ {-2..2}², 4 actions possibles : ±1 sur vx ou vy (clip à [-2,2]).
Champ visuel : fenêtre 10×10, 3 frames glissantes, cases ∈ {vide, sucre, bâton}.
Besoins : faim, énergie, douleur — dynamiques continues, faim/énergie coûtent avec le déplacement.
Réflexe inné câblé : contact bâton → freinage (jamais appris, jamais modifié).
```

---

## Architecture générale (fonctionnellement correcte, jamais éprouvée en pratique)

### Module (unité de base)
Chaque module a une paire encodeur (reconnaissance)/décodeur (génération), chacun avec son propre optimiseur. Un condensateur de certitude (0 à 1) monte avec le succès, descend avec l'échec ; au-delà d'un seuil le module se **verrouille** (poids figés, plus de gradient). Les deux directions (reco/gen) verrouillent indépendamment.

### Pas de backward global
Chaque module apprend sur sa propre tâche locale, jamais sur un gradient qui traverse d'autres modules. L'orchestrateur ne fait que du routage — activation/inhibition (mécanisme de type gate + inhibition latérale locale entre modules concurrents).

### Cycle jour/nuit
Le jour : perception, action, entraînement lent des poids des modules actifs. La nuit : croissance de capacité des modules candidats, clustering des embeddings, consolidation du routage de l'orchestrateur, détection de modules obsolètes (watchdog).

### Mécanismes de création/modification structurelle de modules
- **Rupture** : erreur de prédiction localisée à un point précis du graphe → création d'un module candidat à cet endroit.
- **Fragmentation** : un module dont la certitude s'effondre est scindé en deux (règle générale + exception).
- **Découpe** : un module globalement compétent mais avec une variance conditionnelle d'erreur (bon sur un sous-contexte, mauvais sur un autre) est scindé en noyau + module amovible additif, branché uniquement sur la variable discriminante isolée.

### Orchestrateur / pilote de chantiers
Catalogue des flux disponibles (capteurs + latents des modules démontrés), priorisation par gain mesuré (compression pour les paires identité, capacité prédictive au-delà de la persistance pour les paires croisées), un seul chantier actif à la fois, démonstration après N nuits sous un seuil d'erreur relative, verrouillage puis nouveau flux disponible pour le chantier suivant. Une matrice d'attention flux×flux apprend nuit après nuit quelles paires sont efficaces.

### Mémoire de graines et rêve
À chaque variation notable d'un besoin, la représentation latente minimale du moment est mémorisée (pas le signal brut). La nuit, ces graines sont rejouées à travers les générateurs déjà verrouillés (poids figés) pour produire des trajectoires simulées, sur lesquelles seul le module en cours d'apprentissage s'entraîne.

### Instrumentation
Dashboard web (viewer.py + serveur) lisant un log JSONL (scl_audit.jsonl), affichant courbes de condensateurs/erreurs, panneau "ce qu'il prévoit" (reconstruction générée par le module de vision), fil d'événements (ruptures, verrouillages, watchdogs, décisions du pilote avec alternatives chiffrées), clusters d'embeddings, barreaux de progression (critères É1 à É4, voir plus bas).

---

## Ce qui a échoué, et pourquoi (racine du problème)

### Le module de vision n'a jamais convergé

Trois générations de tentatives, toutes en échec :

**V1 — MLP dense (300 → 32 → 300).** Ne peut pas représenter efficacement une translation (le défilement du champ visuel dominant). Erreur de reconstruction jamais satisfaisante, `gen` (condensateur génératif) proche de 0 en permanence — la voie génération n'apprenait jamais rien d'utile.

**V2 — Convolution simple.** Biais structurel correct en principe (une conv représente naturellement une translation), mais implémentée en aval d'un pipeline où l'encodeur était entraîné à un objectif contradictoire (minimiser la variation temporelle du latent — "prédire que rien ne change" — qui est l'opposé de ce qu'il faut pour permettre au décodeur de reconstruire une scène qui bouge). L'encodeur sabotait le générateur par construction de la loss.

**V3 — Primitive "slots" artisanale.** Représentation en K slots (présence, x, y, intensité), génération par rendu différentiable de bosses gaussiennes, reconnaissance par extraction de pics puis recherche de cible. Quatre bugs successifs, jamais tous résolus simultanément :
1. Bruit de permutation — les pics étaient triés par valeur, pas par identité stable d'objet, donc la cible d'apprentissage changeait d'assignation à chaque frame → l'encodeur apprenait un bruit chaotique.
2. Échelle des coordonnées non normalisée, dominant la loss et rendant les seuils incomparables.
3. Le générateur, entraîné sur la sortie (bruitée) de l'encodeur plutôt que sur une cible propre, convergeait vers une solution dégénérée (gain → 0, reconstruction vide) qui minimise trivialement l'erreur sans rien apprendre.
4. Ancrage temporel des cibles insuffisant — même corrigé sur le tri, la référence utilisée pour aligner les slots restait instable au démarrage (référence = sortie d'un encodeur encore aveugle = bruit).

**Cause racine identifiée a posteriori, jamais appliquée à temps :** ce module aurait dû être un autoencodeur convolutionnel standard, entraîné en isolation totale (script séparé, sans orchestrateur, sans monde vivant, sans besoins/faim/nuit mélangés) par apprentissage mutuel encodeur/décodeur sur la seule tâche de reconstruction — exactement comme un autoencodeur classique. Au lieu de ça, une primitive non standard a été inventée (les slots), déployée directement dans le système complet, où le bruit du monde vivant (faim, action, cycle nocturne, création/fragmentation de modules concurrents) rendait tout diagnostic ambigu.

### Interprétation erronée de la contrainte "pas de backward global"

Cette contrainte, dans sa spécification d'origine, interdit les gradients **entre modules différents** (pas de rétropropagation qui traverse tout le graphe). Elle a été appliquée trop littéralement, empêchant même l'entraînement mutuel encodeur/décodeur **à l'intérieur d'un seul module** — ce qui est pourtant parfaitement conforme à l'intention d'origine et nécessaire au fonctionnement d'un autoencodeur. Cette confusion a bloqué la conception correcte pendant une grande partie du projet.

### Déploiement direct dans le système complet sans validation isolée

Chaque correctif était appliqué directement dans la boucle complète (monde vivant + orchestrateur + besoins + cycle nocturne + machinerie de création de modules), tous actifs simultanément. Impossible de savoir si un échec venait de l'encodeur, du décodeur, du pilote, de l'interaction avec la faim, ou d'un bug de plomberie (log, index, etc.) — plusieurs bugs de plomberie réels ont d'ailleurs été trouvés et corrigés en cours de route (division par erreur relative sur énergie nulle, priorisation du pilote biaisée par une moyenne au lieu d'une somme, désynchronisation entre pilotage et instruments de mesure, perte du flag "inné" lors d'une copie de module ayant entraîné la suppression accidentelle d'un module protégé) — mais aucun n'a suffi tant que le problème de fond (le design du module vision lui-même) n'était pas réglé.

### Ce qui, en revanche, fonctionnait correctement

- Le mécanisme de condensateur/verrouillage/watchdog, une fois les bugs de mesure d'erreur relative corrigés.
- Le pilote de chantiers : la formule de priorisation (énergie totale du flux, gain mesuré par sonde) choisissait correctement de commencer par la compression du champ visuel plutôt que par la proprioception (plus simple mais moins riche en information) — preuve que le principe de priorisation générique fonctionne, indépendamment de l'échec du module vision lui-même.
- La détection de rupture, une fois protégée contre les faux positifs sur modules trop jeunes (maturité minimale en nombre de tentatives avant d'être éligible à une fragmentation).
- L'atrophie nocturne des modules zombies (jamais performants après maturité).
- Le clustering d'embeddings avec rapprochement hebbien préalable (modules co-actifs rapprochés avant clustering, évitant que le clustering ne reflète que le bruit d'initialisation).
- La consolidation nocturne par rêve (rejeu de graines à travers des générateurs figés) — mécaniquement correcte, jamais testée à fond faute d'avoir des générateurs fiables en amont.

---

## Fichiers du projet (état au moment de la passation)

```
scl/module.py           — Module de base (encodeur/décodeur, condensateur, verrouillage)
scl/module_slots.py      — Primitive vision V3 (slots) — À ABANDONNER, voir section échecs
scl/orchestrateur.py      — Gate, inhibition latérale, sens bottom-up/top-down
scl/pilote.py               — Pilote de chantiers (priorisation générique) — fonctionnellement validé
scl/graphe.py                 — Structure de graphe, forward_graphe (perception/imagination/fusion)
scl/cycle_nocturne.py          — Croissance, clustering, watchdog, rêve, consolidation
scl/monde.py                     — Simulation du monde 2D décrit plus haut
scl/memoires.py                    — Graines, mémoire tampon, registre de câblage
scl/logger.py                        — Journalisation JSONL (attention : le format doit exclure NaN/Inf,
                                        cause de crash du viewer, corrigé mais à surveiller)
run_poc.py                             — Boucle principale, checkpoint (.pkl — lié à la version du code,
                                        à supprimer après toute modification de structure)
viewer.py + serveur                       — Dashboard web (à relancer après toute modification du filtre de log)
tests/                                      — ~33 tests, couvrant condensateurs, inhibition latérale,
                                        rupture, fragmentation, pilote, différentiabilité
```

---

## Recommandation pour la suite

Voir le document `PROMPT_TRANSFERT_SCL.md` joint, qui spécifie la marche à suivre : reconstruire le module de vision comme un autoencodeur convolutionnel standard, validé en isolation totale avant toute réintégration, puis laisser le pilote de chantiers (déjà fonctionnel) faire émerger la suite. Ne pas reprendre la primitive à slots. Ne pas redéployer directement dans le système complet sans preuve de convergence isolée au préalable.
