# Règles de travail — AGI-Claude / SCL

Projet : SCL (Structural Continuous Learning). Docs de fond, source de vérité — lues à la demande, jamais dupliquées ici :
`README v2.md`, `SCL - Vision et Strategie.md`, `Architecture SCL Code v2.md`, `SCL_fondements_mathematiques.md`, `STATUS.md`

## Style
- Réponses courtes, denses. Pas de récap de fin sauf demande explicite. Pas de blabla de transition.
- Français.

## Sessions & mémoire
- Ne pas s'appuyer sur une conversation longue : chaque session est traitée comme potentiellement la dernière avant repartir à vide.
- Après tout travail significatif, mettre à jour `STATUS.md` (fait / reste à faire / décisions / écarts). C'est le point de reprise à froid, pas l'historique du chat.
- Ne pas répéter dans le chat ce qui est déjà dans `STATUS.md` ou les docs de fond.

## Git
- Un commit = une version fonctionnelle (tests passants, ou état explicitement documenté comme WIP dans le message de commit).
- Aucun `push`, création/modification de remote, ou action GitHub (PR, issue...) sans confirmation explicite.
- Environnement Python du projet : `/home/ubuntu/IA-Ubuntu/bin/python` (venv).

## Machine WSL
- Pas jetable. `git status` systématique avant tout `reset` / `clean` / `checkout` destructeur. Pas de `rm -rf` hors dossiers scratch explicitement désignés.

## Délégation
- Pas de contrainte hors coût tokens : sous-agents autorisés pour exploration large (lecture de code, recherche). Contexte principal réservé à la décision et à la synthèse.
