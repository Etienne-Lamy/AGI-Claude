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

## Confiance
- Quand tu dis avoir fait une action toi-même (push, install, suppression, config...), c'est pris pour vrai : pas de vérification par outil pour confirmer. On se fait confiance ; les malentendus éventuels, tu les assumes.
- Exception : si l'étape suivante a besoin d'un état précis pour agir dessus (ex. un hash de commit exact), vérifier reste légitime — mais c'est une vérification utile à l'action, pas un contrôle de ta parole.

## Environnement GPU / CUDA
GPU cible : NVIDIA GTX Titan Black (Kepler, sm_35) — les wheels PyTorch officiels ne supportent plus sm_35, d'où un build maison (CUDA 10.2 + gcc 8 + Python 3.10), 5 jours de mise au point (voir `~/pytorch-kepler/CRASH_RECOVERY_CONTEXT.md`).

Activation obligatoire avant tout usage de torch avec GPU (dans cet ordre) :
```bash
source ~/venv_pytorch_kepler/bin/activate
source ~/pytorch-kepler/dist/setup_env.sh
```
- torch 1.12.0a0 (build custom). En cas de problème d'import/lib : `~/pytorch-kepler/dist/WHEEL_BACKUP_README.md` et `CRASH_RECOVERY_CONTEXT.md`.
- CUDA validé le 2026-07-17 : `torch.cuda.is_available()` OK, matmul 2048×2048 : 4.3 ms GPU vs 10.5 s CPU (CPU limité à 2 cœurs sous cette config WSL — écart non représentatif, ne pas s'en servir comme référence perf CPU générale).

## Git
- Un commit = une version fonctionnelle (tests passants, ou état explicitement documenté comme WIP dans le message de commit).
- Aucun `push`, création/modification de remote, ou action GitHub (PR, issue...) sans confirmation explicite.

## Machine WSL
- Pas jetable. `git status` systématique avant tout `reset` / `clean` / `checkout` destructeur. Pas de `rm -rf` hors dossiers scratch explicitement désignés.

## Délégation
- Pas de contrainte hors coût tokens : sous-agents autorisés pour exploration large (lecture de code, recherche). Contexte principal réservé à la décision et à la synthèse.
