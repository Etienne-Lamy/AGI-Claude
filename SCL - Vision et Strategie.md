# SCL — Structural Continual Learning

### Vision, paris de conception, mécanique — Pourquoi, Quoi, Comment

Ce document est le prompt de référence du projet : ce qu'il faut savoir, comprendre et accepter pour concevoir, coder ou juger SCL. Il ne remplace pas le document théorique (`SCL_fondements_mathematiques`, seul texte faisant autorité sur le détail formel) ni l'architecture de code (`Architecture SCL Code v2`), qui restent les références techniques. Ici, l'objectif est différent : relier la vision de long terme aux décisions concrètes, et expliquer pourquoi chacune a été prise.

---

## I. Pourquoi — la vision

### Le problème qu'on refuse d'accepter

Un réseau de neurones actuel a une structure fixée avant l'entraînement : nombre de couches, largeur, connectivité — tout est décidé par l'architecte humain, une fois, à l'avance. L'apprentissage ajuste des poids à l'intérieur d'un squelette immuable. Deux conséquences qu'on refuse comme fatalité :

- **L'oubli catastrophique.** Réentraîner sur une nouvelle tâche dégrade les anciennes, parce que tout le réseau partage les mêmes paramètres et le même gradient global. Rien ne protège ce qui a déjà été appris.
- **La structure ne s'adapte pas au vécu.** Le réseau ne peut pas décider, de lui-même, qu'il a besoin d'un nouveau sous-système parce qu'il rencontre une situation qu'aucun de ses modules actuels ne couvre. La seule réponse disponible est : plus de paramètres, dès le départ, "au cas où".

Un cerveau biologique ne fonctionne pas ainsi. Il ne réentraîne pas ses 86 milliards de neurones à chaque expérience nouvelle. Il recrute localement, verrouille prudemment ce qui marche, laisse le reste plastique, et ne casse presque jamais ce qu'il a déjà consolidé. La structure elle-même — combien de circuits, pour quoi faire, comment ils se composent — est une conséquence de l'expérience vécue, pas un plan dessiné avant la naissance.

### L'ambition

SCL est un pari : une architecture où la structure computationnelle **émerge** de l'expérience au lieu d'être fixée a priori. Le système démarre quasi vide — quelques réflexes câblés, un module sensoriel par défaut — et construit, au fil du temps, sa propre modularité, sa propre hiérarchie, sa propre mémoire, sa propre allocation d'attention, en réponse à ce qu'il rencontre réellement. Aucun réentraînement global n'a jamais lieu. Chaque ajout est local, chaque protection est locale, chaque décision de grandir ou de ne pas grandir est prise au niveau du point de rupture, pas au niveau du réseau entier.

L'horizon final est un système capable d'apprentissage continu véritable : accumuler indéfiniment sans dégrader ce qui est acquis, construire sa propre complexité au rythme où elle devient nécessaire, et rester interprétable parce que chaque morceau de sa structure correspond à quelque chose qu'il a réellement rencontré et appris à traiter.

### Pourquoi un monde-jouet 2D

On ne teste pas cette hypothèse dans le monde réel d'emblée. Un monde 2D minimal (perception visuelle grossière, quelques besoins — faim, douleur —, quelques objets — sucre, bâton) suffit à faire apparaître les phénomènes qu'on veut observer : détection de rupture, création de module, composition, consolidation, oubli sélectif. Si les mécanismes ne fonctionnent pas ici, ils ne fonctionneront pas dans un monde plus riche. La complexité du monde n'est pas ce qu'on teste ; c'est la dynamique de croissance structurelle qu'on teste.

---

## II. Quoi — les paris de conception fondamentaux

Chaque pari ci-dessous répond à une question de la Partie I. Chacun est un choix assumé, pas une évidence — certains sont des piliers démontrés dans la littérature, d'autres des hypothèses que l'expérience doit trancher.

### 1. Modularité locale, jamais de gradient global

Le système est un graphe de modules $(E_i, G_i)$ — un encodeur et un générateur — chacun entraîné uniquement sur ses propres exemples, avec son propre optimiseur. Aucun gradient ne traverse une frontière de module. C'est l'équivalent computationnel de l'apprentissage hebbien local dans le cortex : chaque aire spécialisée ajuste ses propres synapses en fonction de ce qui l'active, sans qu'un signal d'erreur global ne redescende recalibrer l'ensemble du cerveau à chaque expérience.

Conséquence directe : on peut ajouter, verrouiller ou retirer un module sans toucher au reste. C'est la condition nécessaire pour éviter l'oubli catastrophique — si rien ne partage les paramètres, rien ne peut se marcher dessus.

### 2. La croissance est déclenchée, jamais planifiée

Un nouveau module n'apparaît pas parce que l'architecte a prévu qu'il en faudrait un. Il apparaît parce qu'un point précis du graphe échoue de façon statistiquement confirmée (test séquentiel, pas un seuil arbitraire ponctuel) et que la réparation locale, puis la composition avec l'existant, ont toutes deux échoué. C'est l'analogue d'un déclenchement de plasticité synaptique ou de branchement dendritique en réponse à un signal de surprise ou d'erreur soutenue — pas un plan de développement écrit à l'avance.

Ce principe se décline en cascade : détection de rupture → réparation locale → recherche compositionnelle → seulement en dernier recours, création d'un module neuf. Chaque étage est moins coûteux et plus rapide que le suivant, et on ne passe à l'étage suivant qu'après échec confirmé de celui d'avant.

### 3. Verrouillage asymétrique : un plancher, jamais un plafond

Un module qui a prouvé sa fiabilité voit sa vitesse d'apprentissage se réduire fortement (condensateur, verrouillage) — il devient stable, prévisible, il ne se dégrade plus au gré du bruit. Mais ce verrouillage n'est jamais absolu : une mise à jour ultérieure qui améliore réellement le module (test de non-infériorité formel, jamais un simple "ça a l'air mieux") est toujours acceptée. C'est l'image de la mémoire procédurale — le vélo qu'on ne "réapprend" pas à chaque sortie, mais dont le geste peut encore s'affiner years plus tard sans qu'on doive tout redémarrer à zéro.

Sans plancher : le système oublierait sans cesse (instabilité). Sans porte de sortie : le système fossiliserait ses erreurs (rigidité). Le pari est que la protection doit être asymétrique pour obtenir les deux propriétés à la fois.

### 4. Deux mémoires, pas une

Un tampon rapide, à courte fenêtre, indexé par décalage relatif (jamais par horaire absolu — sinon la mémoire devient obsolète à mesure que le temps avance), et une mémoire lente, paramétrique, consolidée pendant le cycle nocturne. C'est directement la théorie des Complementary Learning Systems (hippocampe rapide et plastique / néocortex lent et stable) : on ne peut pas avoir un seul système qui soit à la fois capable d'apprendre vite un événement isolé et de généraliser lentement sans l'oublier. Il en faut deux, avec un mécanisme de transfert entre eux (le rejeu nocturne).

### 5. La parcimonie comme moteur, pas comme contrainte a posteriori

Le critère qui gouverne la compression, la croissance et l'acceptation d'un module n'est pas "est-ce que ça marche", mais un critère de longueur de description (MDL) : est-ce que ce module réduit vraiment la complexité totale nécessaire pour expliquer ce qui est observé. C'est l'analogue du codage parcimonieux (Olshausen & Field) : le cortex visuel ne s'active pas en masse pour chaque stimulus, un petit sous-ensemble de neurones porte l'essentiel de l'information, avec un dictionnaire surcomplet dans lequel peu d'éléments sont actifs à la fois. Le système SCL vise la même signature : à tout instant, seule une fraction des modules disponibles est active (activation creuse), le reste dort jusqu'à ce qu'il redevienne pertinent.

### 6. La nouveauté se détecte statistiquement, pas par seuil arbitraire

Chaque décision qui déclenche une action structurelle irréversible (créer un module, considérer qu'un module dérive, considérer qu'une réparation a échoué de façon durable) repose sur un test séquentiel (SPRT) avec des taux d'erreur de premier et deuxième ordre explicitement choisis — pas sur "l'erreur dépasse 0,3". Un seul incident isolé ne déclenche jamais une action lourde ; c'est l'accumulation confirmée d'évidence qui le fait. C'est ce qui évite au système de sur-réagir au bruit (halluciner un besoin de structure à chaque anomalie ponctuelle) tout en restant sensible à un vrai changement soutenu.

### 7. Un orchestrateur qui compose, ne calcule jamais lui-même

Le rôle de l'orchestrateur n'est pas de résoudre les tâches — c'est de décider *qui*, parmi les modules existants, doit s'en charger, et dans quel ordre les brancher. Il pointe (source, opérateur, cible), il n'invente jamais un calcul de novo. C'est l'analogue fonctionnel du cortex préfrontal : il alloue l'attention et orchestre les aires spécialisées, il ne voit pas lui-même, il ne bouge pas lui-même les muscles. La profondeur de ce que le système peut "penser" n'est donc jamais une capacité arbitraire de l'orchestrateur — c'est une conséquence directe de la qualité de compression des modules qu'il compose.

### 8. Toute composition doit s'ancrer à un point de vérité vérifiable

Composer des modules en cascade sans jamais revérifier contre quelque chose de réel produit un raisonnement en roue libre, qui peut diverger silencieusement de la réalité. Chaque chaîne de composition doit donc atteindre, à un moment donné, soit une observation brute, soit un module déjà certifié — jamais uniquement d'autres suppositions non vérifiées empilées les unes sur les autres.

### 9. La nuit sert à consolider et à imaginer, pas seulement à ranger

Le cycle nocturne rejoue les épisodes de la journée (rejeu contrefactuel : que se serait-il passé avec un autre module, un autre chemin), entraîne les mécanismes coûteux (recherche, pointeur, valeur), et génère des variantes plausibles non vécues (augmentation par rêve/cauchemar, validées par un discriminateur de plausibilité partagé). C'est l'équivalent fonctionnel du rejeu hippocampique pendant le sommeil, dont on sait qu'il consolide la mémoire et rejoue des trajectoires non empruntées pour en tirer un apprentissage sans avoir eu à les vivre.

### 10. L'attention est une ressource rare, allouée dynamiquement

Le nombre de modules pouvant être actifs simultanément est borné très en-dessous du nombre de modules disponibles. L'allocation entre les besoins concurrents (perception, prédiction, création d'un nouveau module, délibération) suit une répartition équitable pondérée par urgence (Weighted Fair Queueing), pas une priorité fixe. C'est le principe de ressource attentionnelle limitée de Kahneman : l'attention n'est pas extensible à volonté, elle s'arbitre.

### 11. Un seul besoin dominant, jamais un mélange

L'action du système à un instant donné est gouvernée par un seul besoin (faim, ennui...) sélectionné par arg-max avec hystérésis — pas par une moyenne pondérée continue de tous les besoins. Un animal qui a très faim ne "un peu mange, un peu explore" en proportion continue de ses besoins : un besoin prend le dessus, avec une marge qui évite les oscillations erratiques entre deux besoins proches en intensité.

### 12. Honnêteté épistémique comme discipline de conception

Chaque mécanisme de la théorie est étiqueté : pilier démontré (appuyé sur une littérature solide), hypothèse à tester (plausible, non prouvée dans ce contexte), ou décision de conception (un choix parmi plusieurs possibles, assumé comme tel). Cette discipline n'est pas cosmétique : elle empêche de traiter un pari comme une certitude, et elle dit exactement où l'expérimentation doit porter en priorité.

---

## III. Comment — la mécanique concrète

### Le rythme jour / nuit

Le système alterne deux régimes, comme un cycle veille/sommeil. Le jour : perception, action, apprentissage local incrémental à faible vitesse, accumulation d'expérience dans le tampon rapide. La nuit : pas de perception nouvelle — rejeu contrefactuel, entraînement des mécanismes globaux (orchestrateur, heuristique de recherche), tests de non-infériorité pour les décisions lourdes (verrouillage, découpe, consolidation, recalibrage de plancher), purge de ce qui est devenu obsolète, croissance dimensionnelle des modules qui le justifient.

### Le pipeline de rupture → création

1. Un module voit sa fiabilité contextuelle $\pi_i(x)$ s'effondrer sur un contexte particulier, alors que ses antécédents restent sains — le point de rupture est localisé précisément là, pas ailleurs.
2. Réparation locale tentée en premier (moins coûteuse).
3. Si la réparation échoue, recherche d'une composition d'autres modules existants qui couvre le cas.
4. Si la composition échoue aussi, un discriminateur partagé $D_\phi$ juge la plausibilité du contexte : implausible et isolé → abandon (pas de gaspillage de cycle nocturne pour un cas unique) ; implausibilité confirmée sur plusieurs occurrences distinctes (SPRT de création) → un nouveau module est créé, accompagné d'un simulateur associé qui rejoue et généralise l'épisode fondateur avant que le module ait assez d'expérience réelle pour apprendre seul.
5. Le nouveau module reste provisoire tant que sa performance n'a été confirmée que contre son propre simulateur — seule une confirmation contre de vraies occurrences ultérieures le certifie.

### Cycle de vie d'un module

Naissance (aléatoire ou héritée par découpe/fragmentation) → apprentissage local avec condensateur croissant → verrouillage progressif si la performance se stabilise (plancher) → possibilité de scission si une variable cachée sépare clairement deux régimes de performance (découpe noyau/amovible) → possibilité de fusion si plusieurs modules redondants peuvent être remplacés par un seul sans perte (consolidation) → atrophie et retrait si le module n'a jamais atteint de certitude utile.

### L'orchestrateur en pratique

À chaque pas, l'orchestrateur assemble un ensemble $T_t$ — le contexte non pointable (condensateurs, fiabilités, besoins) et les éléments pointables (capteurs, latents de modules, trace de la décision précédente). Un Set Transformer encode cet ensemble ; un Pointer Network décode un triplet (source, opérateur, cible), masqué pour respecter la compatibilité de type. L'exécution se fait par lots de largeur bornée (macro-pas), jamais un module à la fois — c'est ce qui rend l'allocation d'attention nécessaire.

### Comment le crédit se répartit

Quand une composition échoue ou réussit, il faut savoir si la faute revient au module choisi ou au choix de l'orchestrateur de le solliciter à cet endroit. Le régret de composition — l'écart entre la perte du module choisi et la meilleure perte parmi les candidats disponibles, mesuré par rejeu contrefactuel — sert de signal d'apprentissage à l'orchestrateur (REINFORCE), séparément de l'apprentissage local de chaque module.

### Ce qu'on cherche à observer

Le monde-jouet (perception visuelle grossière, faim, douleur, sucre, bâton) n'est qu'un support. Ce qu'on veut voir apparaître, ce sont des paliers d'émergence : d'abord une reconnaissance fiable des stimuli de base, puis une prédiction à courte échéance, puis une composition de modules pour des cas non directement appris, puis une création structurelle spontanée face à une vraie nouveauté, puis une consolidation qui simplifie le graphe sans perte. Si ces paliers apparaissent dans cet ordre, sans réentraînement global et sans effondrement de ce qui a été appris avant, l'hypothèse centrale de SCL est soutenue.

---

## En une phrase

SCL parie que l'intelligence continue ne se construit pas en agrandissant un réseau figé, mais en laissant la structure elle-même être une trace de l'expérience — locale, vérifiable, révisable, jamais globalement réécrite.
