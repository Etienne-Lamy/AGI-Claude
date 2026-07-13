"""Graphe inné SCL (§1.3, bootstrap) — 'à la naissance, le cerveau n'est pas
vide'. Structure de départ minimale : moteur, réflexe câblé, module visuel
par défaut (CNN, `module_visuel.py` — remplace l'ancien `module_slots.py`,
documenté comme échec dans README v2.md), discriminateur partagé D_φ
instancié UNE SEULE FOIS ici et nulle part ailleurs (§0 : jamais un
discriminateur par module).

Chemin unique : l'ancien fork `CONFIG["pilotage_chantiers"]` (pilote de
chantiers vs structurel) est abandonné — un seul graphe de naissance,
cohérent avec le reste de la réécriture v6."""
from .config import CONFIG
from .discriminateur import Discriminateur
from .graphe import Graphe
from .logger import log
from .module import Module
from .module_visuel import ModuleVisuel

DIM_PROPRIO = 4


def construire_graphe_inne():
    """Retourne (graphe, discriminateur) — le graphe de naissance et
    l'instance UNIQUE de D_φ, à transporter explicitement partout où elle
    est nécessaire (jamais recréée)."""
    graphe = Graphe()
    discriminateur = Discriminateur()

    vision = ModuleVisuel("vision", innate=True)
    proprio = Module("proprio", n_inputs_reco=DIM_PROPRIO, n_latent=8, innate=True)
    integration = Module("integration",
                         n_inputs_reco=vision.n_latent + proprio.n_latent, n_latent=12,
                         n_hidden_reco=16, n_hidden_gen=16, innate=True)
    action = Module("action", n_inputs_reco=12, n_latent=6,
                    n_outputs_gen=2, latent_input_dim=6,
                    innate=True, is_action=True)
    reflexe = Module("reflexe_frein", n_inputs_reco=DIM_PROPRIO + 1, n_latent=2,
                     n_outputs_gen=2, innate=True)
    # le réflexe est câblé en dur et verrouillé dès la naissance
    reflexe.locked_reco = True
    reflexe.locked_gen = True
    reflexe.condensateur_reco = 1.0
    reflexe.condensateur_gen = 1.0
    reflexe.status = "verrouillé"

    graphe.ajouter_module(vision, input_node=True)
    graphe.ajouter_module(proprio, input_node=True)
    graphe.ajouter_module(integration, parents=["vision", "proprio"])
    graphe.ajouter_module(action, parents=["integration"], output_node=True)
    graphe.ajouter_module(reflexe, input_node=True, output_node=True)

    log("inne", "graphe_inne_construit", modules=list(graphe.modules),
        edges=graphe.edges)
    return graphe, discriminateur


def reflexe_frein(vitesse, douleur):
    """Réflexe de survie câblé (§15.3, garde-fou) : si la douleur dépasse le
    seuil, freiner (décélère la composante de vitesse dominante). Jamais
    appris, jamais atrophié. Retourne None si non déclenché."""
    if douleur <= CONFIG["seuil_reflexe_douleur"]:
        return None
    vx, vy = float(vitesse[0]), float(vitesse[1])
    if vx == 0 and vy == 0:
        commande = (0, 0)
    elif abs(vx) >= abs(vy):
        commande = (-1 if vx > 0 else 1, 0)
    else:
        commande = (0, -1 if vy > 0 else 1)
    log("reflexe_frein", "declenchement", douleur=douleur, vitesse=[vx, vy],
        commande=commande)
    return commande
