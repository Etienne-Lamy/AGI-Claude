"""Orchestrateur NAÏF — première version programmée à la main de l'action
« créer un module sur une entrée incertaine » (§5, parcimonie/MDL).

Au lieu de FIXER la dimension du réseau à la main (ce que je faisais), on
propose un CATALOGUE de tailles de goulot ; on entraîne un module candidat pour
chacune ; on sélectionne celui qui minimise la LONGUEUR DE DESCRIPTION totale :

    MDL(dim) = L(code)      +  L(données | modèle)
             = dim × bits_par_dim  +  résidu de reconstruction en bits/champ

Petit goulot → code bon marché mais résidu élevé ; grand goulot → résidu faible
mais code cher. Le minimum est la taille qui EXTRAIT l'information à valeur sans
gaspiller de capacité sur du signal brut. La taille INTERNE n'est pas pénalisée
(le réseau peut être gros au milieu) — seul le goulot (E/S du code) compte.

Plus tard : l'orchestrateur apprendra par renforcement QUELLE dimension choisir
selon le contexte, au lieu de toutes les essayer.
"""
from .config import CONFIG
from .logger import log
from .module_ae import ModuleAutoencodeur


def essayer_catalogue(flux_train_fn, champs_eval, catalogue=None, pas=1500,
                      bits_par_dim=None, resolution=None):
    """Essaie chaque dimension du catalogue et renvoie (meilleur, resultats).

    `flux_train_fn(pas)` : itérable de `pas` champs d'entraînement, RÉGÉNÉRÉ à
    chaque dimension (comparaison équitable). `champs_eval` : jeu FIXE tenu à
    l'écart. `meilleur` : le dict de plus petit MDL (contient le module gardé)."""
    catalogue = catalogue or CONFIG["catalogue_dims_module"]
    bits_par_dim = bits_par_dim if bits_par_dim is not None else CONFIG["bits_par_dim_mdl"]
    resultats = []
    for dim in catalogue:
        ae = ModuleAutoencodeur(f"cand_dim{dim}", dim_latent=dim, resolution=resolution)
        for champ in flux_train_fn(pas):
            ae.entrainer(champ)
        residuel = ae.cout_residuel_bits(champs_eval)        # L(données|modèle), bits/champ
        code = dim * bits_par_dim                             # L(code)
        mdl = code + residuel
        rappel = sum(ae.fidelite(f)["rappel"] for f in champs_eval) / len(champs_eval)
        resultats.append({"dim": dim, "mdl": round(mdl, 1), "code": round(code, 1),
                          "residuel": round(residuel, 1), "rappel": round(rappel, 3),
                          "module": ae})
        log("orchestrateur_naif", "essai_dimension", dim=dim, mdl=round(mdl, 1),
            residuel=round(residuel, 1), rappel=round(rappel, 3))
    meilleur = min(resultats, key=lambda r: r["mdl"])
    log("orchestrateur_naif", "dimension_choisie", dim=meilleur["dim"], mdl=meilleur["mdl"])
    return meilleur, resultats
