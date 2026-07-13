#!/usr/bin/env python3
"""Serveur du dashboard SCL temps réel.

Suit scl_audit.jsonl pendant que run_poc.py tourne et sert viewer.html.

Usage (dans un 2e terminal, pendant le run) :
    python3 viewer.py --log scl_audit.jsonl --port 8400
puis ouvrir http://localhost:8400 dans le navigateur.
"""
import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

# Actions pertinentes pour le dashboard (le reste du log est trop verbeux).
# Vocabulaire v6 (scl/*.py, réécriture Phases 0-11) — l'ancien vocabulaire
# F1-F25 (choix_mode, chantier_*, barreaux, consolidation_hebbienne...)
# n'existe plus, les mécanismes correspondants ont été remplacés.
ACTIONS_UTILES = {
    # cycle de vie des modules
    "creation_module", "creation_candidat", "creation_simulateur",
    "initialisation_episode_fondateur", "copie_module", "retrait_module",
    "atrophie_abandon", "confirmation_reelle",
    # apprentissage local
    "entrainement_reco", "entrainement_gen", "entrainement_masque",
    "entrainement_proprio", "entrainement_action", "entrainement_simulateur",
    "condensateurs", "verrouillage", "disponibilite_anticipee",
    "logique_acceptation", "croissance", "croissance_gouvernee",
    "saturation_detectee",
    # structure du graphe
    "ajout_module", "localiser_point_branchement", "test_non_inferiorite",
    "fragmentation", "decoupe", "decoupe_impossible", "valider_decoupe",
    "committer_chemin", "recalage_plancher_drift", "controle_multiplicite",
    "rejet_gouverne",
    # statistiques / SPRT
    "sprt_surprise", "sprt_creation", "sprt_drift", "surprise_validee",
    # attention / orchestrateur (Set Transformer + Pointer Network)
    "construire_T_t", "macro_pas", "executer_triplet", "entrainement_pointeurs",
    "accumulateur_orchestrateur", "role_creation", "allouer_capacite",
    # décision / action
    "priorisation_besoin_dominant", "action_par_reflexe", "declenchement",
    "besoin_dominant_change", "changement_etat",
    # mémoire, crédit, recherche
    "amorcage_creation", "regret_composition", "rejeu_contrefactuel_nocturne",
    "maturation_prediction", "generer_contrefactuel", "est_hors_distribution",
    "recuperation", "recuperation_refusee", "a_etoile_trouve", "a_etoile_echec",
    "ancrer_composition", "ancrer_composition_refusee",
    # nuit
    "reve_coordonne", "cycle_nocturne_termine", "entrainement_contrastif",
    "entrainement_v_psi",
    # boucle / monde / persistance
    "action_appliquee", "resume_journee", "sauvegarde", "chargement",
    "creation_refusee", "creation_refusee_cooldown", "abandon_marque",
}
MAX_LIGNES_PAR_REQUETE = 8000


class Handler(BaseHTTPRequestHandler):
    chemin_log = "scl_audit.jsonl"
    chemin_html = "viewer.html"

    def log_message(self, *a):  # silence du serveur
        pass

    def _repondre(self, code, contenu, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(contenu)))
        self.end_headers()
        self.wfile.write(contenu)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            try:
                with open(self.chemin_html, "rb") as f:
                    self._repondre(200, f.read(), "text/html")
            except FileNotFoundError:
                self._repondre(404, b"viewer.html introuvable", "text/plain")
        elif url.path == "/data":
            offset = int(parse_qs(url.query).get("offset", ["0"])[0])
            self._repondre(200, json.dumps(self._lire(offset),
                                           ensure_ascii=False).encode("utf-8"))
        else:
            self._repondre(404, b"?", "text/plain")

    def _lire(self, offset):
        """Lit les nouvelles lignes complètes du JSONL depuis offset (octets)."""
        lignes = []
        try:
            taille = os.path.getsize(self.chemin_log)
        except OSError:
            return {"offset": 0, "lines": [], "eof": True}
        if offset > taille:   # fichier recréé (nouveau run) : repartir de zéro
            offset = 0
        with open(self.chemin_log, "rb") as f:
            f.seek(offset)
            data = f.read()
        fin_derniere_ligne = data.rfind(b"\n")
        if fin_derniere_ligne == -1:
            return {"offset": offset, "lines": [], "eof": True}
        data = data[: fin_derniere_ligne + 1]
        nouveau_offset = offset + len(data)
        for brut in data.split(b"\n"):
            if not brut.strip():
                continue
            try:
                # parse_constant : NaN/Infinity (tolérés par Python, invalides
                # en JSON standard) deviennent null au lieu de tuer JSON.parse
                # côté navigateur — robustesse aux anciens logs
                rec = json.loads(brut, parse_constant=lambda _: None)
            except json.JSONDecodeError:
                continue
            if rec.get("action") in ACTIONS_UTILES:
                lignes.append(rec)
            if len(lignes) >= MAX_LIGNES_PAR_REQUETE:
                break
        return {"offset": nouveau_offset, "lines": lignes,
                "eof": nouveau_offset >= taille}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="scl_audit.jsonl")
    p.add_argument("--port", type=int, default=8400)
    args = p.parse_args()
    Handler.chemin_log = args.log
    Handler.chemin_html = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "viewer.html")
    serveur = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Dashboard SCL : http://localhost:{args.port}  (log : {args.log})")
    try:
        serveur.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
