"""Construction, selection et publication du banc externe de positions.

Ce module contient le noyau pur : selection par quotas (plus grand deficit
relatif, plafonds evenement et joueur, equilibre des couleurs), sous-banc de
recherche par plus grands restes, echographie d'audit, ecriture canonique
JSONL/zstd et manifeste. La CLI orchestre les cinq etapes : extract, screen,
annotate, finalize, build.

Regles de la conception (sections 5 et 6) :

- 10 000 positions, une seule par partie, neuf quotas phase x categorie WDL ;
- au moins 45 % de positions avec les blancs au trait, au plus 55 % ;
- au plus 100 positions par evenement, 20 apparitions par joueur, toutes
  couleurs confondues ;
- sous-banc MCTS de 256 identifiants, proportions arrondies par plus grands
  restes, hash independant ;
- dataset v1 immuable : une version differente ne peut pas ecraser la
  precedente ; le JSONL est trie, compresse a parametres fixes, et le
  manifeste est publie en dernier.

Voir docs/superpowers/specs/2026-09-29-position-bench-design.md (sections 4
a 6) et docs/superpowers/plans/2026-09-30-position-bench.md (tache 5).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import chess.engine
import zstandard as zstd
from position_bench_metrics import (
    SearchConfig,
    SourceInfo,
    validate_manifest,
    wdl_category,
)
from position_bench_sources import (
    ATTRIBUTION_BROADCASTS,
    LICENCE_BROADCASTS,
    barre_progression,
    dedupliquer_candidats,
    download_archive,
    extract_candidates,
    read_games,
)
from position_bench_teacher import (
    NODES_ANNOTATION,
    NODES_AUDIT,
    NODES_CRIBLAGE,
    SCHEMA_STORE,
    AnnotationStore,
    annotate_position,
    audit_labels,
    cle_criblage,
    cle_position,
    configurer_moteur,
    identifier_stockfish,
    screen_position,
)

TAILLE_BANC = 10_000
TAILLE_SOUS_BANC = 256
FACTEUR_RESERVE = 1.25
EVENT_CAP = 100
PLAYER_CAP = 20
AUDIT_COUNT = 500
ZSTD_NIVEAU = 19
BUILDER_REVISION = "lapzero-position-bench-builder-v1"

PHASES = ("ouverture", "milieu", "finale")
BUCKETS = ("disputee", "avantage", "decisive")
CELLULES = tuple((phase, bucket) for phase in PHASES for bucket in BUCKETS)

QUOTAS = {
    ("ouverture", "disputee"): 1000,
    ("ouverture", "avantage"): 600,
    ("ouverture", "decisive"): 400,
    ("milieu", "disputee"): 3000,
    ("milieu", "avantage"): 1800,
    ("milieu", "decisive"): 1200,
    ("finale", "disputee"): 1000,
    ("finale", "avantage"): 600,
    ("finale", "decisive"): 400,
}

SEL_FINAL = "lapzero-position-bench-v1-final"
SEL_RESERVE = "lapzero-position-bench-v1-reserve"
SEL_RECHERCHE = "lapzero-position-bench-v1-search"
SEL_AUDIT = "lapzero-position-bench-v1-audit"
SEL_PILOTE = "lapzero-position-bench-v1-pilot"

URL_BROADCASTS = ("https://database.lichess.org/broadcast/"
                  "lichess_db_broadcast_{mois}.pgn.zst")


class SelectionIncomplete(RuntimeError):
    """Aucune selection complete ne peut etre formee avec ce vivier."""


class AuditEchoue(RuntimeError):
    """L'audit de stabilite des etiquettes a echoue ; v1 n'est pas publie."""


def _hash_cle(position_id: str, sel: str) -> str:
    return hashlib.sha256(f"{position_id}|{sel}".encode()).hexdigest()


def select_positions(records, quotas, *, salt: str, event_cap: int,
                     player_cap: int, rapport: dict | None = None) -> list:
    """Selection gloutonne par plus grand deficit relatif.

    A chaque tour, la case dont le deficit relatif est le plus grand recoit le
    premier candidat admissible dans son ordre de hash. Un candidat est refuse
    s'il fait depasser un plafond d'evenement ou de joueur, s'il apporte une
    seconde position de la meme partie, ou s'il fait depasser 55 % du total
    vise a une couleur. Le rapport distingue une selection complete d'une
    selection ou il manque des positions : aucun filtre n'est relache en
    silence.
    """
    par_case = {}
    for record in records:
        cle = (record["phase"], record["wdl_bucket"])
        if cle in quotas:
            par_case.setdefault(cle, []).append(record)
    for liste in par_case.values():
        liste.sort(key=lambda r: _hash_cle(r["position_id"], salt))

    total_vise = sum(quotas.values())
    curseurs = {cle: 0 for cle in quotas}
    epuisees: set = set()
    comptes = {cle: 0 for cle in quotas}
    par_evenement: dict[str, int] = {}
    par_joueur: dict[str, int] = {}
    par_couleur = {0: 0, 1: 0}
    jeux_utilises: set = set()
    choisis = []

    def reste_relatif(cle):
        if cle in epuisees or quotas[cle] == 0:
            return -1.0
        return (quotas[cle] - comptes[cle]) / quotas[cle]

    while len(choisis) < total_vise:
        cle = max(CELLULES, key=reste_relatif)
        if reste_relatif(cle) < 0.0:
            break

        retenu = None
        while curseurs[cle] < len(par_case.get(cle, [])):
            candidat = par_case[cle][curseurs[cle]]
            curseurs[cle] += 1
            if candidat["game_fingerprint"] in jeux_utilises:
                continue
            if par_evenement.get(candidat["event_id"], 0) >= event_cap:
                continue
            if par_joueur.get(candidat["white_id"], 0) >= player_cap:
                continue
            if par_joueur.get(candidat["black_id"], 0) >= player_cap:
                continue
            couleur = candidat["turn"]
            if par_couleur[couleur] + 1 > 0.55 * total_vise:
                continue
            retenu = candidat
            break
        if retenu is None:
            epuisees.add(cle)
            continue

        choisis.append(retenu)
        comptes[cle] += 1
        jeux_utilises.add(retenu["game_fingerprint"])
        par_evenement[retenu["event_id"]] = (
            par_evenement.get(retenu["event_id"], 0) + 1)
        par_joueur[retenu["white_id"]] = (
            par_joueur.get(retenu["white_id"], 0) + 1)
        par_joueur[retenu["black_id"]] = (
            par_joueur.get(retenu["black_id"], 0) + 1)
        par_couleur[retenu["turn"]] += 1

    manquants = {
        f"{phase}/{bucket}": quotas[(phase, bucket)] - comptes[(phase, bucket)]
        for phase, bucket in CELLULES
        if quotas[(phase, bucket)] > comptes[(phase, bucket)]
    }
    total = len(choisis)
    couleur_ok = bool(total) and all(
        par_couleur[couleur] >= 0.45 * total for couleur in (0, 1))
    if rapport is not None:
        rapport["complet"] = not manquants and couleur_ok
        rapport["manquants"] = manquants
        rapport["couleurs"] = {"blancs": par_couleur[0], "noirs": par_couleur[1]}
        rapport["couleur_ok"] = couleur_ok
    return choisis


def quotas_sous_banc() -> dict:
    """Neuf quotas arrondis a 256 par plus grands restes, ordre des cases."""
    total = sum(QUOTAS.values())
    exacts = {cle: quota * TAILLE_SOUS_BANC / total
              for cle, quota in QUOTAS.items()}
    arrondis = {cle: int(valeur) for cle, valeur in exacts.items()}
    reste = TAILLE_SOUS_BANC - sum(arrondis.values())
    ordre = sorted(
        CELLULES,
        key=lambda cle: (-(exacts[cle] - arrondis[cle]), CELLULES.index(cle)))
    for cle in ordre[:reste]:
        arrondis[cle] += 1
    return arrondis


def select_search_ids(records) -> list[str]:
    """256 identifiants du sous-banc, par cellule puis hash independant."""
    quotas = quotas_sous_banc()
    par_case = {}
    for record in records:
        cle = (record["phase"], record["wdl_bucket"])
        par_case.setdefault(cle, []).append(record)
    resultat = []
    for cle in CELLULES:
        liste = sorted(
            par_case.get(cle, []),
            key=lambda r: _hash_cle(r["position_id"], SEL_RECHERCHE))
        if len(liste) < quotas[cle]:
            raise SelectionIncomplete(
                f"sous-banc : {cle[0]}/{cle[1]} a {len(liste)} candidats "
                f"pour {quotas[cle]}")
        resultat.extend(r["position_id"] for r in liste[:quotas[cle]])
    return resultat


def select_audit_ids(records, count: int = AUDIT_COUNT) -> list[str]:
    """Positions de l'echantillon d'audit, par hash independant."""
    ordonnes = sorted(records,
                      key=lambda r: _hash_cle(r["position_id"], SEL_AUDIT))
    return [r["position_id"] for r in ordonnes[:count]]


def finaliser(selection, deeper, *,
              audit_count: int = AUDIT_COUNT) -> tuple[list, dict]:
    """Sous-banc et audit de la selection ; leve si l'audit echoue.

    `deeper` contient les memes positions annotees au budget d'audit.
    """
    search_ids = select_search_ids(selection)
    ids_audit = select_audit_ids(selection, audit_count)
    # La base garde ses scores de production, la passe profonde les siens :
    # le controle croise les deux jeux d'etiquettes position par position.
    base = {position["position_id"]: position for position in selection}
    profonde = {position["position_id"]: position for position in deeper}
    audit = audit_labels([base[i] for i in ids_audit],
                         [profonde[i] for i in ids_audit])
    if not audit["passed"]:
        raise AuditEchoue(json.dumps(audit, sort_keys=True))
    return search_ids, audit


def assembler_manifeste(*, dataset_version: str, sources, selection,
                        search_ids, stockfish: dict, audit: dict,
                        quotas=None, created: str | None = None) -> dict:
    """Manifeste de publication ; le hash du dataset est pose a l'ecriture."""
    return {
        "schema_version": 1,
        "dataset_version": dataset_version,
        "dataset_sha256": "0" * 64,
        "training_forbidden": True,
        "sources": [dict(source) for source in sources],
        "counts": {"positions": len(selection),
                   "search": len(search_ids)},
        "quotas": _quotas_json(QUOTAS if quotas is None else quotas),
        "stockfish": stockfish,
        "audit": audit,
        "search_ids": list(search_ids),
        "protocol_defaults": {
            "search": asdict(SearchConfig()),
            "policy_batch_size": 1024,
            "target_s": 300.0,
        },
        "builder_revision": BUILDER_REVISION,
        "created": created or datetime.now(UTC).isoformat(),
    }


def _quotas_json(quotas) -> dict:
    return {f"{phase}/{bucket}": quotas[(phase, bucket)]
            for phase, bucket in CELLULES}


def est_productif(manifeste) -> bool:
    """Vrai seulement pour un banc complet, donc publiable et entrainable."""
    counts = manifeste.get("counts", {})
    return (counts.get("positions") == TAILLE_BANC
            and counts.get("search") == TAILLE_SOUS_BANC)


def _jsonl_compresse(records) -> bytes:
    lignes = [
        json.dumps(record, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False)
        for record in sorted(records, key=lambda r: r["position_id"])
    ]
    charge = ("\n".join(lignes) + "\n").encode("utf-8")
    return zstd.ZstdCompressor(level=ZSTD_NIVEAU).compress(charge)


def _ecrire_atomique(chemin: Path, donnees: bytes) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    temporaire = chemin.with_name(chemin.name + ".tmp")
    with open(temporaire, "wb") as flux:
        flux.write(donnees)
    os.replace(temporaire, chemin)


def ecrire_jsonl_zst(records, chemin) -> None:
    _ecrire_atomique(Path(chemin), _jsonl_compresse(records))


def lire_jsonl_zst(chemin) -> list:
    decompresseur = zstd.ZstdDecompressor()
    with open(chemin, "rb") as brut, io.TextIOWrapper(
            decompresseur.stream_reader(brut), encoding="utf-8") as flux:
        return [json.loads(ligne) for ligne in flux if ligne.strip()]


def _readme(manifeste) -> str:
    sources = ", ".join(source["source_id"] for source in manifeste["sources"])
    return (
        "# Banc externe de positions, "
        f"{manifeste['dataset_version']}\n\n"
        "Positions extraites des archives de broadcasts Lichess "
        "(CC BY-SA 4.0, attribution Lichess), annotees par Stockfish hors "
        "ligne. Ce banc est interdit a l'entrainement : "
        "`training_forbidden` vaut true dans le manifeste, et aucun chargeur "
        "de donnees ne le decouvre automatiquement.\n\n"
        f"- Sources : {sources}\n"
        f"- Positions : {manifeste['counts']['positions']}\n"
        f"- Identifiants de recherche : {manifeste['counts']['search']}\n"
        f"- SHA-256 du JSONL compresse : {manifeste['dataset_sha256']}\n"
        f"- Moteur d'annotation : {manifeste['stockfish'].get('name', '?')}\n\n"
        "Les identifiants de recherche sont la liste explicite du manifeste ; "
        "l'ordre des positions du fichier est celui des `position_id` "
        "croissants.\n"
    )


def write_dataset(records, manifest, output) -> Path:
    """Ecrit le JSONL, le README puis le manifeste, dans cet ordre.

    Une version existante avec le meme hash est rendue telle quelle ; une
    version differente est refusee, jamais ecrasee.
    """
    dossier = Path(output)
    compresse = _jsonl_compresse(records)
    digest = hashlib.sha256(compresse).hexdigest()
    manifeste = dict(manifest)
    manifeste["dataset_sha256"] = digest
    validate_manifest(manifeste)

    chemin_manifeste = dossier / "manifest.json"
    if chemin_manifeste.is_file():
        existant = json.loads(chemin_manifeste.read_text(encoding="utf-8"))
        if existant.get("dataset_sha256") != digest:
            raise FileExistsError(
                f"une version differente existe deja : {dossier}")
        return chemin_manifeste

    _ecrire_atomique(dossier / "positions.jsonl.zst", compresse)
    _ecrire_atomique(dossier / "README.md", _readme(manifeste).encode("utf-8"))
    _ecrire_atomique(chemin_manifeste, json.dumps(
        manifeste, sort_keys=True, indent=2, ensure_ascii=False).encode(
            "utf-8"))
    return chemin_manifeste


# ============================================================
#                            CLI
# ============================================================

def _parseur() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="build_position_bench")
    parser.add_argument(
        "commande",
        choices=["extract", "screen", "annotate", "finalize", "build"])
    parser.add_argument("--work-dir", default="data/position_bench_work")
    parser.add_argument("--output", default="data/position_bench/v1")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--months", nargs="+",
                        default=["2026-08", "2026-07", "2026-06"])
    parser.add_argument("--stockfish")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pilot-count", type=int, default=0)
    parser.add_argument("--stage", choices=["reserve", "final"], default="final")
    return parser


def main(argv=None) -> int:
    args = _parseur().parse_args(argv)
    try:
        if args.commande == "extract":
            _cmd_extract(args)
        elif args.commande == "screen":
            _cmd_screen(args)
        elif args.commande == "annotate":
            _cmd_annotate(args)
        elif args.commande == "finalize":
            _cmd_finalize(args)
        else:
            _cmd_build(args)
    except (ValueError, RuntimeError, OSError) as erreur:
        print(f"erreur : {erreur}", file=sys.stderr)
        return 1
    return 0


def _cmd_build(args) -> None:
    _cmd_extract(args)
    _cmd_screen(args)
    args.stage = "reserve"
    _cmd_finalize(args)
    _cmd_annotate(args)
    args.stage = "final"
    _cmd_finalize(args)


def _cmd_extract(args) -> None:
    dossier = Path(args.work_dir)
    archives = dossier / "archives"
    archives.mkdir(parents=True, exist_ok=True)
    sources = []
    candidats = []
    for mois in args.months:
        url = URL_BROADCASTS.format(mois=mois)
        destination = archives / f"lichess_db_broadcast_{mois}.pgn.zst"
        if args.resume and destination.is_file():
            source = _source_reprise(destination, url, mois)
            print(f"  archive {mois} deja presente, telechargement ignore")
        else:
            source = download_archive(
                url, destination, None,
                progress=barre_progression(f"{mois}"))
            print(f"\n  archive {mois} : "
                  f"{destination.stat().st_size / 1e6:.1f} Mo, sha256 "
                  f"{source['archive_sha256'][:16]}...")
        sources.append(dict(source))

        # Un fichier de candidats par mois : une interruption ne perd que le
        # mois en cours, et une reprise reutilise les mois deja extraits.
        chemin_candidats = dossier / f"candidates_{mois}.jsonl.zst"
        if args.resume and chemin_candidats.is_file():
            extraits = lire_jsonl_zst(chemin_candidats)
            print(f"  candidats {mois} deja extraits : {len(extraits)}, "
                  "reutilises")
        else:
            compteurs: dict = {}
            extraits = extract_candidates(
                read_games(destination, source), source,
                compteurs=compteurs,
                progress=_barre_extraction(
                    mois, dossier / f"{mois}.extraction.progress"))
            ecrire_jsonl_zst(extraits, chemin_candidats)
            print(f"\n  {mois} : {len(extraits)} candidats, rejets "
                  f"{compteurs}")
        candidats.extend(extraits)

    candidats = dedupliquer_candidats(candidats)
    ecrire_jsonl_zst(candidats, dossier / "candidates.jsonl.zst")
    _ecrire_atomique(
        dossier / "sources.json",
        json.dumps(sources, sort_keys=True, indent=2).encode("utf-8"))
    print(f"{len(candidats)} candidats extraits de {len(sources)} archives")


def _barre_extraction(mois: str, fichier=None):
    """Etat d'extraction en continu, sur stderr et dans un fichier de suivi."""
    def suivi(etat: dict) -> None:
        ligne = (f"{mois} : {etat['parties_lues']} parties lues, "
                 f"{etat['candidats']} candidats, {etat['rejets']} rejets")
        print(f"\r  {ligne}", end="", file=sys.stderr, flush=True)
        if fichier is not None:
            Path(fichier).write_text(ligne + "\n", encoding="utf-8")

    return suivi


def _source_reprise(destination: Path, url: str, mois: str) -> SourceInfo:
    digest = hashlib.sha256()
    with open(destination, "rb") as flux:
        for bloc in iter(lambda: flux.read(1 << 20), b""):
            digest.update(bloc)
    return SourceInfo(
        source_id=f"lichess-broadcasts-{mois}",
        url=url,
        month=mois,
        archive_sha256=digest.hexdigest(),
        license=LICENCE_BROADCASTS,
        attribution=ATTRIBUTION_BROADCASTS,
    )


def _config_store(identite) -> dict:
    """Configuration figee du cache ; tous les moteurs doivent etre identiques."""
    return {
        "schema": SCHEMA_STORE,
        "engine": identite,
        "budgets": {
            "screen": NODES_CRIBLAGE,
            "annotation": NODES_ANNOTATION,
            "audit": NODES_AUDIT,
        },
    }


def _ouvrir_stockfish(args):
    if not args.stockfish:
        raise ValueError("--stockfish est obligatoire pour cette commande")
    moteur = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    configurer_moteur(moteur)
    identite = identifier_stockfish(
        moteur, args.stockfish,
        dossier_export=Path(args.work_dir) / "nnue")
    return moteur, identite


def _ouvrir_moteurs(args):
    """Ouvre `--workers` moteurs identiques et rend leur identite commune."""
    moteurs = []
    identite = None
    for _ in range(max(1, args.workers)):
        moteur, identite_moteur = _ouvrir_stockfish(args)
        if identite is None:
            identite = identite_moteur
        elif identite_moteur != identite:
            raise RuntimeError(
                "les workers n'utilisent pas le meme Stockfish")
        moteurs.append(moteur)
    return moteurs, identite


def _en_parallele(moteurs, elements, traitement, *, a_reception=None,
                  progress=None):
    """Applique un traitement par tache, chaque moteur occupe une tache a la fois.

    Les moteurs sont des processus Stockfish independants : la tache d'ordre
    `i` prend le moteur `i % len(moteurs)` sous son verrou. Le resultat est
    rendu des qu'il arrive via `a_reception(element, resultat)`, puis
    `progress({"faites", "total"})` est appele. L'ordre de soumission, et donc
    la repartition, ne change pas les resultats : chaque tache est independante.
    """
    resultats = [None] * len(elements)
    if not elements:
        return resultats
    verrous = [threading.Lock() for _ in moteurs]

    def tache(ordre, element):
        index = ordre % len(moteurs)
        with verrous[index]:
            return traitement(moteurs[index], element)

    with ThreadPoolExecutor(max_workers=len(moteurs)) as executeur:
        futurs = {executeur.submit(tache, ordre, element): (ordre, element)
                  for ordre, element in enumerate(elements)}
        for faites, futur in enumerate(as_completed(futurs), start=1):
            ordre, element = futurs[futur]
            resultat = futur.result()
            resultats[ordre] = resultat
            if a_reception is not None:
                a_reception(element, resultat)
            if progress is not None:
                progress({"faites": faites, "total": len(elements)})
    return resultats


def _cmd_screen(args) -> None:
    dossier = Path(args.work_dir)
    candidats = lire_jsonl_zst(dossier / "candidates.jsonl.zst")
    moteurs, identite = _ouvrir_moteurs(args)
    store = AnnotationStore(dossier, _config_store(identite))
    try:
        a_cribler = [position for position in candidats
                     if store.get(cle_criblage(position)) is None]
        _en_parallele(
            moteurs, a_cribler,
            lambda moteur, position: screen_position(moteur, position),
            a_reception=lambda position, note: store.put(
                cle_criblage(position), note),
            progress=_barre_suivi("criblage",
                                  fichier=dossier / "criblage.progress"))
        enrichis = [
            {**candidat,
             "screen_score": store.get(cle_criblage(candidat))}
            for candidat in candidats
        ]
    finally:
        store.close()
        for moteur in moteurs:
            moteur.quit()
    ecrire_jsonl_zst(enrichis, dossier / "screened.jsonl.zst")
    print(f"\n{len(enrichis)} positions criblees")


def _cmd_annotate(args) -> None:
    dossier = Path(args.work_dir)
    if args.pilot_count:
        candidats = lire_jsonl_zst(dossier / "candidates.jsonl.zst")
        positions = sorted(
            candidats,
            key=lambda c: _hash_cle(c["position_id"], SEL_PILOTE)
        )[:args.pilot_count]
        dossier = dossier / "pilot"
    else:
        reserve = json.loads(
            (dossier / "reserve.json").read_text(encoding="utf-8"))
        par_id = {c["position_id"]: c
                  for c in lire_jsonl_zst(dossier / "screened.jsonl.zst")}
        positions = [par_id[i] for i in reserve["ids"]]

    moteurs, identite = _ouvrir_moteurs(args)
    store = AnnotationStore(dossier, _config_store(identite))
    try:
        annoter_positions(
            moteurs, positions, NODES_ANNOTATION, store,
            dossier / "annotated.jsonl.zst",
            progress=_barre_suivi(
                "pilot" if args.pilot_count else "annotation",
                fichier=dossier / "annotation.progress"))
    finally:
        store.close()
        for moteur in moteurs:
            moteur.quit()
    print(f"\n{len(positions)} positions annotees")


def _barre_suivi(nom: str, *, reprises: int = 0, fichier=None):
    """Etat de progression en continu, sur stderr et dans un fichier de suivi."""
    def suivi(etat: dict) -> None:
        complement = (f" ({reprises} reprises du cache)" if reprises else "")
        ligne = (f"{nom} : {etat['faites']}/{etat['total']} positions"
                 f"{complement}")
        print(f"\r  {ligne}", end="", file=sys.stderr, flush=True)
        if fichier is not None:
            Path(fichier).write_text(ligne + "\n", encoding="utf-8")

    return suivi


def annoter_positions(moteurs, positions, nodes: int, store, sortie, *,
                      progress=None) -> list:
    """Annote les positions en parallele et materialise le resultat.

    Chaque position est analysee seule, dans un moteur configure de facon
    identique : le resultat ne depend donc ni du nombre de moteurs ni de la
    repartition. Le cache SQLite est ecrit par le coordinateur, jamais par les
    travailleurs. `progress` est appele apres chaque position nouvellement
    annotee, pour suivre une campagne longue.
    """
    resultats: list = [None] * len(positions)
    for rang, position in enumerate(positions):
        enregistre = store.get(cle_position(position, nodes))
        if enregistre is not None:
            resultats[rang] = enregistre
    a_faire = [rang for rang, resultat in enumerate(resultats)
               if resultat is None]
    deja = len(positions) - len(a_faire)

    def reception(rang: int, annotee) -> None:
        store.put(cle_position(positions[rang], nodes), annotee)
        resultats[rang] = annotee

    def suivi(etat: dict) -> None:
        if progress is None:
            return
        progress({"faites": etat["faites"], "total": etat["total"],
                  "reprises": deja})

    _en_parallele(
        moteurs, a_faire,
        lambda moteur, rang: annotate_position(moteur, positions[rang], nodes),
        a_reception=reception,
        progress=suivi if progress is not None else None)
    ecrire_jsonl_zst(resultats, sortie)
    return resultats


def _cmd_finalize(args) -> None:
    dossier = Path(args.work_dir)
    if args.stage == "reserve":
        _finalize_reserve(dossier)
    else:
        _finalize_final(args, dossier)


def _finalize_reserve(dossier: Path) -> None:
    enrichis = lire_jsonl_zst(dossier / "screened.jsonl.zst")
    for position in enrichis:
        position["wdl_bucket"] = wdl_category(float(position["screen_score"]))
    quotas = {cle: round(quota * FACTEUR_RESERVE)
              for cle, quota in QUOTAS.items()}
    rapport: dict = {}
    reserve = select_positions(enrichis, quotas, salt=SEL_RESERVE,
                               event_cap=EVENT_CAP, player_cap=PLAYER_CAP,
                               rapport=rapport)
    if not rapport["complet"]:
        raise SelectionIncomplete(
            "reserve incomplete, ajouter le mois precedent : "
            + json.dumps(rapport["manquants"], sort_keys=True))
    _ecrire_atomique(dossier / "reserve.json", json.dumps(
        {"ids": [position["position_id"] for position in reserve]},
        sort_keys=True, indent=2).encode("utf-8"))
    print(f"reserve de {len(reserve)} positions")


def _finalize_final(args, dossier: Path) -> None:
    annotes = lire_jsonl_zst(dossier / "annotated.jsonl.zst")
    rapport: dict = {}
    selection = select_positions(annotes, QUOTAS, salt=SEL_FINAL,
                                 event_cap=EVENT_CAP, player_cap=PLAYER_CAP,
                                 rapport=rapport)
    if not rapport["complet"]:
        raise SelectionIncomplete(
            "selection incomplete, ajouter le mois precedent : "
            + json.dumps(rapport["manquants"], sort_keys=True))

    moteur, identite = _ouvrir_stockfish(args)
    try:
        store = AnnotationStore(dossier, _config_store(identite))
        par_id = {position["position_id"]: position for position in annotes}
        profonds = []
        try:
            for position_id in select_audit_ids(selection):
                position = par_id[position_id]
                cle = cle_position(position, NODES_AUDIT)
                enregistre = store.get(cle)
                if enregistre is None:
                    enregistre = annotate_position(moteur, position,
                                                   NODES_AUDIT)
                    store.put(cle, enregistre)
                profonds.append(enregistre)
        finally:
            store.close()
    finally:
        moteur.quit()

    search_ids, audit = finaliser(selection, profonds)
    sources = json.loads(
        (dossier / "sources.json").read_text(encoding="utf-8"))
    manifeste = assembler_manifeste(
        dataset_version=args.version, sources=sources, selection=selection,
        search_ids=search_ids, stockfish=identite, audit=audit)
    chemin = write_dataset(selection, manifeste, args.output)
    print(f"banc {args.version} publie : {chemin}")


if __name__ == "__main__":
    raise SystemExit(main())
