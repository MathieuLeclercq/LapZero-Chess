"""Banc de puzzles Lichess : orchestration.

Mesure un modele sur data/puzzles_bench.txt et ecrit un CSV par puzzle plus un
rapport agrege. Le scoring vit dans bench_metrics, testable sans modele.

Voir docs/superpowers/specs/2026-08-14-puzzle-bench-design.md
"""

import os
import sys
from pathlib import Path

RACINE_PYTHON = Path(__file__).resolve().parent
if str(RACINE_PYTHON) not in sys.path:
    sys.path.insert(0, str(RACINE_PYTHON))
os.add_dll_directory(str(RACINE_PYTHON))

import chess_engine

# Un MCTS neuf est cree a chaque recherche. Au defaut de 2 097 143 entrees a
# 1040 octets, chaque instance reserverait 2,03 Gio, soit 32,5 Gio a 16
# travailleurs pour 31,4 Gio de RAM. Une recherche ne stocke au plus que
# `simulations` positions distinctes, donc 8192 entrees suffisent largement.
TAILLE_TT = 8192


def exporter_onnx(chemin_pt: Path, sortie: Path) -> dict:
    """Exporte un checkpoint .pt en ONNX a axes dynamiques.

    torch n'est importe qu'ici, donc jamais dans un processus travailleur ou il
    couterait 475 Mio. L'architecture est deduite du checkpoint plutot que
    codee en dur, sans quoi un modele de taille differente casserait en
    silence.
    """
    import torch

    from model import ChessNet

    checkpoint = torch.load(chemin_pt, map_location="cpu", weights_only=True)
    etat = checkpoint["model_state_dict"]

    num_filters = etat["conv_input.weight"].shape[0]
    num_res_blocks = 1 + max(
        int(cle.split(".")[1]) for cle in etat if cle.startswith("res_blocks."))

    modele = ChessNet(num_res_blocks=num_res_blocks, num_filters=num_filters)
    modele.load_state_dict(etat)
    modele.eval()

    sortie.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        modele,
        torch.randn(1, 119, 8, 8),
        str(sortie),
        input_names=["input"],
        output_names=["policy", "value"],
        dynamic_axes={"input": {0: "batch_size"},
                      "policy": {0: "batch_size"},
                      "value": {0: "batch_size"}},
        # Sous torch 2.13 l'exporteur par defaut reclame onnxscript, absent.
        dynamo=False,
    )
    return {
        "iteration": checkpoint.get("iteration"),
        "global_step": checkpoint.get("global_step"),
        "num_res_blocks": num_res_blocks,
        "num_filters": num_filters,
    }


def resoudre_modele(chemin: Path, dossier_onnx: Path) -> tuple[Path, dict]:
    """Accepte un .onnx tel quel, ou exporte un .pt s'il le faut."""
    chemin = Path(chemin)
    if chemin.suffix == ".onnx":
        return chemin, {"iteration": None, "global_step": None,
                        "num_res_blocks": None, "num_filters": None}

    sortie = Path(dossier_onnx) / f"{chemin.stem}.onnx"
    if not sortie.exists():
        return sortie, exporter_onnx(chemin, sortie)

    # Un export deja present est reutilise, mais l'architecture est relue pour
    # le contexte du rapport.
    import torch

    checkpoint = torch.load(chemin, map_location="cpu", weights_only=True)
    etat = checkpoint["model_state_dict"]
    return sortie, {
        "iteration": checkpoint.get("iteration"),
        "global_step": checkpoint.get("global_step"),
        "num_res_blocks": 1 + max(int(c.split(".")[1]) for c in etat
                                  if c.startswith("res_blocks.")),
        "num_filters": etat["conv_input.weight"].shape[0],
    }


def faire_policy_fn(session):
    """Renvoie policy_fn(board) -> (probabilites sur les index legaux, value).

    Softmax masque sur les coups legaux. Le C++ fait un softmax sur les 4672
    sorties (onnx_evaluator.cpp:74-89) puis renormalise sur les coups legaux
    (mcts.cpp:184-198) : renormaliser un softmax global sur un sous-ensemble
    est identique a un softmax sur ce seul sous-ensemble, et un test verrouille
    l'accord.
    """
    import math

    import numpy as np

    def policy_fn(board):
        tenseur = np.asarray(board.get_alphazero_tensor(),
                             dtype=np.float32).reshape(1, 119, 8, 8)
        logits, value = session.run(None, {"input": tenseur})
        logits = logits[0]

        indices = board.get_legal_move_indices()
        maxi = max(float(logits[i]) for i in indices)
        exps = {i: math.exp(float(logits[i]) - maxi) for i in indices}
        somme = sum(exps.values())
        return ({i: e / somme for i, e in exps.items()},
                float(np.asarray(value).reshape(-1)[0]))

    return policy_fn


def faire_search_fn(evaluateur, simulations: int, c_puct: float,
                    batch_size: int = 0):
    """Renvoie search_fn(board) -> distribution de visites sur 4672.

    Un MCTS neuf a chaque recherche : la table de transposition est indexee sur
    le seul Zobrist, or les positions successives d'une meme ligne ont des
    historiques differents, donc un hit renverrait une value calculee sous un
    autre historique.
    """
    def search_fn(board):
        mcts = chess_engine.MCTS(evaluateur, TAILLE_TT)
        return mcts.mcts_search(board, simulations, c_puct, False, batch_size)

    return search_fn


CHAMPS_CSV = (
    "ligne", "rating", "themes", "plies_historique", "nb_coups_legaux",
    "coup_reseau", "reussi_reseau", "p_correct_reseau", "rang_correct_reseau",
    "value_reseau", "coup_recherche", "reussi_recherche",
    "part_visites_correct", "duree_s", "erreur",
)

# Etat par processus travailleur : la session onnxruntime et l'evaluateur ne
# sont crees qu'une fois par travailleur, pas a chaque puzzle.
_ETAT: dict = {}


def initialiser_travailleur(onnx: str, simulations: int, c_puct: float,
                            sans_historique: bool, batch_size: int) -> None:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(onnx, options,
                                   providers=["CPUExecutionProvider"])

    _ETAT["policy_fn"] = faire_policy_fn(session)
    _ETAT["search_fn"] = faire_search_fn(
        chess_engine.ONNXEvaluator(onnx, False), simulations, c_puct, batch_size)
    _ETAT["sans_historique"] = sans_historique


def traiter_lot(lot: list) -> list:
    """lot : liste de (index, ligne brute). Renvoie des PuzzleMeasure."""
    from bench_metrics import measure_puzzle, parse_bench_line

    return [
        measure_puzzle(
            parse_bench_line(index, ligne),
            _ETAT["policy_fn"],
            _ETAT["search_fn"],
            sans_historique=_ETAT["sans_historique"],
        )
        for index, ligne in lot
    ]


def ecrire_csv(mesures: list, chemin: Path) -> None:
    """Une ligne par puzzle, dans l'ordre du fichier de banc.

    Le pool rend les lots dans l'ordre d'achevement, donc le tri est necessaire
    pour que l'index de ligne reste un identifiant utilisable.
    """
    import csv
    import dataclasses

    chemin.parent.mkdir(parents=True, exist_ok=True)
    with open(chemin, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CHAMPS_CSV))
        writer.writeheader()
        for mesure in sorted(mesures, key=lambda m: m.ligne):
            writer.writerow(dataclasses.asdict(mesure))


def sous_echantillon(lignes: list, combien: int) -> list:
    """Retient `combien` lignes a pas regulier sur tout le fichier.

    Prendre les premieres lignes donnerait un echantillon biaise : le pipeline
    ecrit le banc tranche de rating par tranche de rating, donc les 200
    premieres lignes appartiennent toutes a la meme tranche. Le pas regulier
    reste deterministe, ce qui compte puisque le fichier n'a pas de PuzzleId et
    que l'index de ligne fait office d'identifiant.
    """
    if combien <= 0 or combien >= len(lignes):
        return lignes

    pas = len(lignes) / combien
    return [lignes[min(len(lignes) - 1, int(i * pas))] for i in range(combien)]


def _lots(lignes: list, taille: int) -> list:
    return [lignes[i:i + taille] for i in range(0, len(lignes), taille)]


def main() -> int:
    import argparse
    import multiprocessing as mp
    import time

    from bench_metrics import aggregate, format_report

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True,
                        help="checkpoint .pt ou modele .onnx")
    parser.add_argument("--banc", type=Path,
                        default=Path("../data/puzzles_bench.txt"))
    parser.add_argument("--dossier-onnx", type=Path,
                        default=Path("checkpoints_onnx"))
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--out-rapport", type=Path, default=None)
    # 700 simulations : le banc sert a deux comparaisons relatives, la
    # recherche contre la policy brute et une iteration contre la precedente.
    # Les deux valent a budget fixe, quel qu'il soit.
    parser.add_argument("--simulations", type=int, default=700)
    parser.add_argument("--c-puct", type=float, default=1.4)
    parser.add_argument("--batch-size", type=int, default=0,
                        help="taille du lot MCTS, 0 conserve la recherche sequentielle")
    parser.add_argument("--travailleurs", type=int, default=16)
    # 2500 sur les 5000 du fichier : la demi-largeur de Wilson globale passe
    # de 1,2 a 1,6 point seulement, et le passage tient en deux fois moins de
    # temps. Le pas etant regulier et le fichier ecrit tranche par tranche, on
    # garde 625 puzzles par tranche de rating.
    parser.add_argument("--limite", type=int, default=2500,
                        help="sous-echantillon de N puzzles, a pas regulier sur "
                             "tout le fichier ; 0 pour tout prendre. Les N "
                             "premieres lignes tomberaient toutes dans la meme "
                             "tranche de rating, le banc etant ecrit tranche "
                             "par tranche")
    parser.add_argument("--sans-historique", action="store_true",
                        help="presente les puzzles avec l'historique vide")
    args = parser.parse_args()

    if args.batch_size < 0:
        parser.error("--batch-size doit etre positif ou nul")

    if not args.banc.exists():
        print(f"fichier de banc introuvable : {args.banc}", file=sys.stderr)
        return 2

    onnx, meta = resoudre_modele(args.model, args.dossier_onnx)
    if not Path(onnx).exists():
        print(f"modele ONNX introuvable : {onnx}", file=sys.stderr)
        return 2

    with open(args.banc, encoding="utf-8") as f:
        lignes = list(enumerate(f))
    total_fichier = len(lignes)
    if args.limite:
        lignes = sous_echantillon(lignes, args.limite)

    # Posee dans le parent pour etre heritee : sous Windows le pool utilise
    # spawn et reimporte le module avant d'executer l'initialiseur, donc la
    # poser dans l'initialiseur serait trop tard.
    os.environ["OMP_NUM_THREADS"] = "1"

    suffixe = " (sans historique)" if args.sans_historique else ""
    print(f"{len(lignes)} puzzles sur {total_fichier} du fichier, "
          f"{args.simulations} simulations, premier coup seul, "
          f"batch {args.batch_size}, {args.travailleurs} travailleurs{suffixe}")

    debut = time.perf_counter()
    lots = _lots(lignes, 16)
    mesures: list = []
    with mp.Pool(args.travailleurs, initializer=initialiser_travailleur,
                 initargs=(str(onnx), args.simulations, args.c_puct,
                           args.sans_historique, args.batch_size)) as pool:
        for i, resultat in enumerate(pool.imap_unordered(traiter_lot, lots), 1):
            mesures.extend(resultat)
            if i % 10 == 0 or i == len(lots):
                ecoule = time.perf_counter() - debut
                print(f"  {len(mesures)}/{len(lignes)} puzzles, "
                      f"{ecoule / 60.0:.1f} min, "
                      f"{len(mesures) / max(1e-9, ecoule):.1f} puzzles/s",
                      flush=True)
    duree = time.perf_counter() - debut

    stats = aggregate(mesures)
    contexte = {
        "modele": Path(onnx).name,
        "iteration": meta.get("iteration"),
        "global_step": meta.get("global_step"),
        "simulations": args.simulations,
        "c_puct": args.c_puct,
        "batch_size": args.batch_size,
        "fichier_banc": str(args.banc),
        "sans_historique": args.sans_historique,
        "duree_totale_s": duree,
        "travailleurs": args.travailleurs,
    }

    out_csv = args.out_csv or Path(
        f"../data/bench_results/{Path(onnx).stem}_batch{args.batch_size}.csv")
    out_rapport = args.out_rapport or Path(
        f"../docs/superpowers/specs/{time.strftime('%Y-%m-%d')}"
        f"-puzzle-bench-batch{args.batch_size}-resultats.md")

    ecrire_csv(mesures, out_csv)
    out_rapport.parent.mkdir(parents=True, exist_ok=True)
    out_rapport.write_text(format_report(stats, contexte), encoding="utf-8")

    print(f"\nCSV     : {out_csv}")
    print(f"Rapport : {out_rapport}")
    print(f"Duree   : {duree / 60.0:.1f} min")
    if stats.erreurs:
        print(f"Erreurs de donnees : {stats.erreurs}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    sys.exit(main())
