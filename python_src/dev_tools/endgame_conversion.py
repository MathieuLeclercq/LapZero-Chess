"""Diagnostic reproductible de conversion de finales par le modele et le MCTS.

python-chess sert uniquement d'oracle pour la legalite et l'etat de partie.
Le choix des coups et la recherche restent ceux du moteur C++.
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import chess


RACINE = Path(__file__).resolve().parents[2]
PYTHON_SRC = RACINE / "python_src"
sys.path.insert(0, str(PYTHON_SRC))
os.add_dll_directory(str(PYTHON_SRC))

import chess_engine
from lib import coords_to_uci, decode_move_index


POSITIONS = {
    "tour": "8/8/8/8/8/2k5/8/R3K3 w - - 0 1",
    "dame_pion_contre_fou": "8/8/8/3b4/6k1/8/6P1/5QK1 w - - 0 1",
}


@dataclass
class Resultat:
    position: str
    batch_size: int
    cache_history_depth: int
    plies: int
    issue: str
    duree_s: float
    max_halfmove: int
    coups: list[str]
    tree_mode: str
    tt_hits: int
    tt_misses: int
    tt_rule50_rejects: int
    tt_context_rejects: int
    tt_history_rejects: int


def construire_evaluateur(model: Path, gpu: bool):
    if gpu:
        try:
            return chess_engine.ONNXEvaluator(str(model), True), "GPU"
        except Exception as exc:
            print(f"GPU indisponible, repli CPU : {exc}")
    return chess_engine.ONNXEvaluator(str(model), False), "CPU"


def verifier_et_pousser(oracle: chess.Board, moteur, uci: str) -> None:
    move = chess.Move.from_uci(uci)
    if move not in oracle.legal_moves:
        raise AssertionError(f"coup moteur illegal selon python-chess : {uci}")
    oracle.push(move)

    moteur_fields = moteur.to_fen().split()
    oracle_fields = oracle.fen().split()
    for index in (0, 1, 2, 4, 5):
        if moteur_fields[index] != oracle_fields[index]:
            raise AssertionError(
                f"desynchronisation apres {uci}:\n"
                f"moteur={moteur.to_fen()}\npython={oracle.fen()}")


def nom_issue(board, oracle: chess.Board, max_plies_atteint: bool) -> str:
    if board.game_state == chess_engine.GameState.CHECKMATE:
        return "mat"
    if board.game_state == chess_engine.GameState.DRAW_50_MOVES:
        return "nulle_50_coups"
    if board.game_state == chess_engine.GameState.DRAW_REPETITION:
        return "nulle_repetition"
    if board.game_state == chess_engine.GameState.STALEMATE:
        return "pat"
    if board.game_state == chess_engine.GameState.DRAW_INSUFF_MATERIAL:
        return "materiel_insuffisant"
    if oracle.is_game_over(claim_draw=True):
        return str(oracle.outcome(claim_draw=True).termination.name).lower()
    return "limite_plies" if max_plies_atteint else "inconnue"


def jouer(position: str, fen: str, evaluator, simulations: int,
          batch_size: int, max_plies: int, tree_mode: str,
          cache_history_depth: int = 1) -> Resultat:
    board = chess_engine.Chessboard()
    board.load_fen(fen)
    oracle = chess.Board(fen)
    mcts = chess_engine.MCTS(evaluator, 131071, cache_history_depth)
    coups = []
    tt_hits = 0
    tt_misses = 0
    tt_rule50_rejects = 0
    tt_context_rejects = 0
    tt_history_rejects = 0
    max_halfmove = board.half_move_clock
    debut = time.perf_counter()

    for _ in range(max_plies):
        if board.game_state != chess_engine.GameState.ONGOING:
            break

        if tree_mode == "fresh":
            mcts = chess_engine.MCTS(evaluator, 131071, cache_history_depth)
        elif tree_mode == "reset":
            # Supprime uniquement l'arbre. La table de transpositions de
            # l'instance reste chaude, ce qui isole son effet.
            mcts.reset_analysis()
        mcts.reset_counters()
        mcts.step_analysis(board, simulations, 1.4, batch_size)
        counters = mcts.get_counters()
        tt_hits += counters.tt_hits
        tt_misses += counters.tt_misses
        tt_rule50_rejects += counters.tt_rule50_rejects
        tt_context_rejects += counters.tt_context_rejects
        tt_history_rejects += counters.tt_history_rejects
        stats = mcts.get_analysis_results()
        if not stats:
            break

        best = stats[0].move_idx
        is_black = board.turn == chess_engine.Color.BLACK
        orig_f, orig_r, dest_f, dest_r, promotion = decode_move_index(
            board, best, is_black)
        uci = coords_to_uci(orig_f, orig_r, dest_f, dest_r, promotion)

        if tree_mode == "reuse":
            mcts.update_root(best)
        if not board.move_piece(
                orig_f, orig_r, dest_f, dest_r, promotion, True):
            raise AssertionError(f"le moteur refuse son propre coup : {uci}")
        verifier_et_pousser(oracle, board, uci)
        coups.append(uci)
        max_halfmove = max(max_halfmove, board.half_move_clock)

    return Resultat(
        position=position,
        batch_size=batch_size,
        cache_history_depth=cache_history_depth,
        plies=len(coups),
        issue=nom_issue(board, oracle, len(coups) >= max_plies),
        duree_s=time.perf_counter() - debut,
        max_halfmove=max_halfmove,
        coups=coups,
        tree_mode=tree_mode,
        tt_hits=tt_hits,
        tt_misses=tt_misses,
        tt_rule50_rejects=tt_rule50_rejects,
        tt_context_rejects=tt_context_rejects,
        tt_history_rejects=tt_history_rejects,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=Path,
        default=PYTHON_SRC / "checkpoints_onnx"
        / "2026_04_23_23h25_iter316_unsupervised.onnx")
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--max-plies", type=int, default=110)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[0, 8])
    parser.add_argument("--cache-history-depths", nargs="+", type=int,
                        default=[1])
    parser.add_argument("--positions", nargs="+", choices=POSITIONS,
                        default=list(POSITIONS))
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--tree-mode", choices=("reuse", "reset", "fresh"), default="reuse",
        help=("reuse conserve arbre et TT, reset supprime l'arbre mais garde "
              "la TT, fresh recree les deux avant chaque coup"))
    args = parser.parse_args()

    if any(depth < -1 or depth > 7 for depth in args.cache_history_depths):
        parser.error("les profondeurs de cache doivent etre comprises entre -1 et 7")

    evaluator, provider = construire_evaluateur(args.model, not args.cpu)
    print(f"Modele : {args.model.name}, provider : {provider}")

    for name in args.positions:
        for depth in args.cache_history_depths:
            for batch_size in args.batch_sizes:
                result = jouer(
                    name, POSITIONS[name], evaluator, args.simulations,
                    batch_size, args.max_plies, args.tree_mode, depth)
                policy = "legacy" if depth == -1 else f"h{depth}"
                print(
                    f"{name:24} TT={policy:6} batch={batch_size:2} "
                    f"issue={result.issue:20} plies={result.plies:3} "
                    f"max_halfmove={result.max_halfmove:3} "
                    f"mode={result.tree_mode:5} "
                    f"tt={result.tt_hits}/{result.tt_hits + result.tt_misses} "
                    f"rejets=50:{result.tt_rule50_rejects},"
                    f"contexte:{result.tt_context_rejects},"
                    f"historique:{result.tt_history_rejects} "
                    f"duree={result.duree_s:.1f}s")
                print("  " + " ".join(result.coups))


if __name__ == "__main__":
    main()
