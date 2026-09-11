"""Le virtual loss, approche LC0 : un compteur separe qui n'entre que dans le
denominateur du terme U.

Tant que rien ne l'incremente, ce champ ne doit rien changer. Les tests de
comportement reel arrivent avec la boucle batchee.
"""
import os
import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE / "python_src"))
os.add_dll_directory(str(RACINE / "python_src"))

import chess_engine

CHECKPOINT = (RACINE / "python_src" / "checkpoints"
              / "2026_04_23_23h25_iter316_unsupervised.pt")

pytestmark = pytest.mark.skipif(not CHECKPOINT.exists(),
                                reason="checkpoint absent")

TAILLE_TT = 8192
DEPART = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


@pytest.fixture(scope="module")
def evaluateur(tmp_path_factory):
    import puzzle_bench

    chemin, _ = puzzle_bench.resoudre_modele(
        CHECKPOINT, tmp_path_factory.mktemp("onnx"))
    return chess_engine.ONNXEvaluator(str(chemin), False)


def _plateau(fen=DEPART):
    board = chess_engine.Chessboard()
    board.load_fen(fen)
    return board


def test_aucun_noeud_ne_reste_en_vol_apres_une_recherche(evaluateur):
    """Un virtual loss non annule laisserait des noeuds en vol, ce qui
    fausserait le terme U de toutes les recherches suivantes."""
    mcts = chess_engine.MCTS(evaluateur, TAILLE_TT)
    mcts.step_analysis(_plateau(), 200, 1.4)

    assert mcts.inspect_tree().en_vol == 0


def test_les_visites_de_la_racine_egalent_les_simulations(evaluateur):
    """Rappel du filet existant : c'est le controle qui detecte un virtual loss
    mal annule, puisqu'il gonflerait ce compte."""
    mcts = chess_engine.MCTS(evaluateur, TAILLE_TT)
    board = _plateau()

    mcts.step_analysis(board, 200, 1.4)

    assert sum(s.visits for s in mcts.get_analysis_results()) == 200


def test_la_recherche_reste_deterministe(evaluateur):
    """Deux recherches identiques doivent donner exactement la meme
    distribution : sans Dirichlet, rien n'introduit d'alea."""
    board_a = _plateau()
    board_b = _plateau()

    pi_a = chess_engine.MCTS(evaluateur, TAILLE_TT).mcts_search(board_a, 200, 1.4, False)
    pi_b = chess_engine.MCTS(evaluateur, TAILLE_TT).mcts_search(board_b, 200, 1.4, False)

    assert list(pi_a) == list(pi_b)
