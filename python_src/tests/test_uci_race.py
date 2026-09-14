"""La race entre parse_position et le fil de recherche.

parse_position appelle update_root et reconstruit self.board. Tant que le fil
de recherche tourne, il descend dans un arbre qu'update_root est en train de
detruire. Aujourd'hui c'est un bug latent ; avec le batching la boucle
detiendra des MCTSNode* bruts pendant plusieurs millisecondes d'appel GPU, et
cela deviendra une ecriture en memoire liberee.
"""
import os
import sys
from pathlib import Path

RACINE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, RACINE)
os.add_dll_directory(RACINE)

import chess_engine
import uci
from uci import UCIEngine


RACINE_PROJET = Path(__file__).resolve().parents[2]


def test_les_trois_lanceurs_utilisent_le_batch_valide():
    """Le batch 8 est le compromis valide par le banc appaire de puzzles."""
    for nom in ("uci.py", "play_against_bot.py", "tournament_elo.py"):
        source = (RACINE_PROJET / "python_src" / nom).read_text(encoding="utf-8")
        assert "MCTS_BATCH_SIZE = 8" in source, nom


def test_le_modele_uci_configure_est_resolu_depuis_le_depot():
    attendu = Path(uci.__file__).resolve().parent / "checkpoints_onnx"

    assert Path(uci.MODEL_PATH).parent == attendu


class _FauxMCTS:
    """Enregistre les appels, sans moteur ni modele."""

    def __init__(self):
        self.appels = []

    def update_root(self, move_idx):
        self.appels.append(("update_root", move_idx))

    def reset_analysis(self):
        self.appels.append(("reset_analysis",))


def _moteur():
    """UCIEngine sans modele : c'est la raison d'etre de l'injection.

    MODEL_PATH (uci.py) pointe vers une autre machine, donc UCIEngine() n'est
    pas constructible ici.
    """
    return UCIEngine(evaluator=object(), mcts=_FauxMCTS())


def test_parse_position_arrete_la_recherche_avant_de_toucher_a_l_arbre():
    moteur = _moteur()
    ordre = []
    moteur.stop_search = lambda: ordre.append("stop_search")
    moteur.mcts.update_root = lambda idx: ordre.append("update_root")

    moteur.parse_position(["startpos", "moves", "e2e4"])

    assert "stop_search" in ordre, "parse_position n'arrete pas la recherche"
    assert ordre[0] == "stop_search", f"stop_search doit venir en premier, ordre={ordre}"


def test_parse_position_arrete_la_recherche_aussi_sur_le_chemin_de_reconstruction():
    """Le chemin 3, nouvelle partie ou ponder miss, reassigne self.board."""
    moteur = _moteur()
    ordre = []
    moteur.stop_search = lambda: ordre.append("stop_search")
    moteur.mcts.reset_analysis = lambda: ordre.append("reset_analysis")

    moteur.parse_position(["startpos"])

    assert ordre[0] == "stop_search", f"ordre={ordre}"
    assert "reset_analysis" in ordre


def test_le_constructeur_accepte_des_dependances_injectees():
    faux = _FauxMCTS()
    moteur = UCIEngine(evaluator=object(), mcts=faux)

    assert moteur.mcts is faux
    assert isinstance(moteur.board, chess_engine.Chessboard)
