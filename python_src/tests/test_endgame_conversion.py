import os
import sys
from pathlib import Path
from types import SimpleNamespace

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE / "python_src"))
sys.path.insert(0, str(RACINE / "python_src" / "dev_tools"))
os.add_dll_directory(str(RACINE / "python_src"))

import endgame_conversion


def test_mode_fresh_conserve_la_profondeur_tt(monkeypatch):
    constructions = []

    class FauxMCTS:
        def __init__(self, evaluator, taille_tt, cache_history_depth):
            constructions.append((taille_tt, cache_history_depth))

        def reset_counters(self):
            pass

        def step_analysis(self, board, simulations, c_puct, batch_size):
            pass

        def get_counters(self):
            return SimpleNamespace(
                tt_hits=0, tt_misses=1, tt_rule50_rejects=0,
                tt_context_rejects=0, tt_history_rejects=0)

        def get_analysis_results(self):
            return []

    monkeypatch.setattr(endgame_conversion.chess_engine, "MCTS", FauxMCTS)

    resultat = endgame_conversion.jouer(
        "tour", endgame_conversion.POSITIONS["tour"], object(),
        simulations=8, batch_size=0, max_plies=1, tree_mode="fresh",
        cache_history_depth=3)

    assert constructions == [(131071, 3), (131071, 3)]
    assert resultat.cache_history_depth == 3
    assert resultat.tt_misses == 1
