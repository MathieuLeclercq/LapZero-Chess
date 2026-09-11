"""Harnais de mesure et de verification de la recherche MCTS.

Reference reproductible du debit, et filet contre la corruption silencieuse
d'arbre, avant le batching avec virtual loss.

Voir docs/superpowers/specs/2026-09-11-search-bench-design.md
"""

import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

RACINE_PYTHON = Path(__file__).resolve().parent
if str(RACINE_PYTHON) not in sys.path:
    sys.path.insert(0, str(RACINE_PYTHON))
os.add_dll_directory(str(RACINE_PYTHON))

import chess_engine

# Jamais le defaut : une TTEntry pese 1040 octets et le defaut de MCTS est
# 2 097 143 entrees, soit 2,03 Gio par instance.
TAILLE_TT = 8192

# Trois positions de reference. Un debit mesure sur une seule position ne
# represente rien : sur une position de mat, 381 simulations sur 400 s'arretent
# sur un noeud terminal sans appeler le reseau, ce qui affiche 5624 sims/s.
# Les trois positions ci-dessous ont ete verifiees exemptes de ce biais, zero
# noeud terminal et 401 inferences pour 400 simulations. Les deux dernieres
# viennent des positions de reference du perft, donc deja verifiees legales.
POSITIONS = (
    ("ouverture",
     "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4"),
    ("milieu",
     "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("finale",
     "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
)


@dataclass(frozen=True)
class Mesure:
    position: str
    chemin: str
    simulations: int
    duree_s: float
    nn_calls: int
    tt_hits: int
    tt_misses: int
    terminal_hits: int


def sims_par_seconde(m: Mesure) -> float:
    return m.simulations / m.duree_s if m.duree_s > 0 else 0.0


def inferences_par_seconde(m: Mesure) -> float:
    """Distincte du debit de simulations : une simulation qui s'arrete sur un
    noeud terminal ou sur un succes de table ne coute aucune inference."""
    return m.nn_calls / m.duree_s if m.duree_s > 0 else 0.0


def taux_table(m: Mesure) -> float:
    total = m.tt_hits + m.tt_misses
    return m.tt_hits / total if total else 0.0


def charger_position(fen: str):
    board = chess_engine.Chessboard()
    board.load_fen(fen)
    return board


def mesurer_mcts_search(evaluateur, fen: str, nom: str,
                        simulations: int, c_puct: float = 1.4) -> Mesure:
    """Arbre neuf a chaque appel."""
    mcts = chess_engine.MCTS(evaluateur, TAILLE_TT)
    board = charger_position(fen)
    mcts.reset_counters()

    debut = time.perf_counter()
    mcts.mcts_search(board, simulations, c_puct, False)
    duree = time.perf_counter() - debut

    c = mcts.get_counters()
    return Mesure(nom, "mcts_search", simulations, duree,
                  c.nn_calls, c.tt_hits, c.tt_misses, c.terminal_hits)


def mesurer_step_analysis(evaluateur, fen: str, nom: str,
                          simulations: int, c_puct: float = 1.4) -> Mesure:
    """Le chemin reel du bot : arbre d'analyse reutilise entre les coups."""
    mcts = chess_engine.MCTS(evaluateur, TAILLE_TT)
    board = charger_position(fen)
    mcts.reset_analysis()
    mcts.reset_counters()

    debut = time.perf_counter()
    mcts.step_analysis(board, simulations, c_puct)
    duree = time.perf_counter() - debut

    c = mcts.get_counters()
    return Mesure(nom, "step_analysis", simulations, duree,
                  c.nn_calls, c.tt_hits, c.tt_misses, c.terminal_hits)


def agreger(mesures: list) -> dict:
    """Groupe par (position, chemin), avec mediane et etendue.

    L'etendue est indispensable : sans elle, un gain de 5 pour cent serait
    indistinguable du bruit de mesure.
    """
    groupes: dict = {}
    for m in mesures:
        groupes.setdefault((m.position, m.chemin), []).append(m)

    resultat = {}
    for cle, lot in groupes.items():
        debits = [sims_par_seconde(m) for m in lot]
        inferences = [inferences_par_seconde(m) for m in lot]
        resultat[cle] = {
            "passages": len(lot),
            "simulations": lot[0].simulations,
            "sims_par_seconde_median": statistics.median(debits),
            "sims_par_seconde_min": min(debits),
            "sims_par_seconde_max": max(debits),
            "inferences_par_seconde_median": statistics.median(inferences),
            "taux_table_median": statistics.median(taux_table(m) for m in lot),
            "terminal_hits_median": statistics.median(
                m.terminal_hits for m in lot),
        }
    return resultat
