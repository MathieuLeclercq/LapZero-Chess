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
    batch_size: int
    duree_s: float
    nn_calls: int
    nn_batches: int
    tt_hits: int
    tt_misses: int
    terminal_hits: int


def sims_par_seconde(m: Mesure) -> float:
    return m.simulations / m.duree_s if m.duree_s > 0 else 0.0


def inferences_par_seconde(m: Mesure) -> float:
    """Positions passees au reseau par seconde.

    Une simulation qui s'arrete sur un noeud terminal ou sur un succes de table
    ne coute aucune evaluation. Ce n'est pas le nombre d'appels batch, expose
    separement par appels_batch_par_seconde.
    """
    return m.nn_calls / m.duree_s if m.duree_s > 0 else 0.0


def appels_batch_par_seconde(m: Mesure) -> float:
    return m.nn_batches / m.duree_s if m.duree_s > 0 else 0.0


def remplissage_moyen(m: Mesure) -> float:
    return m.nn_calls / m.nn_batches if m.nn_batches else 0.0


def taux_table(m: Mesure) -> float:
    total = m.tt_hits + m.tt_misses
    return m.tt_hits / total if total else 0.0


def charger_position(fen: str):
    board = chess_engine.Chessboard()
    board.load_fen(fen)
    return board


def mesurer_mcts_search(evaluateur, fen: str, nom: str,
                        simulations: int, c_puct: float = 1.4,
                        batch_size: int = 0) -> Mesure:
    """Arbre neuf a chaque appel."""
    mcts = chess_engine.MCTS(evaluateur, TAILLE_TT)
    board = charger_position(fen)
    mcts.reset_counters()

    debut = time.perf_counter()
    mcts.mcts_search(board, simulations, c_puct, False, batch_size)
    duree = time.perf_counter() - debut

    c = mcts.get_counters()
    return Mesure(nom, "mcts_search", simulations, batch_size, duree,
                  c.nn_calls, c.nn_batches, c.tt_hits, c.tt_misses,
                  c.terminal_hits)


def mesurer_step_analysis(evaluateur, fen: str, nom: str,
                          simulations: int, c_puct: float = 1.4,
                          batch_size: int = 0) -> Mesure:
    """Le chemin reel du bot : arbre d'analyse reutilise entre les coups."""
    mcts = chess_engine.MCTS(evaluateur, TAILLE_TT)
    board = charger_position(fen)
    mcts.reset_analysis()
    mcts.reset_counters()

    debut = time.perf_counter()
    mcts.step_analysis(board, simulations, c_puct, batch_size)
    duree = time.perf_counter() - debut

    c = mcts.get_counters()
    return Mesure(nom, "step_analysis", simulations, batch_size, duree,
                  c.nn_calls, c.nn_batches, c.tt_hits, c.tt_misses,
                  c.terminal_hits)


def agreger(mesures: list) -> dict:
    """Groupe par (position, chemin, taille de batch), avec mediane et etendue.

    L'etendue est indispensable : sans elle, un gain de 5 pour cent serait
    indistinguable du bruit de mesure.
    """
    groupes: dict = {}
    for m in mesures:
        groupes.setdefault((m.position, m.chemin, m.batch_size), []).append(m)

    resultat = {}
    for cle, lot in groupes.items():
        debits = [sims_par_seconde(m) for m in lot]
        inferences = [inferences_par_seconde(m) for m in lot]
        appels_batch = [appels_batch_par_seconde(m) for m in lot]
        resultat[cle] = {
            "passages": len(lot),
            "simulations": lot[0].simulations,
            "sims_par_seconde_median": statistics.median(debits),
            "sims_par_seconde_min": min(debits),
            "sims_par_seconde_max": max(debits),
            "inferences_par_seconde_median": statistics.median(inferences),
            "appels_batch_par_seconde_median": statistics.median(appels_batch),
            "remplissage_moyen_median": statistics.median(
                remplissage_moyen(m) for m in lot),
            "taux_table_median": statistics.median(taux_table(m) for m in lot),
            "terminal_hits_median": statistics.median(
                m.terminal_hits for m in lot),
        }
    return resultat


_EN_TETE = (
    "| Position | Chemin | batch | passages | sims/s (med) | sims/s (min a max) "
    "| positions reseau/s | appels batch/s | remplissage | taux table |\n"
    "|---|---|---|---|---|---|---|---|---|---|"
)


def format_report(agr: dict, contexte: dict, invariants: list) -> str:
    """Rapport markdown.

    Le mot « noeuds par seconde » est proscrit : le perft mesure la generation
    de coups a plus de 1,6 million de noeuds par seconde, la recherche tourne a
    environ 295 simulations par seconde.
    """
    lignes = [
        "# Banc de recherche : resultats",
        "",
        f"Modele : `{contexte['modele']}`, iteration {contexte['iteration']}, "
        f"global_step {contexte['global_step']}",
        f"Protocole : {contexte['passages']} passages, "
        f"{contexte['simulations']} simulations, c_puct {contexte['c_puct']}, "
        f"{contexte.get('accelerateur', 'CPU')}, un seul processus, "
        f"batches demandes {contexte.get('batch_sizes', [0])}",
        "",
        "Trois grandeurs distinctes. Les **simulations par seconde** mesurent le",
        "debit de la recherche. Les **positions reseau par seconde** comptent les",
        "positions effectivement evaluees, les **appels batch par seconde** les",
        "lancements physiques et leur ratio est le **remplissage**. Le **taux de",
        "table** est la part des consultations reussies.",
        "",
        "## Debit",
        "",
        _EN_TETE,
    ]

    for cle in sorted(agr):
        position, chemin, batch_size = cle
        a = agr[cle]
        lignes.append(
            f"| {position} | {chemin} | {batch_size} | {a['passages']} "
            f"| {a['sims_par_seconde_median']:.1f} "
            f"| {a['sims_par_seconde_min']:.1f} a {a['sims_par_seconde_max']:.1f} "
            f"| {a['inferences_par_seconde_median']:.1f} "
            f"| {a['appels_batch_par_seconde_median']:.1f} "
            f"| {a['remplissage_moyen_median']:.1f} "
            f"| {100 * a['taux_table_median']:.1f} % |"
        )

    lignes += ["", "## Invariants d'arbre", ""]
    if not invariants:
        lignes.append("Non verifies lors de ce passage.")
    else:
        lignes += [
            "| Position | batch | noeuds | profondeur max | violations |",
            "|---|---|---|---|---|",
        ]
        total = 0
        for (position, batch_size, nodes, max_depth, violations, _messages) in invariants:
            total += violations
            lignes.append(
                f"| {position} | {batch_size} | {nodes} | {max_depth} | {violations} |")
        lignes.append("")
        if total == 0:
            lignes.append("Aucune violation.")
        else:
            lignes.append(f"**{total} violations.** Premiers messages :")
            lignes.append("")
            for (position, batch_size, _n, _d, _v, messages) in invariants:
                for message in messages:
                    lignes.append(f"- `{position}`, batch {batch_size} : {message}")

    return "\n".join(lignes) + "\n"


def main() -> int:
    import argparse

    import puzzle_bench

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True,
                        help="checkpoint .pt ou modele .onnx")
    parser.add_argument("--dossier-onnx", type=Path,
                        default=Path("checkpoints_onnx"))
    parser.add_argument("--simulations", type=int, default=400)
    parser.add_argument("--passages", type=int, default=5)
    parser.add_argument("--c-puct", type=float, default=1.4)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[0],
                        help="tailles de batch a balayer, 0 designe la boucle "
                             "sequentielle conservee")
    parser.add_argument("--gpu", action="store_true",
                        help="utilise le provider CUDA du moteur C++")
    parser.add_argument("--invariants", action="store_true",
                        help="verifie les invariants d'arbre, sans mesurer le debit")
    parser.add_argument("--out-rapport", type=Path, default=None)
    args = parser.parse_args()

    if any(taille < 0 for taille in args.batch_sizes):
        parser.error("les tailles de batch doivent etre positives ou nulles")

    onnx, meta = puzzle_bench.resoudre_modele(args.model, args.dossier_onnx)
    if not Path(onnx).exists():
        print(f"modele ONNX introuvable : {onnx}", file=sys.stderr)
        return 2

    if args.gpu:
        # Les DLL CUDA de torch rendent celles du provider ONNX visibles au
        # processus avant la construction de l'evaluateur C++.
        import torch  # noqa: F401
    evaluateur = chess_engine.ONNXEvaluator(str(onnx), args.gpu)

    # Rodage : la premiere inference initialise la session.
    chauffe = chess_engine.MCTS(evaluateur, TAILLE_TT)
    chauffe.mcts_search(charger_position(POSITIONS[0][1]), 8, args.c_puct,
                        False, args.batch_sizes[0])

    invariants = []
    mesures = []

    if args.invariants:
        # inspect_tree n'est jamais appele dans la boucle de mesure de debit :
        # son cout croit avec la taille de l'arbre et fausserait la mesure
        # qu'il protege. Les deux jambes sont donc exclusives.
        for batch_size in args.batch_sizes:
            for nom, fen in POSITIONS:
                mcts = chess_engine.MCTS(evaluateur, TAILLE_TT)
                mcts.step_analysis(charger_position(fen), args.simulations,
                                   args.c_puct, batch_size)
                r = mcts.inspect_tree()
                invariants.append((nom, batch_size, r.nodes, r.max_depth,
                                   r.violations, list(r.messages)))
                etat = "OK" if r.violations == 0 else f"{r.violations} VIOLATIONS"
                print(f"  {nom:10} batch {batch_size:2} : {r.nodes} noeuds, "
                      f"profondeur {r.max_depth}, {etat}")
    else:
        for passage in range(args.passages):
            for batch_size in args.batch_sizes:
                for nom, fen in POSITIONS:
                    mesures.append(mesurer_mcts_search(
                        evaluateur, fen, nom, args.simulations, args.c_puct,
                        batch_size))
                    mesures.append(mesurer_step_analysis(
                        evaluateur, fen, nom, args.simulations, args.c_puct,
                        batch_size))
            print(f"  passage {passage + 1}/{args.passages}", flush=True)

    agr = agreger(mesures)
    for cle in sorted(agr):
        a = agr[cle]
        print(f"  {cle[0]:10} {cle[1]:14} batch {cle[2]:2} : "
            f"{a['sims_par_seconde_median']:7.1f} sims/s, "
              f"{a['inferences_par_seconde_median']:7.1f} positions/s, "
              f"remplissage {a['remplissage_moyen_median']:.1f}, "
              f"table {100 * a['taux_table_median']:.1f} %")

    contexte = {
        "modele": Path(onnx).name,
        "iteration": meta.get("iteration"),
        "global_step": meta.get("global_step"),
        "passages": args.passages if not args.invariants else 1,
        "simulations": args.simulations,
        "c_puct": args.c_puct,
        "accelerateur": "GPU" if args.gpu else "CPU",
        "batch_sizes": args.batch_sizes,
    }

    sortie = args.out_rapport or Path(
        f"../docs/superpowers/specs/{time.strftime('%Y-%m-%d')}"
        "-search-bench-resultats.md")
    sortie.parent.mkdir(parents=True, exist_ok=True)
    sortie.write_text(format_report(agr, contexte, invariants), encoding="utf-8")
    print(f"\nRapport : {sortie}")

    total_violations = sum(v for (_, _, _, _, v, _) in invariants)
    if total_violations:
        print(f"{total_violations} violations d'invariants", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
