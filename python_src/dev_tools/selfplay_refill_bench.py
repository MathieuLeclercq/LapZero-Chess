"""Banc de debit self-play : pool vide contre pool renouvele.

Le SelfPlayManager joue `total_games` parties avec `concurrent_games` places.
Quand une partie finit et qu'il reste des parties a jouer, sa place est
reprise par une nouvelle partie : le pool reste plein. Quand total == concurrent,
aucune reprise n'a lieu et le pool se vide au fil des fins de parties, donc les
lots d'evaluation retrecissent. Or le cout par position de l'evaluateur monte
quand le lot descend (0.10 ms a 128, 0.18 a 32, 0.41 a 8, mesures de septembre).

Ce banc compare, a simulations egales, un pool vide (regime de production
actuel : train_self_play.py passe total == concurrent) et un pool renouvele
(total = k x concurrent), en mesurant le debit reel.

Outil de diagnostic, hors validation.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402,F401  (charge les DLL CUDA de torch : cublasLt)
import chess_engine  # noqa: E402

MODELE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "checkpoints_onnx",
                      "2026_04_30_09h53_iter436_unsupervised.onnx")


def mesurer(evaluateur, concurrent, total, slow, fast, ratio, tt_size):
    debut = time.perf_counter()
    parties = chess_engine.generate_self_play_games(
        evaluateur, concurrent, slow, fast, total, ratio, tt_size)
    duree = time.perf_counter() - debut

    plies = sum(p.total_real_moves for p in parties)
    positions = sum(p.state_tensors.shape[0] for p in parties)
    return {
        "concurrent": concurrent,
        "total": total,
        "duree": duree,
        "parties": len(parties),
        "parties_par_s": len(parties) / duree,
        "plies": plies,
        "plies_par_s": plies / duree,
        "ms_par_ply": 1000.0 * duree / plies,
        "positions": positions,
        "positions_par_s": positions / duree,
    }


def afficher(resultats):
    entete = "  {:>10} {:>7} {:>10} {:>11} {:>10} {:>10}"
    print("[bench]" + entete.format("pool", "total", "duree (s)", "parties/s",
                                    "plies/s", "ms/ply"))
    for r in resultats:
        print("[bench]" + entete.format(
            f"({r['concurrent']},{r['total']})", str(r["total"]),
            f"{r['duree']:.0f}", f"{r['parties_par_s']:.3f}",
            f"{r['plies_par_s']:.1f}", f"{r['ms_par_ply']:.2f}"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODELE)
    parser.add_argument("--sims", type=int, default=100,
                        help="simulations par coup lent (defaut 100)")
    parser.add_argument("--fast", type=int, default=20,
                        help="simulations par coup rapide (defaut 20)")
    parser.add_argument("--ratio", type=float, default=0.25)
    parser.add_argument("--tt", type=int, default=4_000_000)
    parser.add_argument("--paires", default="32:1,32:4,64:1,64:4",
                        help="liste concurrent:total, total en multiples de concurrent")
    args = parser.parse_args()

    print(f"[bench] modele {os.path.basename(args.model)}")
    print(f"[bench] simulations {args.sims} lentes / {args.fast} rapides, "
          f"ratio {args.ratio}")

    evaluateur = chess_engine.ONNXEvaluator(args.model, True)
    resultats = []
    for paire in args.paires.split(","):
        concurrent, multiple = (int(x) for x in paire.split(":"))
        total = concurrent * multiple
        print(f"[bench] lancement pool ({concurrent},{total})", flush=True)
        resultats.append(mesurer(evaluateur, concurrent, total, args.sims,
                                 args.fast, args.ratio, args.tt))

    afficher(resultats)

    for i in range(0, len(resultats) - 1, 2):
        a, b = resultats[i], resultats[i + 1]
        if a["concurrent"] == b["concurrent"]:
            print(f"[bench] pool {a['concurrent']} : renouvele / vide = "
                  f"x{b['plies_par_s'] / a['plies_par_s']:.2f} sur les plies, "
                  f"x{b['parties_par_s'] / a['parties_par_s']:.2f} sur les parties, "
                  f"{a['ms_par_ply']:.2f} -> {b['ms_par_ply']:.2f} ms/ply")


if __name__ == "__main__":
    main()
