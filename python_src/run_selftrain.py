"""Lance la boucle self-play + entrainement sur une machine distante.

Ne touche pas a la configuration de train_self_play.py : toutes les valeurs
qui changent d'une campagne a l'autre sont des arguments. Les defauts
reprennent la production locale (512 parties, 256 places, 700/100 simulations).

A lancer depuis le dossier python_src, pour que les chemins relatifs
`checkpoints/` et `replay_buffer/` tombent a cote des scripts.

La garde `if __name__ == "__main__"` est obligatoire : l'evaluation Stockfish
utilise multiprocessing en mode spawn, et chaque processus enfant reimporte ce
script. Sans la garde, chaque worker relance une campagne complete.

Exemple :

    python run_selftrain.py --iterations 4
    python run_selftrain.py --iterations 4 --wandb-mode disabled
"""
import argparse
import glob
import os


def etat_du_buffer(dossier):
    """Compte les shards et les positions deja presentes dans le buffer."""
    shards = sorted(glob.glob(os.path.join(dossier, "shard_*.npz")))
    positions = 0
    for shard in shards:
        try:
            positions += int(os.path.splitext(shard)[0].split("_")[-1])
        except (ValueError, IndexError):
            continue
    return shards, positions


def date_du_shard(chemin):
    morceaux = os.path.basename(chemin).split("_")
    if len(morceaux) > 2 and len(morceaux[1]) == 8:
        return f"{morceaux[1][:4]}-{morceaux[1][4:6]}-{morceaux[1][6:8]}"
    return "?"


def analyser_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--games", type=int, default=512,
                        help="parties par iteration")
    parser.add_argument("--concurrent", type=int, default=256,
                        help="places du pool self-play")
    parser.add_argument("--slow-sims", type=int, default=700)
    parser.add_argument("--fast-sims", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--buffer", type=int, default=750_000,
                        help="plafond du replay buffer sur disque")
    parser.add_argument("--sampling-ratio", type=float, default=14.0)
    parser.add_argument("--eval-every", type=int, default=8,
                        help="evaluer contre Stockfish tous les N tours")
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint .pt de reprise ; par defaut iter436")
    parser.add_argument("--stockfish", default=None,
                        help="chemin de l'executable Stockfish de l'ancre")
    parser.add_argument("--stockfish-elo", type=int, default=2600)
    parser.add_argument("--wandb-mode", choices=["offline", "disabled", "online"],
                        default="offline",
                        help="offline ecrit les metriques en local sans reseau, "
                             "disabled ne logue rien, online envoie directement")
    return parser.parse_args()


def main():
    args = analyser_arguments()

    if args.wandb_mode != "online":
        os.environ["WANDB_MODE"] = args.wandb_mode

    shards, positions = etat_du_buffer("replay_buffer")
    if shards:
        print(f"[Buffer] {len(shards)} shards, {positions} positions, "
              f"du {date_du_shard(shards[0])} au {date_du_shard(shards[-1])} "
              f"(plafond {args.buffer})")
    else:
        print(f"[Buffer] vide (plafond {args.buffer})")

    from train_self_play import STOCKFISH_PATH, pipeline

    pipeline(
        num_iterations=args.iterations,
        games_per_iter=args.games,
        concurrent_games=args.concurrent,
        slow_sims=args.slow_sims,
        fast_sims=args.fast_sims,
        slow_ratio=0.25,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_buffer_size=args.buffer,
        target_sampling_ratio=args.sampling_ratio,
        eval_stockfish_every=args.eval_every,
        checkpoint_path=(args.checkpoint
                         or "checkpoints/2026_04_30_09h53_iter436_unsupervised.pt"),
        stockfish_path=args.stockfish or STOCKFISH_PATH,
        stockfish_elo=args.stockfish_elo,
        stockfish_nodes=200_000,
    )


if __name__ == "__main__":
    main()
