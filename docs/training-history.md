# Training history of the iter436 checkpoint

Reconstruction of the training that produced
`checkpoints/2026_04_30_09h53_iter436_unsupervised.pt`, the model used by
`uci.py` through `2026_04_30_09h53_iter436_unsupervised.onnx`.

Figures come from the checkpoint metadata (Lightning loop state, `global_step`,
`iteration`), the scripts in `python_src/`, the archived datasets, and the wandb
project `leclercqmathieu14/alphazero-chess`. The points that could not be
verified exactly are listed at the end.

## Lineage

The current network is the second generation of the architecture. The first one
trained from 2026-03-07 to 2026-04-04, then the model was replaced on 2026-04-06
by a Squeeze-and-Excitation variant with wider heads (commit `75977ce`).

A weight transfer from the old model was prepared (`transfer_weights.py`, output
`modele_generation0_init.pt`: 153 of 187 tensors copied bit for bit, 34 left
random) and two short Lichess runs did use it on the evening of 2026-04-06. Both
were abandoned, and the pretraining the current model descends from was
restarted from random weights. The wandb evidence below shows the initial top-1
policy accuracy of each run (39% with transferred weights, 5% from random).

## 1. Supervised pretraining, Lichess

| item | value |
|---|---|
| Date | 2026-04-06, run started 19:33 local (17:33 UTC) |
| Init | random weights (`jhqhhx5q` run, initial top-1 accuracy 5%) |
| Data | Lichess standard rated dump 2026-02, both players >= 1800 Elo |
| Games | ~1.2M, 2.37 GB PGN (`top_games_quality.pgn`) |
| Shards | ~2,400 files of 500 games, ~16.6k positions each (~39.9M positions) |
| Batch size | 4096 |
| Optimizer | Adam, lr 1e-3, 16-mixed |
| Steps | 19,495 = 2 epochs of ~9,750 batches |
| Samples seen | ~79.8M |
| Result | `2026_04_06_22h25_SUPERVISED_LICHESS_NEW_MODEL.ckpt` |

### Wandb evidence for that evening

Three `supervised_phase_2_lichess` runs were launched in a row:

| local time | wandb id | init | first logged top-1 accuracy | outcome |
|---|---|---|---|---|
| 18:49 | `hamgvtvc` | transferred weights | 39% | stopped after ~3,200 steps, produced `alphazero-supervised-step=3000.ckpt` |
| 19:16 | `c3t86nnd` | transferred weights | 40% | stopped after ~1,800 steps |
| 19:33 | `jhqhhx5q` | random weights | 5% | ran 19,495 steps, produced the retained checkpoint |

The two transferred runs also log a value loss near 1.9, twice the ~0.9 of the
random run, which suggests the transferred value head was mismatched with the new
architecture. The retained lineage is the third run, so the current model was
trained from scratch at the supervised stage.

## 2. Supervised fine-tuning, grandmaster games

| item | value |
|---|---|
| Date | 2026-04-06 22h25 to 2026-04-07 00h05 (~1h40) |
| Data | pgnmentor player archives, one game per PGN file (`clean_pgns`) |
| Games | ~210k (19.45M positions at ~91.5 ply per game) |
| Batch size | 2048 |
| Optimizer | Adam, lr 1e-3, 16-mixed |
| Steps | 18,996 = 2 epochs of 9,498 batches |
| Samples seen | ~38.9M |
| Result | `2026_04_07_00h05_SUPERVISED_GM_NEW_MODEL.ckpt` (global step 38,491) |

## 3. Self-play reinforcement learning

| item | value |
|---|---|
| Date | 2026-04-07 00h10 to 2026-04-30 09h53 |
| Init | `2026_04_07_00h10_UNSUPERVISED_NEW_MODEL.pt`, converted from the GM fine-tune |
| Iterations | 436, over 28 runs resuming from each other |
| Optimizer steps | 26,356 |
| Games per iteration | 512 (start), 256 (iterations 27 to 37), 512 from 04-13 evening |
| Games total | ~220k |
| Search | 600 simulations per move at first, then 700 slow / 100 fast (25% slow) |
| Training | AdamW, lr 5e-6 then 5e-5 then 2e-5 then 4e-5 (from 04-14), batch 2048 then 4096, weight decay 1e-4, 16-mixed |
| Replay buffer | 750k positions on disk shards, 14x new positions sampled per iteration |
| Anchor | Stockfish 2200 at 200k nodes, raised to 2300, 2450, then 2600 for the last runs |
| Result | `2026_04_30_09h53_iter436_unsupervised.pt` |

Changes applied during the run: GPU-batched self-play (04-13), multithreaded
C++ MCTS and 200-move draws (04-14), replay buffer shards (04-15), sampling
ratio 14 (04-17), endgame slow moves (04-21), tactical puzzle injections and
300-move draws (04-23), history dropout at 1% from iteration 346 (04-27).

## Reference: previous generation

| date | step | artifact |
|---|---|---|
| 2026-03-07 | supervised, GM games only, 4,749 steps = 1 epoch (batch 4096) | `supervised_best_03_07.ckpt` |
| 2026-03-09 | Lichess shards added, up to step 9,755 | `supervised_best_03_09_lichess.ckpt` |
| 2026-03-11 | policy head fix, up to step 15,752 | `supervised_best_03_11_lichess_FIXED.ckpt` |
| 2026-03-10 to 2026-04-04 | self-play, 431 iterations, 25,429 optimizer steps | `2026_04_04_12h38_iter431_unsupervised.pt` |

This generation is the source of the weights copied by `transfer_weights.py`.
The transfer was used on 2026-04-06 by the two short runs listed above, but not
by the retained supervised lineage.

## Open points

- Lichess conversion input: the preserved `top_games_quality.pgn` holds exactly
  1,000,000 games, which is 2,000 shards of 500 games. The pretraining epoch
  length implies ~9,750 batches x 4096 = 39.9M positions, i.e. ~2,400 shards.
  The conversion input was therefore slightly larger than the preserved file, or
  the file was re-extracted with a 1M game cap afterwards.
- The archived `docs/clean_pgns.7z` holds 56,562 games, a partial snapshot from
  2026-03-03; the fine-tuning used ~210k games.
- Replay buffer shards show self-play activity until 2026-05-01, but no
  checkpoint after iter436 is present in the repository.
