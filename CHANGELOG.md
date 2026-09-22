# Improvement history

Areas: `engine` C++ core and rules, `mcts` search, `model` network and
encodings, `training` training loop, `data` datasets, `uci` UCI and Lichess
bot, `tools` benchmarks and interfaces, `build` build and dependencies,
`misc` project life.

- 2026-09-22 : `tools` perft calibrated tier (43.5M nodes) in CTest.
- 2026-09-20 : `uci` lichess-bot update, faster games, bots first.
- 2026-09-20 : `mcts` evaluation batches padded to a fixed shape.
- 2026-09-19 : `uci` per-move search log in uci.py.
- 2026-09-16 : `mcts` multicore wave-based MCTS.
- 2026-09-15 : `tools` portable, automated C++ tests.
- 2026-09-15 : `mcts` semantic transposition table key.
- 2026-09-14 : `mcts` virtual-loss batched MCTS (batch 8) and GPU enabled in uci.py.
- 2026-09-11 : `tools` search instrumentation (counters, inspect_tree, throughput bench).
- 2026-08-14 : `training` Dirichlet noise restored to normal after the tactical move.
- 2026-08-14 : `tools` runnable puzzle benchmark and reference results.
- 2026-08-08 : `tools` first puzzle benchmark.
- 2026-08-07 : `data` Lichess puzzle pipeline with real history.
- 2026-08-07 : `build` uv environment replacing requirements.txt.
- 2026-08-07 : `tools` differential movegen fuzzer against python-chess.
- 2026-08-07 : `tools` perft validation suite (bench, divide, deep, FEN round-trip check).
- 2026-04-27 : `model` history dropout (amnesia mode).
- 2026-04-26 : `uci` refined UCI time management (opening, zeitnot).
- 2026-04-23 : `training` draws at 300 plies.
- 2026-04-23 : `training` Lichess tactical puzzles injected into self-play.
- 2026-04-15 : `training` sharded replay buffer to cap RAM.
- 2026-04-14 : `mcts` multithreaded MCTS.
- 2026-04-13 : `training` GPU-batched self-play, ONNX GPU support.
- 2026-04-10 : `build` Python 3.13 bindings.
- 2026-04-07 : `uci` pondering on Lichess.
- 2026-04-06 : `model` Squeeze-and-Excitation blocks.
- 2026-04-03 : `uci` first Lichess bot online.
- 2026-04-02 : `uci` UCI wrapper.
- 2026-04-02 : `engine` FEN loading.
- 2026-03-30 : `mcts` Lc0-style FPU reduction.
- 2026-03-21 : `training` AdamW optimizer and deque replay buffer.
- 2026-03-18 : `engine` C++ make/unmake.
- 2026-03-16 : `training` fast/slow move self-play.
- 2026-03-13 : `build` Release build for the engine and the bindings.
- 2026-03-13 : `tools` Stockfish anchor in the benchmark.
- 2026-03-12 : `mcts` FPU inheriting parent value (Lc0 approach).
- 2026-03-12 : `engine` Zobrist hashing.
- 2026-03-11 : `model` fix of the policy head in use.
- 2026-03-09 : `training` multiprocess self-play.
- 2026-03-09 : `tools` multi-model Elo tournament.
- 2026-03-09 : `mcts` C++ MCTS with transposition table.
- 2026-03-09 : `model` ONNX Runtime inference.
- 2026-03-08 : `mcts` MCTS fix.
- 2026-03-08 : `tools` local play script and first Elo evaluation.
- 2026-03-07 : `mcts` MCTS with Dirichlet noise.
- 2026-03-03 : `model` AlphaZero board encoding (history, flip for Black).
- 2026-03-03 : `model` first network (model.py).
- 2026-03-03 : `data` PGN dataset.
- 2026-03-03 : `engine` threefold repetition.
- 2026-03-02 : `engine` 50-move rule.
- 2026-03-02 : `build` Python bindings.
- 2026-03-02 : `tools` Pygame GUI.
- 2026-03-02 : `engine` board optimization (64-square array, check detection).
- 2026-03-02 : `data` PGN parser and game extraction.
- 2026-02-20 : `misc` project resumed.
- 2022-12-08 : `engine` checkmate detection.
- 2022-11-19 : `engine` promotion.
- 2022-11-19 : `build` CMake build.
- 2022-11-06 : `engine` en passant.
- 2022-11-03 : `engine` pin detection.
- 2022-10-31 : `engine` first piece moves and basic board rules in C++.
