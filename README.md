# LapZero-Chess

A UCI chess engine with a learned evaluation, written from scratch and trained
on a single consumer GPU. It speaks the UCI protocol, so it can be plugged into
any chess interface that supports UCI engines (Arena, Cute Chess, Nibbler, or
the Pygame GUI in this repository). It also plays on Lichess as
[Mboobot](https://lichess.org/@/mboobot) (around 2300 rapid).

The engine implements its own board, move generation and rules. There is no
chess library, no opening book and no heuristic evaluation inside the engine;
Stockfish is used only as an external reference opponent in evaluation runs.
The third-party components are ONNX Runtime for inference and PyTorch with
Lightning for training.

## Repository layout

- `src/`: C++17 engine. Board, move generation, rules, Zobrist hashing, MCTS
  with transposition table, batched GPU inference, self-play manager, PGN
  parser, pybind11 bindings. Move generation runs at about 1.8M nodes per
  second in perft, single-threaded, each node including legal move generation,
  move execution and unmake.
- `python_src/`: model, training scripts, dataset tools, tournament with Whole
  History Rating, Stockfish anchor, puzzle benchmark, UCI adapter (`uci.py`),
  Pygame GUI. The compiled engine module is
  `python_src/chess_engine.cp313-win_amd64.pyd`, built from `src/`.
- `tests/`: C++ tests and benchmarks. `python_src/tests/`: Python tests.
- `docs/`: engineering reports, specs, training history, technical backlog.

## Model

- ResNet: 10 residual blocks, 128 filters, Squeeze-and-Excitation blocks.
- Input: 119 planes of 8x8 (piece placement, 8-ply history, repetition,
  castling rights, move counters).
- Outputs: policy over 4672 moves, value in [-1, 1].
- Inference with ONNX Runtime, on GPU (CUDA) or CPU.

Differences from the AlphaZero paper: supervised initialization instead of
random weights, AdamW instead of SGD with momentum, self-play that mixes fast
moves (100 simulations) and slow moves (700), unvisited tree nodes inheriting
their parent's value (FPU, as in Leela Chess Zero), 20% of self-play games
started from a Lichess puzzle position with a deeper first search, history
dropout on 1% of moves, and a smaller network (10 blocks of 128 filters against
20 of 256) to fit one GPU.

## Training phases

The current model was trained in
April 2026 in three phases.

| Phase                  | Data                                                                         | Recipe                                                                                                     | Steps  |
| ---------------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | ------ |
| Supervised pretraining | Lichess 2026-02 dump, both players >= 1800 Elo, ~1.2M games, 39.9M positions | 2 epochs, batch 4096, Adam 1e-3                                                                            | 19,495 |
| Supervised fine-tuning | pgnmentor grandmaster games, ~210k games, 19.5M positions                    | 2 epochs, batch 2048, Adam 1e-3                                                                            | 18,996 |
| Self-play              | 211k games generated, 5.93M training positions                               | 436 iterations, 512 games per iteration, 700 slow and 100 fast simulations per move, AdamW 4e-5 at the end | 26,356 |

Self-play keeps a replay buffer of 750k positions on disk and samples 14 times
the volume of newly generated positions at each iteration. Every 8 iterations,
the model plays 16 games against Stockfish 2600 limited to 200k nodes to
estimate its level.

Training history and uncertainties: `docs/training-history.md`. Curves and
detailed figures: `docs/rapports/2026-04-entrainement-iter436/rapport.pdf`.

## History

Main milestones. The full list, tagged by area, is in `CHANGELOG.md`.

- 2022-10: first commits, board and basic rules in C++.
- 2026-02: project resumed after a pause of three years.
- 2026-03: PGN pipeline, AlphaZero input encoding, first network, C++ MCTS with
  transposition table, first self-play runs, multiprocess game generation, Elo
  tournament.
- 2026-04: first Lichess bot, Squeeze-and-Excitation blocks, GPU-batched
  self-play, 436 self-play iterations.
- 2026-08: perft validation suite and differential fuzzer, uv environment,
  Lichess puzzle pipeline carrying the real move history of each puzzle, puzzle
  benchmark.
- 2026-09: batched MCTS with virtual loss and multicore waves, 2 to 3 times
  faster search; semantic transposition table key; self-play pool kept full,
  1.22 times faster generation.

## Build

Prerequisites: Windows, CMake >= 3.15, MSVC with C++17, Python 3.13
(`CMakeLists.txt` requires it and the compiled module is a cp313), CUDA 12.x
with cuDNN for GPU inference, and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cmake -S . -B build -DPython3_EXECUTABLE=.venv/Scripts/python.exe
cmake --build build --config Release
```

The build downloads ONNX Runtime and pybind11, then produces the `chess_engine`
module in `python_src/` and `chess_perft` in `build/Release`. For a custom
interpreter, set `Python3_EXECUTABLE` in a `CMakeUserPresets.json` at the root.

## Usage

Run the training, tournament and GUI scripts from `python_src`: they use paths
relative to that directory, such as `checkpoints/` and `replay_buffer/`.

| Command                                    | Description                          |
| ------------------------------------------ | ------------------------------------ |
| `uv run python train_supervised.py`        | supervised training                  |
| `uv run python train_self_play.py`         | self-play training loop              |
| `uv run python tournament_elo.py`          | tournament with Whole History Rating |
| `uv run python play_against_bot.py`        | Pygame GUI to play against a model   |
| `python uci.py`                            | UCI engine, for any chess interface  |
| `run_lichess_bot.bat`                      | Lichess bot, from any directory      |

`train_supervised.py` still holds the dataset and checkpoint paths of the
original machine (`C:/Users/M47h1/...`); update them before a supervised pass.

For an external chess interface (Arena, Cute Chess, Nibbler, and others),
configure the engine with the virtual environment interpreter and `uci.py`:

```bash
.venv\Scripts\python.exe python_src\uci.py
```

Absolute paths work as well. The model and the per-move log are resolved
relative to `uci.py`, so the working directory set by the interface does not
matter.

The bot is a vendored copy of
[lichess-bot](https://github.com/lichess-bot-devs/lichess-bot) (AGPLv3) in
`python_src/lichess_bot/`, configured to use `python_src/uci.py` as the engine.
`config.yml` holds the OAuth token and is not versioned. The bot accepts blitz
and rapid games of 3 to 10 minutes with at most 2 seconds of increment,
prefers bot opponents, and saves its games as PGN in
`python_src/lichess_bot/game_records/`.

## Position bench

The periodic assessment is a frozen set of external positions, not a match.
The `v1` dataset holds 10,000 positions taken from Lichess broadcast archives
(CC BY-SA 4.0), each with its real game history and every legal move annotated
offline by Stockfish (WDL and centipawns). A fixed sub-bench of 256 positions
runs a 384-simulation MCTS. The main metric is the expected policy regret over
the Stockfish outcome expectation; the soft duration target is about five
minutes per pass. Training calls the bench every 4 iterations by default
(`--eval-every`, `0` disables it, `--position-bench` and `--eval-output-dir`
override the default paths). The bench is forbidden as training data.

Build the dataset once (offline, network and Stockfish required), then qualify
it and evaluate checkpoints. Run these from the repository root:

```bash
uv run python python_src/build_position_bench.py build --months 2026-08 2026-07 2026-06 --stockfish <stockfish.exe> --workers 4 --work-dir data/position_bench_work --output data/position_bench/v1 --resume
uv run python python_src/position_bench.py validate --bench data/position_bench/v1
uv run python python_src/position_bench.py evaluate --model <model.onnx> --bench data/position_bench/v1 --output-dir position_bench_results/qualification-1 --iteration 680
uv run python python_src/position_bench.py compare --current position_bench_results/qualification-1/<model>.npz --previous position_bench_results/<ref>/<model>.npz
uv run python python_src/position_bench.py reaggregate --result <result.npz> --bench data/position_bench/v1 --output metrics.json
uv run python python_src/position_bench.py profile --candidates data/position_bench_work/candidates.jsonl.zst --model <model.onnx> --search-count 32 --output-dir position_bench_results
```

Each result is a compressed NPZ plus a JSON sidecar holding the model, dataset,
C++ module, ONNX Runtime and protocol hashes; comparisons are paired by
position and refuse different datasets or protocols. A 340-second pass is a
valid pass: `over_target` is logged, nothing is shortened. The bench measures
agreement with Stockfish on external positions; it is not an Elo rating. The
periodic Stockfish match is no longer started by the training loop;
`stockfish_player.py` remains the manual anchor tool.

If a reserve or final selection cannot satisfy its quotas, `build` automatically
adds the preceding archive month and retries, keeping all selection constraints.
It never goes earlier than March 2026, to avoid the pretraining corpus. Effective
months are saved in `build_months.json` so `--resume` with the same initial months
also resumes an automatically extended build. Keep the entire work directory:
per-month candidates and the Stockfish SQLite cache are reused, and only missing
analyses are run. Use the same Stockfish binary and analysis configuration.

The quotas approved on October 1, 2026, before the first `v1` publication are:

| Phase | Contested | Advantage | Decisive | Total |
|---|---:|---:|---:|---:|
| Opening | 1,100 | 500 | 400 | 2,000 |
| Middlegame | 3,600 | 1,000 | 1,400 | 6,000 |
| Endgame | 1,250 | 250 | 500 | 2,000 |

If the full reserve is already in `annotated.jsonl.zst`, finish directly with
the following command. This reuses its annotations and runs the 500-position
audit before publication; `build` would instead reconstruct the reserve.

```bash
uv run python python_src/build_position_bench.py finalize --stage final --stockfish <stockfish.exe> --workers 8 --work-dir data/position_bench_work --output data/position_bench/v1 --resume
```

The first published `v1` explicitly accepts the original audit warning
(94.2% acceptable best-move regrets, mean score variation 0.02036). Its manifest
keeps `audit.passed = false` and `audit.accepted_with_warning = true`. The 500
cached 400,000-node analyses were promoted before reselecting the dataset;
497 remain in the final 10,000 positions. All 256 MCTS reference positions are
selected by the usual stratified hash from those 497. This does not change
LapZero's 384-simulation search budget. No independent post-repair audit is
claimed, and the original reserve remains untouched.

To reproduce that explicitly approved publication from a complete audit cache:

```bash
uv run python python_src/build_position_bench.py finalize --stage final --repair-from-cached-audit --stockfish <stockfish.exe> --work-dir data/position_bench_work --output data/position_bench/v1 --resume
```

This mode refuses missing audit entries instead of launching new analyses.
`annotation_budget_nodes` records each published position's requested teacher
budget; each move's `nodes` field records the nodes actually examined.

On a Windows SSH host, `python_src/dev_tools/run_position_bench_finalize.ps1`
can supervise this command with explicit `-Root`, `-Python`, `-Stockfish` and
`-JobDirectory` absolute paths. Launch the supervisor through a one-off Windows
Task Scheduler task without a recurring trigger. A plain `Start-Process` from
SSH may be killed when that SSH connection closes. The supervisor records
`status.json` (Python PID and exit code),
`stdout.log` and `stderr.log` in a new job directory. Progress is also written
to `data/position_bench_work/audit.progress`. A completed audit is not sufficient
by itself: check the supervisor's exit code and the published manifest.

## Testing

`chess_perft` validates the move generator against the six standard perft
positions of the
[Chess Programming Wiki](https://www.chessprogramming.org/Perft_Results), with
reference counts hardcoded so the suite needs no external library.

```bash
./build/Release/chess_perft.exe bench --strict --check-fen   # depths 1-3, instant
./build/Release/chess_perft.exe deep --strict                # up to depth 6, ~6 min
./build/Release/chess_perft.exe divide startpos 5            # per-root-move breakdown
```

`--strict` also checks that `encodeMove` loses no legal move and produces
unique in-range policy indices, and that `hasAnyLegalMove` agrees with
`getAllLegalMoves`. `--check-fen` checks that `loadFEN(toFEN(b))` returns the
same Zobrist hash.

`dev_tools/fuzz_movegen.py` plays random legal games and compares the legal
move set against `python-chess` at every ply. It is a diagnostic tool, outside
the validation path, and can bisect to the position where a perft count
diverges.

```bash
python python_src/dev_tools/fuzz_movegen.py --bisect "<fen>" 5
```

`python_src/puzzle_bench.py` measures search and evaluation quality on Lichess
puzzle positions with known solutions. `docs/backlog.md` lists the open items.
