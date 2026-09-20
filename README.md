# LapZero-Chess

A custom chess engine built in C++ and Python.

* AlphaZero-style (MCTS + Neural Network).
* Designed for training on consumer-grade setups (single GPU).
* Supervised pretraining (Lichess & Grandmaster datasets), Unsupervised self-play.
* UCI compatible.
* [Lichess Profile (rating: around 2200 in Rapid)](https://lichess.org/@/mboobot)

![Interface graphique AlphaChess-Zero](docs/screenshots/gui.png)

## 🚀 Overview

The goal of this project was to implement a deep learning-based chess entity on a single laptop. To overcome hardware limitations, the project follows a two-stage learning process:
1. **Supervised Learning (SL):** Initializing the policy and value networks using a dataset of Grandmaster games and high-level Lichess games (freely available on [Lichess Datasets](https://database.lichess.org/)).
2. **Reinforcement Learning (RL):** Improving the model through self-play using Monte Carlo Tree Search (MCTS).

![Évolution du classement Elo des bots](docs/screenshots/bot_elo.png)

## 🆚 Differences from the Original AlphaZero

While the core architecture heavily relies on DeepMind's 2017 paper, several adaptations were made to allow efficient training on a consumer-grade laptop (RTX 3070):

- **Supervised Initialization:** Instead of starting from purely random weights (Zero-knowledge), the Policy and Value networks were pre-trained on a dataset of Grandmaster and high-level Lichess games. This massively accelerates the initial grasp of chess fundamentals.
- **Optimizer:** The original implementation used SGD with Momentum and manual step decay. This project uses **AdamW**, which provides decoupled weight decay and faster, more stable convergence for this scale.
- **Compute-Aware Self-Play (Fast/Slow Moves):** To maximize hardware efficiency, self-play games mix "fast" moves (100 MCTS simulations) and "slow" moves (700 simulations). This generates more terminal game states to train the Value head faster, while maintaining enough deep MCTS searches to provide high-quality targets for the Policy head.
- **First Play Urgency (FPU):** In DeepMind's paper, unvisited MCTS nodes are initialized with a Q-value of 0. In this engine, inheriting [LeelaChessZero](https://github.com/LeelaChessZero/lc0)'s approach, unvisited nodes inherit their parent's value. This reduces catastrophic blunders during early exploration.
- **Tactical FEN Injection:** 20% of self-play games start from a Lichess puzzle position, with a deeper first search and stronger root noise.
- **History Dropout ("amnesia mode"):** The 8-ply history stack is collapsed to a single position on 5% of moves, so the network cannot use the presence of history planes as a shortcut signal.
- **Network Size & Pipeline:** The ResNet is scaled down (10 blocks, 128 filters vs 20 blocks, 256 filters) to fit local VRAM constraints, with Squeeze-and-Excitation blocks added. The training loop is synchronous (Self-Play -> Train -> Evaluate) rather than fully asynchronous across thousands of TPUs.

## 🛠 Core Features

### ⚡ High-Performance C++ Engine
Unlike many Python-based RL projects, this engine is built in **C++17** for maximum efficiency:
- **Custom Move Generator:** No external chess libraries used. Every rule (castling, en passant, promotion) is implemented from scratch.
- **Speed:** Roughly **1.8M nodes/second** in perft, single-threaded, each node including legal move generation, move execution and unmake.
- **Pybind11 Integration:** The core logic is exposed to Python as a highly optimized module (`chess_engine`), allowing the RL loop to interact with the C++ state without overhead.

### 🧠 AlphaZero Pipeline
- **Zero-Knowledge Philosophy:** The engine provides no heuristic evaluation; the model learns purely from board geometry and game outcomes.
- **Batched MCTS:** Self-play runs many concurrent games in C++ and groups their leaf evaluations into a single GPU batch.
- **Model Architecture:** A deep Residual Convolutional Neural Network (ResNet) with Squeeze-and-Excitation blocks, Policy and Value heads.

### 📊 Evaluation & Tournament System
- **WHR (Whole History Rating):** Instead of a simple Elo, the project uses a WHR system to track the relative strength evolution of different model iterations.
- **Dynamic Tournament:** A script manages matches between bots. New models are automatically challenged by the current "Champion" to ensure accurate ranking.
- **Stockfish Anchor:** Periodic matches against a node-limited Stockfish give an absolute reference point during training.

### 🖥 GUI & Tools
- **Custom GUI:** A Pygame-based interface to play against your trained models in real-time.
- **Dataset Pipeline:** Tools to extract, clean, and shard Lichess/GM data into a binary format for high-speed training.

## 📂 Project Structure

### C++ Source (`/src`)
- `chessboard.cpp/hpp`: Core board representation, move generation and validation.
- `mcts.cpp/hpp`: Tree search, transposition table, FPU, Dirichlet noise.
- `selfplay_manager.cpp/hpp`: Concurrent game orchestration and GPU batching.
- `onnx_evaluator.cpp/hpp`: ONNX Runtime inference wrapper (CUDA or CPU).
- `perft.cpp/hpp`, `perft_main.cpp`: Move generator validation suite.
- `zobrist.cpp/hpp`: Incremental position hashing.
- `pgn_parser.cpp/hpp`: High-speed PGN/SAN string processing.
- `piece.cpp`, `square.cpp`, `move.cpp`: Atomic chess entities.
- `bindings.cpp`: Pybind11 bridge definitions.

### Python Source (`/python_src`)
- `model.py`: PyTorch implementation of the ResNet.
- `train_supervised.py`: Script for the initial imitation learning phase.
- `train_self_play.py`: The RL loop (multi-processed on CPU for game generation, GPU for training).
- `tournament_elo.py`: Tournament manager using the WHR algorithm.
- `stockfish_player.py`: Stockfish anchor evaluation.
- `uci.py`: UCI adapter, used for the Lichess bot.
- `play_against_bot.py`: Visual GUI for human-vs-bot matches.
- `lib.py`: Common utilities for move decoding and model loading.
- `dev_tools/`: Diagnostic tooling, not part of the engine or its validation.

## 🧪 Testing

`chess_perft` validates the move generator against the six standard perft positions
of the [Chess Programming Wiki](https://www.chessprogramming.org/Perft_Results),
with reference counts hardcoded so the suite needs no external library.

```bash
./build/Release/chess_perft.exe bench --strict --check-fen   # depths 1-3, instant
./build/Release/chess_perft.exe deep  --strict               # up to depth 6, ~6 min
./build/Release/chess_perft.exe divide startpos 5            # per-root-move breakdown
```

`--strict` also checks that `encodeMove` loses no legal move and produces unique
in-range policy indices, and that `hasAnyLegalMove` agrees with `getAllLegalMoves`.
`--check-fen` checks that `loadFEN(toFEN(b))` gives back the same Zobrist hash.

`dev_tools/fuzz_movegen.py` is a diagnostic companion, outside the validation path:
it plays random legal games, compares the legal move set against `python-chess` at
every ply, and can bisect to the position where a perft count diverges.

```bash
python python_src/dev_tools/fuzz_movegen.py --bisect "<fen>" 5
```

## 📈 Performance & Technical Notes

- **Independent Logic:** This project does **not** rely on `python-chess` for game simulation. All chess logic is handled by the custom C++ core.
- **Resource Optimization:** Game generation is parallelized across CPU cores using OpenMP to saturate the hardware during the Self-Play phase, while the GPU is reserved for neural network batches and backpropagation.

## 🛠 Installation & Build

### 1. Prerequisites
- **Windows** (Tested on W11)
- **CMake** (>= 3.15, the one bundled with Visual Studio works)
- **C++17 Compiler** (MSVC)
- **Python 3.13** (required: `CMakeLists.txt` asks for it and the compiled module is a `cp313`)
- **CUDA Toolkit 12.x** with cuDNN, for ONNX Runtime GPU inference
- **[uv](https://docs.astral.sh/uv/)** for the Python environment

### 2. Python Environment

Dependencies are declared in `pyproject.toml` and locked in `uv.lock`. uv installs the
right Python version on its own:

```bash
uv sync
```

This creates `.venv` with Python 3.13 and a CUDA build of PyTorch (`cu126`). To target
a different CUDA version, edit the `[[tool.uv.index]]` block in `pyproject.toml` and
re-run `uv lock`.

Without uv, install manually. Note the separate index for the CUDA build of torch:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install lightning numpy onnx chess zstandard whr wandb tqdm pygame requests urllib3 beautifulsoup4 psutil pybind11-stubgen
```

### 3. Build the C++ Engine

The build system automatically downloads ONNX Runtime and Pybind11.

```bash
cmake -S . -B build -DPython3_EXECUTABLE=.venv/Scripts/python.exe
cmake --build build --config Release
```

This generates the `chess_engine` shared library in the `python_src` folder, plus the
`chess_perft` validation executable in `build/Release`.

#### Local Configuration (IDE)
If you have multiple Python environments, create a `CMakeUserPresets.json` at the root
to point to your specific interpreter:

```
{
  "version": 3,
  "configurePresets": [
    {
      "name": "local-env",
      "displayName": "Local Environment Override",
      "cacheVariables": {
        "Python3_EXECUTABLE": "C:/path/to/your/python.exe"
      }
    }
  ]
}
```

### 🎮 Usage
- Train Supervised: ```uv run python python_src/train_supervised.py```
- Self-Play (RL): ```uv run python python_src/train_self_play.py```
- Run Tournament: ```uv run python python_src/tournament_elo.py```
- Play against Bot: ```uv run python python_src/play_against_bot.py```
- Run Lichess Bot: ```run_lichess_bot.bat```

### 🤖 Lichess Bot

The bot plays on Lichess through a vendored copy of
[lichess-bot](https://github.com/lichess-bot-devs/lichess-bot) (upstream 2026.8.9.2, AGPLv3)
in `python_src/lichess_bot/`, with `python_src/uci.py` as the engine. The configuration lives
in `python_src/lichess_bot/config.yml` (gitignored, it holds the Lichess OAuth token).

```bash
run_lichess_bot.bat    # cd's into python_src/lichess_bot, uses .venv, forwards extra args
```

Game preferences: blitz and rapid only (3 to 10 minutes, increment up to 2 s), bot opponents
first, matchmaking offers 5+0, 5+2, 10+0 and 10+2. Bot games are saved as PGN under
`python_src/lichess_bot/game_records/`.


## 🔮 Future Work & Roadmap

- **Transformer Architecture:** Exploring Attention mechanisms to replace or augment the current ResNet topology, following recent architectural shifts in modern engines like Leela Chess Zero.
- **Bitboard Representation:** Refactoring the internal board state in C++ to use bitboards. Bitwise operations would further optimize move generation speed.
