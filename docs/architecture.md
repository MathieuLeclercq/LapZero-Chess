# Architecture

Vue d'ensemble technique de LapZero-Chess. Les détails chiffrés, les mesures et
l'historique des décisions sont dans `docs/backlog.md`, `docs/devlog.md`,
`docs/training-history.md`, `docs/superpowers/` et `docs/rapports/`.

## Vue d'ensemble

Deux programmes communiquent par un seul pont, le module Python `chess_engine`
compilé depuis `src/` (pybind11).

1. **Le moteur (C++)** joue aux échecs : plateau, règles, MCTS, table de
   transposition, inférence ONNX, self-play. Il ne contient aucune heuristique
   d'évaluation ni livre d'ouverture : tout vient du réseau.
2. **La boucle d'entraînement (Python)** produit les poids : import PGN,
   datasets, PyTorch/Lightning, export ONNX, tournois, bancs de mesure.

Règle de séparation : tout ce qui est chaud (recherche, construction du tenseur,
self-play) reste en C++. Python orchestre, entraîne et mesure.

## Stack technique et dépendances clés

| Composant      | Choix                                                                  |
| -------------- | ---------------------------------------------------------------------- |
| Moteur         | C++17, MSVC, CMake >= 3.15, OpenMP, Threads                            |
| Inférence     | ONNX Runtime 1.24.3 GPU (`FetchContent`, provider CUDA, repli CPU)   |
| Liaison Python | pybind11 v3.0.3 (`FetchContent`), stubs `.pyi` générés au build |
| Interpréteur  | Python 3.13 exigé par`CMakeLists.txt` (module `cp313`)            |
| Entraînement  | torch >= 2.13 (index cu126), lightning, numpy, onnx, wandb             |
| Outils         | whr (tournoi), chess (diagnostic seul), zstandard, pygame, pytest      |
| Environnement  | `uv` (`pyproject.toml` + `uv.lock`), CTest pour le C++           |

Le module compilé final est `python_src/chess_engine.cp313-win_amd64.pyd`, copié
dans `python_src/` par CMake avec les DLL ONNX.

## Structure des dossiers

```
src/                     moteur C++
  chessboard.*           plateau, règles, make/unmake, historique, tensor 119 plans
  square/piece/move.*    types de base (index de case, bitboard absent)
  zobrist.*              hachage des règles (répétition, TT moteur)
  perft.*  perft_main.cpp  validation du générateur de coups
  pgn_parser.*           lecture PGN côté C++
  evaluator.hpp          interface Evaluator (injection)
  onnx_evaluator.*       implémentation ONNX Runtime, softmax, export des timings
  evaluation_key.*       clé sémantique du cache réseau
  evaluation_cache.*     table de transposition des évaluations, striée
  mcts.*                 noeuds, UCB/FPU, sélection, expansion, backup, racine
  mcts_batch.cpp         noyau run_search (séquentiel et batché à virtual loss)
  mcts_wave.*            collecte de feuilles par vagues (multicœur)
  mcts_reservation.*     NodeState + PathReservation, propriété des feuilles
  search_children.hpp    matérialisation des enfants depuis une sonde de table
  search_terminal.hpp    valeur terminale partagée, mat prioritaire sur les nulles
  board_rollback.hpp     garde RAII : restauration du plateau sur toute sortie
  search_executor.*      pool de workers persistants, barrières de vague
  mcts_observe.cpp       compteurs, invariants, inspect_tree
  search_timing.*        chronométrages optionnels par phase
  selfplay_manager.*     parties concurrentes, pool GPU, puzzles tactiques
  bindings.cpp           surface Python du module chess_engine
  main.cpp               exécutable de smoke tests C++
python_src/              orchestration Python
  uci.py                 adaptateur UCI (thread de recherche, pondering, horloge)
  lib.py                 utilitaires, conversion de coups, export ONNX, WHR
  model.py               ChessNet (ResNet + SE), 119 plans, 4672 coups
  dataset.py sharded_dataset.py convert_pgn_to_binary.py   pipeline supervisé
  train_supervised.py    entraînement supervisé (Lightning)
  train_self_play.py     boucle self-play complète (génération, buffer, training)
  run_selftrain.py       lanceur de campagne self-play distante (itérations successives)
  convert_ckpt.py transfer_weights.py   outils de checkpoints
  stockfish_player.py    ancrage Stockfish (adversaire de référence)
  tournament_elo.py      tournoi multi-modèles avec Whole History Rating
  puzzle_bench.py        banc de puzzles (instrument de qualité principal)
  search_bench.py bench_metrics.py multicore_comparison.py hot_tree_bench.py
  tt_policy_comparison.py  instruments de débit et de compteurs
  play_against_bot.py lib_gui.py   GUI Pygame
  lichess_games.py extract_lichess_puzzle.py extract_pgn_from_lichess.py
  build_puzzle_dataset.py  pipeline puzzles Lichess avec historique réel
  dev_tools/             fuzz_movegen.py, endgame_conversion.py, selfplay_refill_bench.py
  tests/                 pytest (UCI, MCTS, TT, bancs, règles)
  checkpoints/ checkpoints_onnx/ replay_buffer/ lichess_bot/
tests/                   tests C++ (CTest) et données de référence
tests/cpp/               bancs et tests unitaires C++ autonomes
data/ training_data/     puzzles committés, PGN, shards
docs/                    backlog, devlog, superpowers (specs/plans), rapports
out/                     résultats de campagnes (JSON), non versionné
build/                   build CMake
```

## Flux de données

### 1. Préentraînement supervisé

`download_pgns.py` / `clean_pgn.py` -> `convert_pgn_to_binary.py` (tenseurs 119
plans, shards `.npz`) -> `train_supervised.py` (Lightning) -> `.ckpt` ->
`convert_ckpt.py` / `lib.export_model_to_onnx` -> ONNX.

### 2. Self-play (boucle RL)

`train_self_play.py` appelle `chess_engine.generate_self_play_games`, qui pilote
`SelfPlayManager` : N parties concurrentes (256 en production), un `MCTS`
partagé par pool de workers, inférence GPU batchée (512), bruit de Dirichlet,
slow/fast moves et injections de puzzles. Chaque partie ressort en `GameResult`
(flat states, flat policies, résultat). `convert_game_results` en fait des
shards de replay buffer ; l'itération suivante échantillonne environ 14 fois le
volume nouveau, réentraîne `ChessNet`, puis exporte l'ONNX. Tous les 8
itérations, `stockfish_player.py` mesure le niveau contre un Stockfish ancré
(2600 Elo, 200k noeuds).

### 3. Jeu

`uci.py` implémente le protocole UCI et le thread de recherche. L'interface
(Arena, Cute Chess, Nibbler, lichess-bot) envoie les commandes ; `step_analysis`
fait avancer la recherche par tranches de `BATCH_SIZE = 64` simulations, avec
`MCTS_BATCH_SIZE = 8`, `MCTS_WORKER_COUNT = 8` et le lot de forme fixe. Le même
`chess_engine` sert la GUI Pygame et `tournament_elo.py`.

### 4. Mesure et validation

`chess_perft` (CTest) valide les règles sur des comptes de référence ;
`dev_tools/fuzz_movegen.py` compare le générateur à python-chess ;
`puzzle_bench.py` mesure recherche et réseau sur 5000 puzzles avec historique
réel et comparaisons appariées (McNemar) ; `search_bench.py`,
`multicore_comparison.py` et `hot_tree_bench.py` mesurent le débit et les
compteurs ; `tournament_elo.py` donne l'Elo par WHR.

## Design patterns

- **Injection de dépendance.** `Evaluator` (`src/evaluator.hpp`) est une
  interface ; `ONNXEvaluator` en est l'implémentation réelle, les bancs C++ et
  les tests pytest injectent des faux (réseau synthétique, comptage d'appels).
- **Deux régimes de recherche qui partagent un noyau.** `run_search`
  (`mcts_batch.cpp`) a un chemin séquentiel historique (`batch_size = 0`) et un
  chemin batché à virtual loss. Les vagues multicœur se posent par-dessus avec
  `SearchExecutor`, `PathReservation` et `NodeState`.
- **Cache réseau séparé des règles.** Le Zobrist reste la clé des règles et du
  moteur ; le cache d'évaluations utilise une clé sémantique propre
  (`EvaluationCacheKey`), car le tensor 119 plans dépend de l'historique et des
  compteurs, pas seulement de la position.
- **Concurrence par propriété explicite.** Un noeud passe par
  `Unexpanded -> Pending -> Expanded/Terminal`. Une descente réserve son chemin
  et revendique la feuille : deux workers ne développent jamais le même noeud,
  et les porteurs de pointeurs survivent à tout `update_root`. Chaque entrée du
  chemin mémorise le nombre exact d'unités posées, égal à l'amplitude du virtual
  loss, et les rend telles quelles.
- **Restauration sur toute sortie.** `BoardRollback` (`src/board_rollback.hpp`)
  défait les coups d'une descente à la fermeture du scope, exception comprise ;
  `select_leaf` l'arme coup par coup, si bien qu'une erreur de l'évaluateur ne
  peut plus laisser le plateau sur une feuille.
- **Persistance à deux formats.** Checkpoints PyTorch `.pt`/`.ckpt` pour
  l'entraînement, ONNX pour l'inférence C++, shards `.npz` pour le replay
  buffer.

## Décisions structurantes

- **Moteur zero-knowledge.** Aucune fonction d'évaluation écrite à la main.
  Stockfish n'est qu'un adversaire de référence dans les campagnes, python-chess
  qu'un oracle de légalité dans les outils de diagnostic.
- **Entrée de 119 plans figée.** Placement, 8 plies d'historique, répétitions,
  droits de roque, compteurs. Conforme au papier AlphaZero (pas de plan de prise
  en passant), avec amnésie à 1 % depuis 2026-08-07. La réduction du nombre de
  plans reste une optimisation en réserve, sans décision.
- **Clé de TT sémantique.** Défaut `h0` : compteur des 50 coups exact, historique
  exclu de la clé. `legacy` (comportement d'origine sans contrôle), `h1`, `h3` et
  `h7` restent sélectionnables. La décision a été prise sur 2500 puzzles et 48 finales, et
  reste à confirmer par tournoi.
- **Virtual loss à la LC0.** `n_in_flight` n'entre que dans le dénominateur du
  terme U de l'UCB, jamais dans `q_value()`. Le FPU réduit la valeur d'un noeud
  non visité de 0.30 (défaut). L'amplitude 2 est active dans le bot, la GUI, le
  tournoi et l'ancrage Stockfish depuis le rebalayage du 2026-09-22, qui donne
  +9 à +20 % de débit et une qualité non inférieure ; le self-play garde 1, où
  elle est inerte, et la classe `MCTS` garde 1 par défaut pour ne pas fausser
  les anciens rapports.
- **Lot de forme fixe.** Chaque vague est paddée à la taille de lot avant
  l'appel ONNX Runtime, sinon le changement de forme quadruple le coût de
  `session->Run`. Les sorties dupliquées sont ignorées, la recherche est
  inchangée.
- **Recherche multicœur par vagues.** Workers persistants, collecte parallèle
  avec réservation précoce, collisions de feuilles comptées et abandonnées ; le
  débit plafonne dès 2 workers, 8 est retenu pour la marge de queue. Le gain
  vient du remplissage des lots et de la préparation CPU, pas du GPU. Les vagues
  du bot collectent plusieurs feuilles du même arbre, donc le virtual loss y
  agit, contrairement au self-play ; la qualité y est vérifiée sur le même
  échantillon de 500 puzzles que le préfiltre.
- **Le GPU est le goulet.** L'évaluateur représente 96.6 à 98.7 % du temps
  mural de la recherche. Toute optimisation doit passer par le remplissage ou
  la forme des lots, pas par la génération de coups.
- **Instrument de qualité : le banc de puzzles.** Les puzzles sont présentés
  avec leur historique réel et une TT neuve par puzzle ; les comparaisons sont
  appariées. Toute modification de MCTS, de TT ou de réseau passe par lui avant
  toute décision.
- **Taille de table.** 4 000 000 d'entrées en UCI (environ 4.16 Go) ; le banc
  utilise volontairement une table minuscule. `TTEntry` réserve 128 coups
  (1040 octets) alors que 35 suffisent, dette identifiée au backlog §2.

## Contraintes connues

- Le build est Windows-only en pratique : URL ONNX Runtime win-x64, DLL CUDA
  copiées dans `python_src/`, MSVC. Le code C++ vise C++17 portable, mais
  quelques constructions MSVC subsistent (voir backlog §9).
- `train_supervised.py` conserve des chemins de la machine d'origine
  (`C:/Users/M47h1/...`) ; le self-play, lui, reçoit ses chemins depuis Python
  depuis le 2026-09-20.
- Le self-play et le bot vivent dans deux univers de configuration : le premier
  est piloté par `train_self_play.py`, le second par les constantes en tête de
  `uci.py`. Toute activation mesurée doit être répercutée là où elle agit : le
  virtual loss, inerte en self-play, ne vit que dans les lanceurs du bot, de la
  GUI, du tournoi et de l'ancrage Stockfish.

Dettes et pistes ouvertes : `docs/backlog.md`. Journal des découvertes et des
pistes abandonnées : `docs/devlog.md`.
