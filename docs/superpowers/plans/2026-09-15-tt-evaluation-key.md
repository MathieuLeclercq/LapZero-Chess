# TT Evaluation Key Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Corriger les collisions sémantiques du cache réseau avec un compteur des 50 coups exact et sélectionner par benchmark la profondeur historique 0 ou 1.

**Architecture:** `Chessboard` produit une empreinte qui correspond aux informations réellement envoyées au réseau. `MCTS` conserve l'index direct par Zobrist courant, mais valide l'empreinte avant d'accepter une entrée, ce qui préserve la distribution actuelle de la table et permet de mesurer les rejets. Une profondeur configurable de `-1` à `7` expose le mode historique `legacy` et les politiques expérimentales sans dupliquer les chemins séquentiel, batché ou self-play.

**Tech Stack:** C++17 avec MSVC, pybind11 3.0.3, Python 3.13, ONNX Runtime 1.24.3, pytest, CMake, uv.

**Spec:** `docs/superpowers/specs/2026-09-15-tt-evaluation-key-design.md`

## Global Constraints

- Le compteur des 50 coups est exact dans toutes les politiques sauf `legacy`.
- `cache_history_depth=-1` reproduit la clé actuelle ; `0..7` désigne le nombre de positions antérieures couvertes.
- Le défaut provisoire est `cache_history_depth=1`. La tâche 6 le remplace par 0 si et seulement si les mesures le justifient.
- Le tensor 119 x 8 x 8 et le modèle ONNX ne changent pas.
- Une position historique ne contribue que par ses 12 plans de pièces et sa catégorie de répétition, jamais par ses anciens droits de roque, son ancienne prise en passant ou son ancien trait.
- En profondeur 0, normal et amnésique partagent la même clé. À partir de 1, un bloc masqué et un bloc présent diffèrent.
- La TT reste indexée par `position_hash % m_tt_size`. L'empreinte d'évaluation sert à valider l'entrée, pas à choisir son emplacement.
- Tous les sites de lecture et d'écriture TT passent par les mêmes helpers MCTS.
- Les tests et harnais utilisent `tt_size=8192`, jamais les 2 097 143 entrées par défaut.
- Checkpoint de référence : `python_src/checkpoints/2026_04_23_23h25_iter316_unsupervised.pt`.
- `python-chess` reste limité au diagnostic de finale et à la vérification de légalité.
- Commande de build PowerShell : `& "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build build --config Release --target chess_engine chess_perft`.
- Ne jamais ajouter de ligne `Co-Authored-By` aux commits.

---

### Task 1: Construire une clé d'évaluation pure et testable

**Files:**
- Create: `src/evaluation_key.cpp`
- Create: `python_src/tests/test_evaluation_cache_key.py`
- Modify: `src/chessboard.hpp:20-120`
- Modify: `src/bindings.cpp:35-125`
- Modify: `CMakeLists.txt:32-47`

**Interfaces:**
- Produces: `EvaluationCacheKey Chessboard::getEvaluationCacheKey(int history_depth) const`
- Produces: `EvaluationCacheKey { position_hash, current_context_hash, history_hash, combined_hash, half_move_clock, repetition_category, total_moves_bucket }`
- Produces: binding Python `Chessboard.get_evaluation_cache_key(history_depth)` et `Chessboard.set_amnesia_mode(bool)`
- Consumes: `m_current_zobrist_hash`, `m_snapshotHistory`, `m_boardHistory`, `m_half_move_clock`, `m_amnesia_mode` et les constantes de `zobrist.hpp`

- [ ] **Step 1: Écrire les tests discriminants de la clé**

Créer `python_src/tests/test_evaluation_cache_key.py` avec des aides qui construisent deux transpositions légales :

```python
import os
import sys
from pathlib import Path

import numpy as np
import pytest

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE / "python_src"))
os.add_dll_directory(str(RACINE / "python_src"))

import chess_engine


def board_from_moves(moves):
    board = chess_engine.Chessboard()
    board.set_startup_pieces()
    for move in moves:
        assert board.move_piece_uci(move)
    return board


ORDRE_A = ("g1f3", "g8f6", "b1c3", "b8c6")
ORDRE_B = ("b1c3", "b8c6", "g1f3", "g8f6")


def test_compteur_exact_meme_si_le_plateau_est_identique():
    a = chess_engine.Chessboard()
    b = chess_engine.Chessboard()
    a.load_fen("8/8/8/8/8/2k5/8/R3K3 w - - 2 1")
    b.load_fen("8/8/8/8/8/2k5/8/R3K3 w - - 3 1")

    assert a.get_evaluation_cache_key(0).position_hash == b.get_evaluation_cache_key(0).position_hash
    assert a.get_evaluation_cache_key(0).combined_hash != b.get_evaluation_cache_key(0).combined_hash


def test_h0_ignore_l_historique_mais_h1_le_distingue():
    a = board_from_moves(ORDRE_A)
    b = board_from_moves(ORDRE_B)

    assert a.to_fen() == b.to_fen()
    assert a.get_evaluation_cache_key(0).combined_hash == b.get_evaluation_cache_key(0).combined_hash
    assert a.get_evaluation_cache_key(1).combined_hash != b.get_evaluation_cache_key(1).combined_hash


def test_h1_ignore_ce_qui_precede_la_position_precedente():
    a = board_from_moves(ORDRE_A + ("e2e3",))
    b = board_from_moves(ORDRE_B + ("e2e3",))

    assert a.to_fen() == b.to_fen()
    assert a.get_evaluation_cache_key(1).combined_hash == b.get_evaluation_cache_key(1).combined_hash
    assert a.get_evaluation_cache_key(3).combined_hash != b.get_evaluation_cache_key(3).combined_hash


def test_historique_ne_hash_pas_les_anciens_droits_de_roque():
    avec = chess_engine.Chessboard()
    sans = chess_engine.Chessboard()
    avec.load_fen("4k3/8/8/8/8/8/8/4K2R w K - 0 1")
    sans.load_fen("4k3/8/8/8/8/8/8/4K2R w - - 0 1")
    assert avec.move_piece_uci("e1e2")
    assert sans.move_piece_uci("e1e2")

    np.testing.assert_array_equal(
        avec.get_alphazero_tensor(), sans.get_alphazero_tensor())
    assert avec.get_evaluation_cache_key(1).combined_hash == sans.get_evaluation_cache_key(1).combined_hash


def test_amnesie_est_ignoree_en_h0_et_visible_en_h1():
    normal = board_from_moves(ORDRE_A)
    amnesique = board_from_moves(ORDRE_A)
    amnesique.set_amnesia_mode(True)

    assert normal.get_evaluation_cache_key(0).combined_hash == amnesique.get_evaluation_cache_key(0).combined_hash
    assert normal.get_evaluation_cache_key(1).combined_hash != amnesique.get_evaluation_cache_key(1).combined_hash


def test_repetition_et_numero_de_coup_suivent_exactement_le_tensor():
    board = board_from_moves(("g1f3", "g8f6", "f3g1", "f6g8"))
    key = board.get_evaluation_cache_key(0)
    tensor = board.get_alphazero_tensor()

    assert key.repetition_category == 1
    assert key.total_moves_bucket == 2
    np.testing.assert_array_equal(tensor[12], np.ones((8, 8)))
    np.testing.assert_allclose(tensor[113], 0.02)


@pytest.mark.parametrize("depth", [-1, 8])
def test_la_cle_refuse_une_profondeur_hors_bornes(depth):
    with pytest.raises(ValueError):
        board_from_moves(()).get_evaluation_cache_key(depth)
```

- [ ] **Step 2: Vérifier que les tests échouent pour la bonne raison**

Run: `uv run pytest python_src/tests/test_evaluation_cache_key.py -q`

Expected: FAIL avec `AttributeError: 'Chessboard' object has no attribute 'get_evaluation_cache_key'`, pas une erreur d'import du module.

- [ ] **Step 3: Déclarer la structure et l'API dans `chessboard.hpp`**

Ajouter avant `class Chessboard` :

```cpp
struct EvaluationCacheKey {
    uint64_t position_hash = 0;
    uint64_t current_context_hash = 0;
    uint64_t history_hash = 0;
    uint64_t combined_hash = 0;
    uint16_t half_move_clock = 0;
    uint8_t repetition_category = 0;
    uint8_t total_moves_bucket = 0;
};
```

Ajouter dans l'API publique :

```cpp
EvaluationCacheKey getEvaluationCacheKey(int history_depth) const;
```

- [ ] **Step 4: Implémenter l'empreinte dans `evaluation_key.cpp`**

Le fichier doit inclure `chessboard.hpp`, `zobrist.hpp`, `<algorithm>` et `<stdexcept>`. Utiliser une fonction de mélange déterministe SplitMix64 et des tags de domaine distincts :

```cpp
namespace {
uint64_t melanger(uint64_t seed, uint64_t value) {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    value ^= value >> 31;
    return seed ^ (value + 0x9E3779B97F4A7C15ULL +
                   (seed << 6) + (seed >> 2));
}

uint8_t categorie_repetition(const std::vector<uint64_t>& hashes,
                             size_t index) {
    int occurrences = 0;
    for (size_t i = 0; i <= index; ++i) {
        if (hashes[i] == hashes[index]) ++occurrences;
    }
    return occurrences >= 3 ? 2 : occurrences == 2 ? 1 : 0;
}
}
```

Dans `Chessboard::getEvaluationCacheKey` :

1. Refuser toute profondeur hors de `[0, 7]`.
2. Construire `all_hashes` exactement comme `getAlphaZeroTensor` : snapshots puis position courante.
3. Déduire `repetition_category` avec les mêmes seuils que les plans 12 et 13.
4. Définir `total_moves_bucket` par `min(100, m_boardHistory.size() / 2)`, exactement comme le tensor actuel, sans utiliser `m_initial_ply_offset`.
5. Construire `current_context_hash` avec le Zobrist courant, le compteur exact, la répétition courante et ce bucket.
6. Pour chaque slot `t` de 1 à `history_depth`, mettre un tag `EMPTY` si `m_amnesia_mode` est actif ou si le slot n'existe pas.
7. Sinon, retirer du Zobrist du snapshot le trait, la clé de roque et la clé en passant correspondant au snapshot. Il ne reste que l'empreinte des 12 plans de pièces. Le trait historique se déduit de `m_turn` et de la parité de `t`.
8. Mélanger ce placement avec sa catégorie de répétition.
9. Mélanger `current_context_hash`, `history_hash` et `history_depth` dans `combined_hash`.

Ne jamais itérer sur les 64 cases pour recalculer les placements historiques. Les métadonnées de `StateSnapshot` permettent de retirer en O(1) les composantes non visuelles du Zobrist existant.

- [ ] **Step 5: Exposer la clé et l'amnésie dans pybind11**

Ajouter :

```cpp
py::class_<EvaluationCacheKey>(m, "EvaluationCacheKey")
    .def_readonly("position_hash", &EvaluationCacheKey::position_hash)
    .def_readonly("current_context_hash", &EvaluationCacheKey::current_context_hash)
    .def_readonly("history_hash", &EvaluationCacheKey::history_hash)
    .def_readonly("combined_hash", &EvaluationCacheKey::combined_hash)
    .def_readonly("half_move_clock", &EvaluationCacheKey::half_move_clock)
    .def_readonly("repetition_category", &EvaluationCacheKey::repetition_category)
    .def_readonly("total_moves_bucket", &EvaluationCacheKey::total_moves_bucket);
```

Et sur `Chessboard` :

```cpp
.def("get_evaluation_cache_key", &Chessboard::getEvaluationCacheKey,
     py::arg("history_depth"))
.def("set_amnesia_mode", &Chessboard::setAmnesiaMode,
     py::arg("amnesia"))
```

Ajouter `src/evaluation_key.cpp` à `chess_core` dans `CMakeLists.txt`.

- [ ] **Step 6: Compiler et faire passer les tests purs**

Run: `& "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build build --config Release --target chess_engine`

Run: `uv run pytest python_src/tests/test_evaluation_cache_key.py python_src/tests/test_move_piece_uci.py -q`

Expected: PASS. Le test des anciens droits de roque prouve que la clé ne devient pas inutilement plus stricte que le tensor.

- [ ] **Step 7: Commit**

```bash
git add CMakeLists.txt src/chessboard.hpp src/evaluation_key.cpp src/bindings.cpp python_src/tests/test_evaluation_cache_key.py
git commit -m "Ajoute la cle semantique des evaluations"
```

---

### Task 2: Centraliser les lectures et écritures de la TT

**Files:**
- Modify: `src/mcts.hpp:40-145`
- Modify: `src/mcts.cpp:58-205,398-465`
- Modify: `src/mcts_batch.cpp:20-175`
- Modify: `src/mcts_observe.cpp:8-35`
- Modify: `src/bindings.cpp:105-160`
- Modify: `python_src/tests/test_endgame_rules.py:135-165`
- Modify: `python_src/tests/test_search_counters.py`

**Interfaces:**
- Consumes: `EvaluationCacheKey Chessboard::getEvaluationCacheKey(int)` de la tâche 1
- Produces: `MCTS(ONNXEvaluator*, size_t, int cache_history_depth = 1)`
- Produces: `TTProbe MCTS::probe_tt(const EvaluationCacheKey&) const`
- Produces: `void MCTS::record_tt_probe(TTProbeStatus)`
- Produces: compteurs `tt_position_matches`, `tt_rule50_rejects`, `tt_context_rejects`, `tt_history_rejects`

- [ ] **Step 1: Transformer le test `xfail` en exigence et ajouter le témoin legacy**

Retirer `@pytest.mark.xfail` de `test_tt_ne_reutilise_pas_une_evaluation_avec_un_autre_compteur`, puis ajouter :

```python
def test_legacy_reproduit_le_hit_avec_un_autre_compteur(evaluator):
    mcts = chess_engine.MCTS(evaluator, 8192, -1)
    mcts.mcts_search(plateau(
        "8/8/8/8/8/2k5/8/R3K3 w - - 0 1"), 0, 1.4, False, 8)
    mcts.reset_counters()

    mcts.mcts_search(plateau(
        "8/8/8/8/8/2k5/8/R3K3 w - - 99 1"), 0, 1.4, False, 8)
    c = mcts.get_counters()

    assert c.tt_hits == 1
    assert c.tt_rule50_rejects == 0


def test_compteur_different_est_un_rejet_mesure(evaluator):
    mcts = chess_engine.MCTS(evaluator, 8192, 0)
    mcts.mcts_search(plateau(
        "8/8/8/8/8/2k5/8/R3K3 w - - 0 1"), 0, 1.4, False, 8)
    mcts.reset_counters()

    mcts.mcts_search(plateau(
        "8/8/8/8/8/2k5/8/R3K3 w - - 99 1"), 0, 1.4, False, 8)
    c = mcts.get_counters()

    assert c.tt_hits == 0
    assert c.tt_misses == 1
    assert c.tt_position_matches == 1
    assert c.tt_rule50_rejects == 1
```

- [ ] **Step 2: Vérifier le RED**

Run: `uv run pytest python_src/tests/test_endgame_rules.py -q`

Expected: FAIL parce que le troisième argument et les nouveaux compteurs n'existent pas ; l'ancien test sans `xfail` doit aussi échouer.

- [ ] **Step 3: Étendre les types MCTS et TT**

Dans `mcts.hpp`, définir les constantes et l'enum avant `TTEntry` :

```cpp
static constexpr int LEGACY_CACHE_HISTORY_DEPTH = -1;
static constexpr int DEFAULT_CACHE_HISTORY_DEPTH = 1;

enum class TTProbeStatus {
    HIT,
    MISS,
    RULE50_REJECT,
    CONTEXT_REJECT,
    HISTORY_REJECT,
};

```

Étendre `TTEntry` sans modifier `legal_policy` :

```cpp
uint64_t evaluation_hash = 0;
uint64_t current_context_hash = 0;
uint64_t history_hash = 0;
uint16_t half_move_clock = 0;
```

Définir ensuite `TTProbe`, après la déclaration complète de `TTEntry` :

```cpp
struct TTProbe {
    const TTEntry* entry = nullptr;
    TTProbeStatus status = TTProbeStatus::MISS;
};
```

Ajouter `int m_cache_history_depth`, les quatre atomiques, les quatre champs dans `SearchCounters`, puis modifier le constructeur :

```cpp
MCTS(ONNXEvaluator* evaluator, size_t tt_size = DEFAULT_TT_SIZE,
     int cache_history_depth = DEFAULT_CACHE_HISTORY_DEPTH);
```

Valider dans le constructeur que la profondeur vaut `-1` ou appartient à `[0, 7]`, sinon lever `std::invalid_argument`.

- [ ] **Step 4: Implémenter un probe unique avec classification**

Ajouter les helpers privés suivants :

```cpp
EvaluationCacheKey make_cache_key(const Chessboard& board) const;
TTProbe probe_tt(const EvaluationCacheKey& key) const;
void record_tt_probe(TTProbeStatus status);
void store_tt(const EvaluationCacheKey& key,
              const std::vector<int>& legal_indices,
              const float* policy, float value);
```

`make_cache_key` renvoie une clé legacy dont les quatre hashes valent le Zobrist courant lorsque `m_cache_history_depth == -1`, sinon appelle la méthode de la tâche 1.

`probe_tt` conserve l'index direct actuel :

```cpp
const TTEntry& entry = transposition_table[
    key.position_hash % m_tt_size];
if (entry.hash != key.position_hash || entry.policy_size == 0)
    return {nullptr, TTProbeStatus::MISS};
if (m_cache_history_depth == LEGACY_CACHE_HISTORY_DEPTH)
    return {&entry, TTProbeStatus::HIT};
if (entry.half_move_clock != key.half_move_clock)
    return {nullptr, TTProbeStatus::RULE50_REJECT};
if (entry.current_context_hash != key.current_context_hash)
    return {nullptr, TTProbeStatus::CONTEXT_REJECT};
if (entry.history_hash != key.history_hash ||
    entry.evaluation_hash != key.combined_hash)
    return {nullptr, TTProbeStatus::HISTORY_REJECT};
return {&entry, TTProbeStatus::HIT};
```

`record_tt_probe` incrémente un seul résultat logique : `tt_hits` sur `HIT`, sinon `tt_misses`, plus le compteur de rejet correspondant. `tt_position_matches` augmente pour `HIT` et les trois rejets, jamais pour `MISS`.

- [ ] **Step 5: Remplacer tous les accès directs à la TT**

Remplacer les trois lectures de `mcts.cpp` par `probe_tt` :

- expansion paresseuse de `select_leaf` : enregistrer seulement un `HIT`. Ne pas compter son échec provisoire ;
- `expand_node_single` : toujours appeler `record_tt_probe`, puis inférer uniquement en cas de non-hit ;
- `advance_to_leaf` : toujours appeler `record_tt_probe`, puis retourner la feuille uniquement en cas de non-hit.

Le chemin batché n'appelle ni `expand_node_single` ni `advance_to_leaf` lorsqu'il collecte une nouvelle feuille. Dans `run_search`, juste après le garde `n_in_flight` et avant de capturer le tensor :

1. construire la clé avec `make_cache_key(board)` ;
2. refaire `probe_tt(key)` ;
3. lever `std::logic_error` si ce second probe retourne `HIT`, car `select_leaf` aurait dû matérialiser cette entrée et continuer sa descente ;
4. appeler `record_tt_probe(probe.status)` une seule fois ;
5. stocker cette même clé dans `FeuilleCollectee`.

Supprimer ensuite `m_tt_misses.fetch_add(batch_count, ...)` placé avant `evaluate_batch`. Sans cette suppression, les défauts batchés seraient comptés deux fois et les catégories de rejet ne totaliseraient pas les défauts observés.

Remplacer les deux écritures par `store_tt`. Modifier :

```cpp
void expand_and_backup_prepared(
    MCTSNode* leaf_node,
    const std::vector<int>& legal_indices,
    const EvaluationCacheKey& key,
    const float* policy,
    float value);
```

Dans `mcts_batch.cpp`, remplacer `FeuilleCollectee::hash` par :

```cpp
EvaluationCacheKey key;
```

La clé doit être capturée pendant que le plateau est encore sur la feuille, au même moment que le tensor et les coups légaux. Cela évite de la recalculer après les `undoMove()`.

Dans `test_search_counters.py`, paramétrer le test `tt_misses == nn_calls` sur `batch_size in [0, 8]`, en passant réellement ce batch à `step_analysis`. Ajouter pour chaque cas :

```python
assert c.tt_misses == c.nn_calls
assert c.tt_position_matches == (
    c.tt_hits + c.tt_rule50_rejects
    + c.tt_context_rejects + c.tt_history_rejects)
```

Ce test interdit notamment de conserver l'incrément groupé historique des défauts dans `mcts_batch.cpp`.

- [ ] **Step 6: Exposer constructeur et compteurs**

Modifier le binding :

```cpp
.def(py::init<ONNXEvaluator*, size_t, int>(),
     py::arg("evaluator"),
     py::arg("tt_size") = 2097143,
     py::arg("cache_history_depth") = DEFAULT_CACHE_HISTORY_DEPTH)
```

Exposer les quatre nouveaux champs de `SearchCounters`. Les ajouter aussi à `get_counters()` et `reset_counters()` dans `mcts_observe.cpp`.

- [ ] **Step 7: Compiler et faire passer les régressions ciblées**

Run: `& "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build build --config Release --target chess_engine`

Run: `uv run pytest python_src/tests/test_endgame_rules.py python_src/tests/test_search_counters.py python_src/tests/test_virtual_loss.py -q`

Expected: PASS, aucun `xfail`. Vérifier explicitement que `tt_misses == nn_calls` reste vrai sur les positions non terminales du test existant.

- [ ] **Step 8: Commit**

```bash
git add src/mcts.hpp src/mcts.cpp src/mcts_batch.cpp src/mcts_observe.cpp src/bindings.cpp python_src/tests/test_endgame_rules.py python_src/tests/test_search_counters.py
git commit -m "Valide le contexte des entrees de TT"
```

---

### Task 3: Verrouiller la sémantique des profondeurs et les deux chemins de recherche

**Files:**
- Create: `python_src/tests/test_tt_history_depth.py`
- Modify: `python_src/tests/test_tree_invariants.py`
- Modify: `python_src/tests/test_virtual_loss.py`

**Interfaces:**
- Consumes: constructeur `MCTS(evaluator, tt_size, cache_history_depth)`
- Consumes: compteurs de rejet de la tâche 2
- Produces: matrice automatisée `legacy`, `h0`, `h1`, `h3`, `h7`, avec invariants exercés sur batch 0 et batch 8

- [ ] **Step 1: Écrire les tests de hit et rejet entre transpositions**

Créer `test_tt_history_depth.py` avec le fixture évaluateur déjà utilisé dans `test_endgame_rules.py`. Une recherche de zéro simulation suffit à remplir ou consulter l'entrée racine :

```python
def remplir_racine(mcts, board):
    mcts.mcts_search(board, 0, 1.4, False, 8)


def test_h0_accepte_deux_historiques_differents(evaluator):
    mcts = chess_engine.MCTS(evaluator, 8192, 0)
    remplir_racine(mcts, board_from_moves(ORDRE_A))
    mcts.reset_counters()
    remplir_racine(mcts, board_from_moves(ORDRE_B))

    c = mcts.get_counters()
    assert c.tt_hits == 1
    assert c.tt_history_rejects == 0


@pytest.mark.parametrize("depth", [1, 3, 7])
def test_historique_recent_different_est_rejete(evaluator, depth):
    mcts = chess_engine.MCTS(evaluator, 8192, depth)
    remplir_racine(mcts, board_from_moves(ORDRE_A))
    mcts.reset_counters()
    remplir_racine(mcts, board_from_moves(ORDRE_B))

    c = mcts.get_counters()
    assert c.tt_hits == 0
    assert c.tt_history_rejects == 1


def test_h1_accepte_des_historiques_anciens_differents(evaluator):
    mcts = chess_engine.MCTS(evaluator, 8192, 1)
    remplir_racine(mcts, board_from_moves(ORDRE_A + ("e2e3",)))
    mcts.reset_counters()
    remplir_racine(mcts, board_from_moves(ORDRE_B + ("e2e3",)))

    c = mcts.get_counters()
    assert c.tt_hits == 1
    assert c.tt_history_rejects == 0


def test_h3_rejette_ces_memes_historiques_anciens(evaluator):
    mcts = chess_engine.MCTS(evaluator, 8192, 3)
    remplir_racine(mcts, board_from_moves(ORDRE_A + ("e2e3",)))
    mcts.reset_counters()
    remplir_racine(mcts, board_from_moves(ORDRE_B + ("e2e3",)))

    assert mcts.get_counters().tt_history_rejects == 1
```

Importer les aides depuis `test_evaluation_cache_key.py` est interdit, pour que chaque fichier reste exécutable seul. Recopier les petites constantes et `board_from_moves`.

Les recherches à zéro simulation ci-dessus ne prétendent pas exercer le noyau batché : elles testent uniquement le probe de racine. La couverture réelle des deux noyaux vient du test `tt_misses == nn_calls` de la tâche 2 et de la matrice d'invariants ci-dessous, qui effectue 200 simulations.

Clarification issue de la revue : `h0` ignore uniquement les positions antérieures. Il doit accepter un hit lorsque le plateau, le compteur des 50 coups, la répétition courante et le plan du nombre total de coups sont identiques. Il doit encore rejeter deux plateaux identiques lorsque leur répétition courante ou leur plan du nombre de coups diffère. Ajouter un test où deux historiques de quatre demi-coups atteignent la position initiale avec le même compteur et le même bucket, mais où un seul historique constitue une répétition. Cela isole `repetition_category` sans la confondre avec `total_moves_bucket`.

- [ ] **Step 2: Ajouter les contrôles d'invariants pour chaque profondeur**

Paramétrer un test court dans `test_tree_invariants.py` :

```python
@pytest.mark.parametrize("depth", [-1, 0, 1, 3, 7])
@pytest.mark.parametrize("batch_size", [0, 8])
def test_la_politique_tt_ne_corrompt_pas_l_arbre(
        evaluateur, depth, batch_size):
    mcts = chess_engine.MCTS(evaluateur, TAILLE_TT, depth)
    mcts.step_analysis(_plateau(None), 200, 1.4, batch_size)

    rapport = mcts.inspect_tree()
    assert rapport.violations == 0, list(rapport.messages)
    assert rapport.en_vol == 0
    assert sum(s.visits for s in mcts.get_analysis_results()) == 200
```

- [ ] **Step 3: Vérifier que les tests détectent une politique mal câblée**

Avant l'exécution finale, neutraliser localement et temporairement toute la condition `HISTORY_REJECT` dans `probe_tt`. Mettre seulement `history_hash` à zéro ne suffit pas, car `evaluation_hash` est volontairement un second garde-fou. Recompiler et exécuter :

Run: `uv run pytest python_src/tests/test_tt_history_depth.py -q`

Expected: au moins `test_historique_recent_different_est_rejete` échoue. Annuler uniquement cette mutation temporaire avec `apply_patch`, jamais avec `git checkout --`.

- [ ] **Step 4: Exécuter la matrice complète ciblée**

Run: `uv run pytest python_src/tests/test_evaluation_cache_key.py python_src/tests/test_tt_history_depth.py python_src/tests/test_endgame_rules.py python_src/tests/test_virtual_loss.py python_src/tests/test_tree_invariants.py -q`

Expected: PASS, aucun skip si le checkpoint est présent, aucun `xfail`, `rapport.en_vol == 0` partout.

`rapport.en_vol == 0` est le garde-fou direct contre une fuite de virtual loss après un hit ou un rejet de TT dans le chemin batché.

- [ ] **Step 5: Commit**

```bash
git add python_src/tests/test_tt_history_depth.py python_src/tests/test_tree_invariants.py python_src/tests/test_virtual_loss.py
git commit -m "Verrouille les politiques historiques de TT"
```

---

### Task 4: Étendre les trois bancs sans dupliquer la recherche

**Files:**
- Modify: `python_src/search_bench.py`
- Modify: `python_src/puzzle_bench.py`
- Modify: `python_src/dev_tools/endgame_conversion.py`
- Modify: `python_src/bench_metrics.py`
- Modify: `python_src/tests/test_search_bench.py`
- Modify: `python_src/tests/test_bench_engine.py`
- Create: `python_src/tests/test_tt_policy_comparison.py`
- Create: `python_src/tt_policy_comparison.py`

**Interfaces:**
- Produces: CLI `search_bench.py --cache-history-depths -1 0 1 3 7`
- Produces: CLI `puzzle_bench.py --cache-history-depth DEPTH`
- Produces: CLI `endgame_conversion.py --cache-history-depths -1 0 1 3 7`
- Produces: `compare_policy_csv(inputs: dict[str, Path]) -> PolicyComparison`

- [ ] **Step 1: Écrire les tests purs du rapport de débit**

Étendre `Mesure` avec :

```python
cache_history_depth: int
tt_position_matches: int
tt_rule50_rejects: int
tt_context_rejects: int
tt_history_rejects: int
```

Écrire d'abord les assertions suivantes dans `test_search_bench.py` :

```python
def test_agreger_separe_les_profondeurs_de_cache():
    agr = agreger([
        _m(cache_history_depth=0),
        _m(cache_history_depth=1),
    ])
    assert set(agr) == {
        ("depart", "mcts_search", 0, 0),
        ("depart", "mcts_search", 0, 1),
    }


def test_rapport_affiche_les_rejets_semantiques():
    texte = format_report(agreger([_m(
        cache_history_depth=1,
        tt_position_matches=20,
        tt_rule50_rejects=3,
        tt_context_rejects=2,
        tt_history_rejects=5,
    )]), CONTEXTE, [])
    assert "h1" in texte
    assert "50 coups" in texte
    assert "historique" in texte
```

Run: `uv run pytest python_src/tests/test_search_bench.py -q`

Expected: FAIL sur la signature de `Mesure` ou les clés d'agrégation.

- [ ] **Step 2: Propager la profondeur dans `search_bench.py`**

Ajouter `cache_history_depth` à `mesurer_mcts_search` et `mesurer_step_analysis`, puis construire chaque MCTS ainsi :

```python
mcts = chess_engine.MCTS(
    evaluateur, TAILLE_TT, cache_history_depth)
```

Grouper par `(position, chemin, batch_size, cache_history_depth)`. Ajouter au rapport les quatre compteurs de classification et les ratios par consultation. Les profondeurs s'affichent `legacy`, `h0`, `h1`, etc.

La boucle CLI doit placer `cache_history_depth` à l'extérieur de la boucle des positions et à l'intérieur de chaque passage, pour que chaque mode soit mesuré dans le même ordre à chaque passage.

- [ ] **Step 3: Propager un mode unique dans le banc puzzle**

Modifier les signatures :

```python
def faire_search_fn(evaluateur, simulations: int, c_puct: float,
                    batch_size: int = 0,
                    cache_history_depth: int = 1):

def initialiser_travailleur(onnx: str, simulations: int, c_puct: float,
                            sans_historique: bool, batch_size: int,
                            cache_history_depth: int) -> None:
```

Ajouter `--cache-history-depth`, valider `-1 <= depth <= 7`, passer la valeur au constructeur et l'inscrire dans le rapport et les noms de sortie. Mettre à jour le faux MCTS de `test_bench_engine.py` pour accepter le troisième argument et vérifier sa valeur.

Supprimer du docstring de `faire_search_fn` l'affirmation devenue périmée selon laquelle la TT est indexée sur le seul Zobrist.

- [ ] **Step 4: Propager la profondeur dans le diagnostic de finale**

Ajouter `cache_history_depth` à `Resultat` et `jouer`, puis remplacer la construction par :

```python
mcts = chess_engine.MCTS(evaluator, 131071, cache_history_depth)
```

Le mode `fresh` doit recréer un MCTS avec la même profondeur. La sortie console doit afficher `legacy` ou `hN`, les hits, les trois catégories de rejet, l'issue, la longueur et `max_halfmove`.

- [ ] **Step 5: Ajouter une comparaison appariée des CSV puzzle**

Dans `tt_policy_comparison.py`, indexer chaque CSV par `ligne`, refuser les doublons, ne comparer que les lignes sans erreur présentes dans tous les fichiers, puis calculer :

```python
@dataclass(frozen=True)
class PolicySummary:
    name: str
    total: int
    solved: int
    solve_rate: float
    duration_s: float

@dataclass(frozen=True)
class PairSummary:
    left: str
    right: str
    left_only: int
    right_only: int
    move_agreement: float
    solve_rate_delta: float
    mcnemar_p: float

@dataclass(frozen=True)
class PolicyComparison:
    common_lines: int
    policies: tuple[PolicySummary, ...]
    pairs: tuple[PairSummary, ...]
```

Réutiliser `bench_metrics.mcnemar(left_only, right_only)`. Tester avec quatre lignes synthétiques que `left_only`, `right_only`, l'accord des coups et le signe de `solve_rate_delta` sont corrects. Le CLI accepte des entrées nommées répétables et écrit un rapport Markdown sans relancer aucune recherche :

```text
tt_policy_comparison.py --input legacy=legacy.csv h0=h0.csv h1=h1.csv h7=h7.csv --out-report rapport.md
```

- [ ] **Step 6: Faire passer les tests de harnais**

Run: `uv run pytest python_src/tests/test_search_bench.py python_src/tests/test_bench_engine.py python_src/tests/test_tt_policy_comparison.py -q`

Expected: PASS. Le test de bout en bout du banc puzzle doit toujours traiter six puzzles et écrire `cache_history_depth` dans son contexte.

- [ ] **Step 7: Commit**

```bash
git add python_src/search_bench.py python_src/puzzle_bench.py python_src/dev_tools/endgame_conversion.py python_src/bench_metrics.py python_src/tt_policy_comparison.py python_src/tests/test_search_bench.py python_src/tests/test_bench_engine.py python_src/tests/test_tt_policy_comparison.py
git commit -m "Mesure les politiques de cle TT"
```

---

### Task 5: Valider rapidement avant la campagne longue

**Files:**
- No planned file changes

**Interfaces:**
- Consumes: all implementation and benchmark interfaces from Tasks 1 to 4
- Produces: a green fast gate before any multi-run puzzle campaign

- [ ] **Step 1: Recompiler tous les targets affectés**

Run: `& "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build build --config Release --target chess_engine chess_perft`

Expected: exit code 0, stubs pybind régénérés.

- [ ] **Step 2: Exécuter toute la suite Python**

Run: `uv run pytest python_src/tests -q`

Expected: tous les tests passent, aucun `xfail`. Un skip est acceptable uniquement pour une ressource explicitement absente ; le checkpoint de référence étant présent sur cette machine, les tests MCTS ne doivent pas être skippés.

- [ ] **Step 3: Exécuter le filet movegen indépendant**

Run: `./build/Release/chess_perft.exe bench --strict --check-fen`

Expected: toutes les positions de référence passent, zéro violation stricte.

Run: `uv run python python_src/dev_tools/fuzz_movegen.py --positions 50000 --seed 42`

Expected: zéro divergence avec python-chess.

- [ ] **Step 4: Vérifier les invariants pour toutes les politiques**

Run:

```powershell
uv run python python_src/search_bench.py `
  --model python_src/checkpoints/2026_04_23_23h25_iter316_unsupervised.pt `
  --gpu --invariants --simulations 700 --batch-sizes 0 8 `
  --cache-history-depths -1 0 1 3 7 `
  --out-rapport "$env:TEMP\lapzero-tt-key-invariants.md"
```

Expected: zéro violation et zéro nœud en vol pour chaque couple profondeur/batch.

- [ ] **Step 5: Lancer un smoke puzzle court**

Run:

```powershell
$modes = [ordered]@{ legacy = -1; h0 = 0; h1 = 1; h3 = 3; h7 = 7 }
foreach ($name in $modes.Keys) {
  $depth = $modes[$name]
  uv run python python_src/puzzle_bench.py `
    --model python_src/checkpoints/2026_04_23_23h25_iter316_unsupervised.pt `
    --banc data/puzzles_bench.txt `
    --dossier-onnx python_src/checkpoints_onnx `
    --limite 50 --simulations 64 --travailleurs 4 --batch-size 8 `
    --cache-history-depth $depth `
    --out-csv "$env:TEMP\lapzero-tt-smoke-$name.csv" `
    --out-rapport "$env:TEMP\lapzero-tt-smoke-$name.md"
}
```

Expected: 50 lignes valides par CSV, aucune erreur de données, aucun crash de processus travailleur. Ne pas committer les CSV de smoke.

- [ ] **Step 6: Corriger toute régression par TDD puis refaire le gate entier**

Pour chaque échec, arrêter cette tâche et revenir à la tâche d'implémentation propriétaire. Ajouter ou renforcer d'abord le test qui reproduit l'échec, vérifier son échec, appliquer la correction minimale, refaire le commit concerné, puis reprendre ce gate à l'étape 1. Ne jamais affaiblir un invariant pour obtenir du vert et ne pas créer de commit propre au gate.

---

### Task 6: Mesurer, choisir `h0` ou `h1`, puis figer le défaut

**Files:**
- Create: `docs/superpowers/specs/2026-09-15-tt-key-benchmark-results.md`
- Create: `docs/superpowers/specs/2026-09-15-tt-key-search-results.md`
- Modify: `src/mcts.hpp` seulement si le défaut retenu est `h0`
- Modify: `src/bindings.cpp` si le binding ne reprend pas la constante C++
- Modify: `docs/backlog.md`
- Modify: `docs/superpowers/specs/2026-09-14-endgame-draw-audit.md`

**Interfaces:**
- Consumes: CLIs des Tasks 4 et 5
- Produces: choix documenté de `DEFAULT_CACHE_HISTORY_DEPTH`
- Produces: mesures avant/après de débit, hits, appels réseau, qualité puzzle et conversion de finale

- [ ] **Step 1: Mesurer le débit brut dans un ordre alterné**

Run:

```powershell
uv run python python_src/search_bench.py `
  --model python_src/checkpoints/2026_04_23_23h25_iter316_unsupervised.pt `
  --gpu --simulations 700 --passages 7 --batch-sizes 8 `
  --cache-history-depths -1 0 1 3 7 `
  --out-rapport docs/superpowers/specs/2026-09-15-tt-key-search-results.md
```

Le rapport doit contenir pour chaque mode la médiane et l'étendue des simulations par seconde, positions réseau/s, remplissage, taux de hits et rejets. Si l'ordre fixe montre une dérive thermique monotone, refaire avec l'ordre inversé `7 3 1 0 -1` et conserver les deux rapports.

- [ ] **Step 2: Rejouer la finale causale avec TT persistante**

Run:

```powershell
uv run python python_src/dev_tools/endgame_conversion.py `
  --model python_src/checkpoints_onnx/2026_04_23_23h25_iter316_unsupervised.onnx `
  --positions tour --simulations 700 --batch-sizes 8 `
  --cache-history-depths -1 0 1 3 7 --tree-mode reuse
```

Expected:

- `legacy` doit reproduire la nulle ou, si la trajectoire varie, conserver des rejets nuls ;
- `h0`, `h1`, `h3` et `h7` doivent mater avant 100 demi-coups ;
- aucun mode corrigé ne doit atteindre `max_halfmove=100` ;
- les compteurs doivent montrer des `tt_rule50_rejects` dans au moins un mode corrigé.

Une trajectoire stochastiquement différente n'autorise pas à supprimer le test automatisé du compteur. Le test unitaire reste la preuve déterministe.

- [ ] **Step 3: Faire un préfiltre puzzle de 200 positions**

Run:

```powershell
$modes = [ordered]@{ legacy = -1; h0 = 0; h1 = 1; h7 = 7 }
foreach ($name in $modes.Keys) {
  $depth = $modes[$name]
  uv run python python_src/puzzle_bench.py `
    --model python_src/checkpoints/2026_04_23_23h25_iter316_unsupervised.pt `
    --banc data/puzzles_bench.txt `
    --dossier-onnx python_src/checkpoints_onnx `
    --limite 200 --simulations 700 --travailleurs 16 --batch-size 8 `
    --cache-history-depth $depth `
    --out-csv "$env:TEMP\lapzero-tt-prefilter-$name.csv" `
    --out-rapport "$env:TEMP\lapzero-tt-prefilter-$name.md"
}
uv run python python_src/tt_policy_comparison.py `
  --input "legacy=$env:TEMP\lapzero-tt-prefilter-legacy.csv" `
          "h0=$env:TEMP\lapzero-tt-prefilter-h0.csv" `
          "h1=$env:TEMP\lapzero-tt-prefilter-h1.csv" `
          "h7=$env:TEMP\lapzero-tt-prefilter-h7.csv" `
  --out-report "$env:TEMP\lapzero-tt-prefilter-comparison.md"
```

Conserver les quatre CSV et rapports dans le dossier temporaire. Ils servent uniquement à décider si la campagne longue est justifiée et ne doivent pas surcharger le dépôt.

Écarter immédiatement un mode corrigé qui plante, produit des erreurs de données, ne mate pas la finale ou perd plus de 2 points absolus de réussite puzzle sur cet échantillon. `h3` n'est pas exécuté ici car il n'est pas candidat final.

- [ ] **Step 4: Demander confirmation avant la campagne puzzle complète**

Présenter les durées du préfiltre et l'estimation totale. La campagne complète comporte quatre passages de 2 500 puzzles à 700 simulations et peut approcher deux heures sur ce laptop. Ne la lancer qu'après confirmation explicite de l'utilisateur.

- [ ] **Step 5: Exécuter la campagne puzzle complète appariée**

Après confirmation, exécuter `legacy`, `h0`, `h1` et `h7` avec exactement :

```powershell
$modes = [ordered]@{ legacy = -1; h0 = 0; h1 = 1; h7 = 7 }
foreach ($name in $modes.Keys) {
  $depth = $modes[$name]
  uv run python python_src/puzzle_bench.py `
    --model python_src/checkpoints/2026_04_23_23h25_iter316_unsupervised.pt `
    --banc data/puzzles_bench.txt `
    --dossier-onnx python_src/checkpoints_onnx `
    --limite 2500 --simulations 700 --travailleurs 16 --batch-size 8 `
    --cache-history-depth $depth `
    --out-csv "$env:TEMP\lapzero-tt-full-$name.csv" `
    --out-rapport "$env:TEMP\lapzero-tt-full-$name.md"
}
uv run python python_src/tt_policy_comparison.py `
  --input "legacy=$env:TEMP\lapzero-tt-full-legacy.csv" `
          "h0=$env:TEMP\lapzero-tt-full-h0.csv" `
          "h1=$env:TEMP\lapzero-tt-full-h1.csv" `
          "h7=$env:TEMP\lapzero-tt-full-h7.csv" `
  --out-report docs/superpowers/specs/2026-09-15-tt-key-benchmark-results.md
```

Ne changer ni le modèle, ni le fichier de banc, ni l'ordre déterministe des puzzles entre les quatre passages. Écrire quatre CSV distincts, puis produire le rapport apparié avec `tt_policy_comparison.py`.

- [ ] **Step 6: Appliquer la règle de décision**

Choisir `h0` seulement si toutes les conditions suivantes sont satisfaites :

1. il passe tous les tests et mate la finale tour contre roi ;
2. son taux puzzle n'est pas inférieur de plus de 0,5 point à `h1` ;
3. McNemar entre `h0` et `h1` ne détecte pas de dégradation (`p >= 0,05` ou avantage à `h0`) ;
4. il ne perd pas plus de 0,5 point par rapport à `legacy` ;
5. son débit ou son taux de hits est strictement meilleur que `h1`.

Sinon conserver `h1`. `h7` sert à interpréter l'accord des coups et le coût de l'exactitude, jamais à remplacer automatiquement les deux candidats.

Si `h0` et `h1` échouent tous les deux au diagnostic de finale ou perdent chacun plus de 0,5 point par rapport à `legacy`, ne figer aucun défaut. Arrêter la tâche, conserver `legacy` provisoirement et diagnostiquer la cause avec un nouveau test de régression.

- [ ] **Step 7: Figer le défaut et refaire les tests ciblés**

Si `h0` gagne, modifier une seule constante :

```cpp
static constexpr int DEFAULT_CACHE_HISTORY_DEPTH = 0;
```

Le binding doit utiliser cette constante et non recopier `0` ou `1`. Exécuter :

Run: `uv run pytest python_src/tests/test_evaluation_cache_key.py python_src/tests/test_tt_history_depth.py python_src/tests/test_endgame_rules.py python_src/tests/test_virtual_loss.py python_src/tests/test_tree_invariants.py -q`

Expected: PASS.

- [ ] **Step 8: Rédiger le rapport de décision**

`2026-09-15-tt-key-benchmark-results.md` doit contenir :

- commit et modèle testés ;
- machine, provider GPU, tailles de batch, TT et simulations ;
- tableau brut des cinq politiques pour le débit ;
- catégories de rejets ;
- résultat de la finale ;
- scores puzzle et comparaisons appariées ;
- profondeur retenue et application explicite des cinq critères ;
- limites, notamment l'absence d'un tournoi Elo assez long.

Mettre à jour le backlog pour marquer la clé comme traitée et l'audit pour pointer vers les résultats.

- [ ] **Step 9: Validation finale complète**

Run: `uv run pytest python_src/tests -q`

Run: `./build/Release/chess_perft.exe bench --strict --check-fen`

Run: `uv run python python_src/dev_tools/fuzz_movegen.py --positions 200000 --seed 42`

Run: `git diff --check`

Expected: suite verte sans `xfail`, zéro violation perft, zéro divergence différentielle, zéro erreur de whitespace.

- [ ] **Step 10: Commit final**

```bash
git add src/mcts.hpp src/bindings.cpp docs/backlog.md docs/superpowers/specs/2026-09-14-endgame-draw-audit.md docs/superpowers/specs/2026-09-15-tt-key-benchmark-results.md docs/superpowers/specs/2026-09-15-tt-key-search-results.md
git commit -m "Selectionne la politique de cle TT"
```

Ajouter uniquement les fichiers de résultats réellement produits et utiles. Ne pas committer les CSV volumineux si le rapport apparié contient toutes les statistiques nécessaires à la décision.
