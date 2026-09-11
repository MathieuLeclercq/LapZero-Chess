# Pipeline de mesure de la recherche : plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Établir une référence reproductible du débit de la recherche MCTS et un filet contre la corruption silencieuse d'arbre, avant d'entreprendre le batching avec virtual loss.

**Architecture:** Deux additions C++ à `MCTS` (des compteurs atomiques, et une méthode `inspect_tree` qui parcourt l'arbre d'analyse et renvoie un rapport calqué sur `PerftReport`), un harnais Python `search_bench.py` qui mesure le débit et vérifie les invariants, et la correction de la race `parse_position` qui devient un prérequis bloquant du batching.

**Tech Stack:** C++20 avec MSVC, pybind11, Python 3.13, onnxruntime 1.24.3 CPU, pytest, CMake.

Spec de référence : `docs/superpowers/specs/2026-09-11-search-bench-design.md`.

## Global Constraints

- **Les compteurs doivent être `std::atomic<uint64_t>` en `memory_order_relaxed`.** Le self-play appelle `m_shared_mcts->advance_to_leaf` depuis une région OpenMP à 8 fils (`selfplay_manager.cpp:376-387`) sur une instance de `MCTS` partagée. Des entiers simples y seraient une course de données.
- **`inspect_tree` porte sur `m_analysis_root`, et sur lui seul.** `mcts_search` construit sa racine en variable locale (`mcts.cpp:237`) et la détruit en revenant.
- **`inspect_tree` ne doit jamais être appelé dans la boucle de mesure de débit.** Son coût croît avec la taille de l'arbre et fausserait la mesure qu'il protège.
- **`tt_size` doit valoir 8192 dans le harnais, jamais le défaut.** Une `TTEntry` pèse 1040 octets et le défaut de `MCTS.__init__` est 2 097 143 entrées, soit 2,03 Gio par instance.
- **Ne jamais écrire « nœuds par seconde » dans ce harnais.** Le perft mesure la génération de coups à plus de 1,6 million de nœuds par seconde, la recherche tourne à 375 simulations par seconde. Les trois grandeurs sont **simulations par seconde**, **inférences par seconde** et **taux de succès de la table**.
- **`use_gpu=False`.** Mesuré : le GPU n'apporte rien à batch 1 (376 sims/s contre 373 sur CPU).
- Le module compilé et ses DLL vivent dans `python_src/`. Tout script doit faire `os.add_dll_directory` sur ce répertoire avant `import chess_engine`.
- Le `.pyd` est suivi par git : il doit être commité **avec** la source C++ correspondante, jamais séparément.
- Pas de tiret cadratin dans le code, les commentaires ou les messages de commit. **Jamais de ligne `Co-Authored-By`.**
- `cmake` n'est pas dans le PATH : `"/c/Program Files/Microsoft Visual Studio/18/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe"`.
- Tests : `.venv/Scripts/python.exe -m pytest python_src/tests -q`. Filet perft : `./build/Release/chess_perft.exe bench --strict --check-fen`.

---

### Task 1: Corriger la race `parse_position` et rendre `UCIEngine` testable

**Files:**
- Modify: `python_src/uci.py:25-31` (constructeur), `python_src/uci.py:97` (début de `parse_position`)
- Test: `python_src/tests/test_uci_race.py` (créer)

**Interfaces:**
- Consomme : rien.
- Produit : `UCIEngine.__init__(self, evaluator=None, mcts=None)`. Les deux arguments valent `None` par défaut et le constructeur charge alors le modèle comme aujourd'hui. `parse_position` appelle `self.stop_search()` en première instruction.

`stop_search` existe déjà (`uci.py:245`) et joint le fil de recherche. Aujourd'hui `parse_position` appelle `self.mcts.update_root()` et réassigne `self.board` pendant que le fil tourne.

Note pour l'implémenteur : `UCIEngine()` **n'est pas constructible sur cette machine**. `MODEL_PATH` (`uci.py:14`) pointe vers `C:\Users\M47h1\Documents\chess_cpp\...`, un chemin d'une autre machine. C'est la raison de l'injection : elle rend la classe testable sans modèle, sur le modèle du `fetcher` de `lichess_games.fetch_games`.

- [ ] **Step 1: Écrire le test qui échoue**

```python
"""La race entre parse_position et le fil de recherche.

parse_position appelle update_root et reconstruit self.board. Tant que le fil
de recherche tourne, il descend dans un arbre qu'update_root est en train de
detruire. Aujourd'hui c'est un bug latent ; avec le batching la boucle
detiendra des MCTSNode* bruts pendant plusieurs millisecondes d'appel GPU, et
cela deviendra une ecriture en memoire liberee.
"""
import os
import sys

import pytest

RACINE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, RACINE)
os.add_dll_directory(RACINE)

import chess_engine
from uci import UCIEngine


class _FauxMCTS:
    """Enregistre les appels, sans moteur ni modele."""

    def __init__(self):
        self.appels = []

    def update_root(self, move_idx):
        self.appels.append(("update_root", move_idx))

    def reset_analysis(self):
        self.appels.append(("reset_analysis",))


def _moteur():
    """UCIEngine sans modele : c'est la raison d'etre de l'injection."""
    return UCIEngine(evaluator=object(), mcts=_FauxMCTS())


def test_parse_position_arrete_la_recherche_avant_de_toucher_a_l_arbre():
    moteur = _moteur()
    ordre = []
    moteur.stop_search = lambda: ordre.append("stop_search")
    moteur.mcts.update_root = lambda idx: ordre.append("update_root")

    moteur.parse_position(["startpos", "moves", "e2e4"])

    assert "stop_search" in ordre, "parse_position n'arrete pas la recherche"
    assert ordre[0] == "stop_search", f"stop_search doit venir en premier, ordre={ordre}"


def test_parse_position_arrete_la_recherche_aussi_sur_le_chemin_de_reconstruction():
    """Le chemin 3, nouvelle partie ou ponder miss, reassigne self.board."""
    moteur = _moteur()
    ordre = []
    moteur.stop_search = lambda: ordre.append("stop_search")
    moteur.mcts.reset_analysis = lambda: ordre.append("reset_analysis")

    moteur.parse_position(["startpos"])

    assert ordre[0] == "stop_search", f"ordre={ordre}"
    assert "reset_analysis" in ordre


def test_le_constructeur_accepte_des_dependances_injectees():
    faux = _FauxMCTS()
    moteur = UCIEngine(evaluator=object(), mcts=faux)

    assert moteur.mcts is faux
    assert isinstance(moteur.board, chess_engine.Chessboard)
```

- [ ] **Step 2: Lancer le test et vérifier l'échec**

Run: `.venv/Scripts/python.exe -m pytest python_src/tests/test_uci_race.py -q`
Attendu : ÉCHEC. Le constructeur n'accepte pas d'arguments, donc `TypeError: UCIEngine.__init__() got an unexpected keyword argument 'evaluator'`.

- [ ] **Step 3: Rendre le constructeur injectable**

Remplacer `python_src/uci.py:25-29` :

```python
class UCIEngine:
    def __init__(self, evaluator=None, mcts=None):
        self.board = chess_engine.Chessboard()
        # Injection pour les tests : sans elle, construire un UCIEngine exige un
        # modele ONNX, et MODEL_PATH pointe vers une autre machine.
        self.evaluator = evaluator if evaluator is not None else \
            chess_engine.ONNXEvaluator(MODEL_PATH)
        self.mcts = mcts if mcts is not None else \
            chess_engine.MCTS(self.evaluator, tt_size=4_000_000)
        self.search_thread = None
```

- [ ] **Step 4: Corriger la race**

Remplacer `python_src/uci.py:97-98` :

```python
    def parse_position(self, tokens):
        # La recherche doit etre arretee AVANT toute modification de l'arbre ou
        # du plateau. update_root detruit le reste de l'arbre par unique_ptr, et
        # le fil de recherche y descend encore. Voir
        # docs/superpowers/specs/2026-09-11-search-bench-design.md section 3.
        self.stop_search()

        moves_idx = -1
```

- [ ] **Step 5: Lancer les tests et vérifier qu'ils passent**

Run: `.venv/Scripts/python.exe -m pytest python_src/tests -q`
Attendu : toute la suite passe, les 97 tests préexistants compris.

- [ ] **Step 6: Prouver que le test mord**

Commenter la ligne `self.stop_search()` ajoutée à l'étape 4.
Run: `.venv/Scripts/python.exe -m pytest python_src/tests/test_uci_race.py -q`
Attendu : les deux tests d'ordre échouent. Rétablir ensuite la ligne.

- [ ] **Step 7: Commit**

```bash
git add python_src/uci.py python_src/tests/test_uci_race.py
git commit -m "Arrete la recherche avant de modifier l'arbre dans parse_position"
```

---

### Task 2: Les compteurs d'inférences et de succès de table

**Files:**
- Modify: `src/mcts.hpp` (structure `SearchCounters`, membres, déclarations), `src/mcts.cpp` (quatre sites d'incrémentation), `src/bindings.cpp:127-131` (après `MoveStats`)
- Create: `src/mcts_observe.cpp`
- Modify: `CMakeLists.txt:32-42` (sources de `chess_core`)
- Test: `python_src/tests/test_search_counters.py` (créer)

**Interfaces:**
- Consomme : rien.
- Produit : `chess_engine.SearchCounters` avec les propriétés en lecture seule `nn_calls`, `tt_hits`, `tt_misses`, `terminal_hits` (entiers). `MCTS.get_counters() -> SearchCounters` et `MCTS.reset_counters() -> None`.

Il y a **trois** sites de consultation de la table, pas deux. L'expansion paresseuse de `select_leaf` est celle qu'on oublie.

- [ ] **Step 1: Écrire le test qui échoue**

```python
"""Les compteurs d'instrumentation de la recherche.

Sans eux, le nombre d'inferences par seconde et le taux de succes de la table
ne sont pas observables de l'exterieur : le moteur n'avait aucune
instrumentation.
"""
import os
import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE / "python_src"))
os.add_dll_directory(str(RACINE / "python_src"))

import chess_engine

CHECKPOINT = (RACINE / "python_src" / "checkpoints"
              / "2026_04_23_23h25_iter316_unsupervised.pt")

pytestmark = pytest.mark.skipif(not CHECKPOINT.exists(),
                                reason="checkpoint absent")

TAILLE_TT = 8192


@pytest.fixture(scope="module")
def evaluateur(tmp_path_factory):
    import puzzle_bench

    chemin, _ = puzzle_bench.resoudre_modele(
        CHECKPOINT, tmp_path_factory.mktemp("onnx"))
    return chess_engine.ONNXEvaluator(str(chemin), False)


def _plateau():
    board = chess_engine.Chessboard()
    board.set_startup_pieces()
    return board


def test_les_compteurs_partent_a_zero(evaluateur):
    mcts = chess_engine.MCTS(evaluateur, TAILLE_TT)
    c = mcts.get_counters()

    assert c.nn_calls == 0
    assert c.tt_hits == 0
    assert c.tt_misses == 0
    assert c.terminal_hits == 0


def test_une_recherche_declenche_des_inferences(evaluateur):
    mcts = chess_engine.MCTS(evaluateur, TAILLE_TT)
    mcts.step_analysis(_plateau(), 50, 1.4)

    c = mcts.get_counters()

    assert c.nn_calls > 0, "aucune inference comptee"
    assert c.nn_calls <= 51, f"plus d'inferences que de simulations : {c.nn_calls}"


def test_les_consultations_de_table_sont_comptees(evaluateur):
    """Sur 400 simulations depuis la position de depart, l'arbre revisite
    forcement des positions, donc la table sert."""
    mcts = chess_engine.MCTS(evaluateur, TAILLE_TT)
    mcts.step_analysis(_plateau(), 400, 1.4)

    c = mcts.get_counters()

    assert c.tt_hits + c.tt_misses > 0
    assert 0.0 <= c.tt_hits / (c.tt_hits + c.tt_misses) <= 1.0


def test_reset_counters_remet_tout_a_zero(evaluateur):
    mcts = chess_engine.MCTS(evaluateur, TAILLE_TT)
    mcts.step_analysis(_plateau(), 20, 1.4)
    assert mcts.get_counters().nn_calls > 0

    mcts.reset_counters()

    c = mcts.get_counters()
    assert (c.nn_calls, c.tt_hits, c.tt_misses, c.terminal_hits) == (0, 0, 0, 0)


def test_les_compteurs_sont_par_instance(evaluateur):
    a = chess_engine.MCTS(evaluateur, TAILLE_TT)
    b = chess_engine.MCTS(evaluateur, TAILLE_TT)

    a.step_analysis(_plateau(), 20, 1.4)

    assert a.get_counters().nn_calls > 0
    assert b.get_counters().nn_calls == 0
```

- [ ] **Step 2: Lancer le test et vérifier l'échec**

Run: `.venv/Scripts/python.exe -m pytest python_src/tests/test_search_counters.py -q`
Attendu : ÉCHEC, `AttributeError: 'chess_engine.MCTS' object has no attribute 'get_counters'`.

- [ ] **Step 3: Déclarer la structure et les membres dans `src/mcts.hpp`**

Ajouter `#include <atomic>` aux includes, puis après la structure `MoveStats` :

```cpp
// Instrumentation de la recherche. Les compteurs sont atomiques parce que le
// self-play appelle advance_to_leaf depuis une region OpenMP a 8 fils sur une
// instance de MCTS partagee (selfplay_manager.cpp:376-387). L'ordre relache
// suffit : on ne lit ces compteurs qu'apres la recherche, et un incremente
// relache coute quelques dizaines de cycles contre 2,7 ms d'inference.
struct SearchCounters {
    uint64_t nn_calls = 0;
    uint64_t tt_hits = 0;
    uint64_t tt_misses = 0;
    uint64_t terminal_hits = 0;
};
```

Dans la section `private` de `class MCTS`, après `std::mt19937 m_noise_rng;` :

```cpp
    std::atomic<uint64_t> m_nn_calls{ 0 };
    std::atomic<uint64_t> m_tt_hits{ 0 };
    std::atomic<uint64_t> m_tt_misses{ 0 };
    std::atomic<uint64_t> m_terminal_hits{ 0 };
```

Dans la section `public`, après `void add_dirichlet_noise(...);` :

```cpp
    SearchCounters get_counters() const;
    void reset_counters();
    TreeReport inspect_tree() const;   // definie a la tache 3
```

Note : `TreeReport` est déclarée à la tâche 3. Pour que la tâche 2 compile seule, déclarer ici la structure vide et la remplir à la tâche 3 :

```cpp
struct TreeReport {
    uint64_t nodes = 0;
    uint64_t max_depth = 0;
    uint64_t violations = 0;
    std::vector<std::string> messages;
};
```

Ajouter `#include <string>` si absent.

- [ ] **Step 4: Créer `src/mcts_observe.cpp` avec les accesseurs**

```cpp
// Observabilite de la recherche : compteurs et parcours d'arbre.
//
// Separe de mcts.cpp, qui fait deja 512 lignes, et sur le modele de perft.cpp :
// un parcours de diagnostic a sa propre responsabilite et son propre fichier.

#include "mcts.hpp"

SearchCounters MCTS::get_counters() const {
    SearchCounters c;
    c.nn_calls = m_nn_calls.load(std::memory_order_relaxed);
    c.tt_hits = m_tt_hits.load(std::memory_order_relaxed);
    c.tt_misses = m_tt_misses.load(std::memory_order_relaxed);
    c.terminal_hits = m_terminal_hits.load(std::memory_order_relaxed);
    return c;
}

void MCTS::reset_counters() {
    m_nn_calls.store(0, std::memory_order_relaxed);
    m_tt_hits.store(0, std::memory_order_relaxed);
    m_tt_misses.store(0, std::memory_order_relaxed);
    m_terminal_hits.store(0, std::memory_order_relaxed);
}
```

- [ ] **Step 5: Incrémenter aux quatre sites dans `src/mcts.cpp`**

Site 1, `select_leaf`, expansion paresseuse. Après `if (entry.hash == hash && entry.policy_size > 0) {` (ligne 88), ajouter en première instruction du bloc :

```cpp
                m_tt_hits.fetch_add(1, std::memory_order_relaxed);
```

et dans la branche `else` correspondante, ou juste avant le `return` qui renvoie le nœud comme feuille :

```cpp
                m_tt_misses.fetch_add(1, std::memory_order_relaxed);
```

Site 2, `expand_node_single`, succès de table. Après `if (transposition_table[tt_idx].hash == hash && transposition_table[tt_idx].policy_size > 0) {` (ligne 163) :

```cpp
        m_tt_hits.fetch_add(1, std::memory_order_relaxed);
```

Site 3, `expand_node_single`, appel au réseau. Juste avant `m_evaluator->evaluate(...)` (ligne 176) :

```cpp
    m_tt_misses.fetch_add(1, std::memory_order_relaxed);
    m_nn_calls.fetch_add(1, std::memory_order_relaxed);
```

Site 4, le chemin batché du self-play, ligne 460. Même traitement que le site 2 pour la branche de succès, et incrémenter `m_tt_misses` sur la branche d'échec.

Enfin, dans `step_analysis` et `mcts_search`, dans la branche `if (node->is_terminal) {` :

```cpp
            m_terminal_hits.fetch_add(1, std::memory_order_relaxed);
```

- [ ] **Step 6: Ajouter le fichier aux sources CMake**

Dans `CMakeLists.txt`, remplacer la ligne `src/mcts.cpp ` de `add_library(chess_core STATIC` par :

```cmake
    src/mcts.cpp 
    src/mcts_observe.cpp
```

- [ ] **Step 7: Ajouter les bindings**

Dans `src/bindings.cpp`, après le bloc `py::class_<MoveStats>` (ligne 131) :

```cpp
    // --- Instrumentation de la recherche ---
    py::class_<SearchCounters>(m, "SearchCounters")
        .def_readonly("nn_calls", &SearchCounters::nn_calls)
        .def_readonly("tt_hits", &SearchCounters::tt_hits)
        .def_readonly("tt_misses", &SearchCounters::tt_misses)
        .def_readonly("terminal_hits", &SearchCounters::terminal_hits);
```

Et dans le bloc `py::class_<MCTS>`, après `.def("get_analysis_results", &MCTS::get_analysis_results)` :

```cpp
        .def("get_counters", &MCTS::get_counters)
        .def("reset_counters", &MCTS::reset_counters)
```

- [ ] **Step 8: Compiler et lancer les tests**

Run: `"/c/Program Files/Microsoft Visual Studio/18/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe" --build build --config Release --target chess_engine`
Attendu : compilation sans erreur, le `.pyd` est réécrit dans `python_src/`.

Run: `.venv/Scripts/python.exe -m pytest python_src/tests -q`
Attendu : toute la suite passe.

Run: `./build/Release/chess_perft.exe bench --strict --check-fen`
Attendu : `Resultat : SUCCES`.

- [ ] **Step 9: Vérifier que l'instrumentation n'a pas dégradé le self-play**

```bash
cd python_src && ../.venv/Scripts/python.exe -c "
import os, time
os.add_dll_directory(os.getcwd())
import torch, chess_engine
ev = chess_engine.ONNXEvaluator('checkpoints_onnx/2026_04_23_23h25_iter316_unsupervised.onnx', True)
t0 = time.perf_counter()
res = chess_engine.generate_self_play_games(ev, 6, 24, 8, 24, 0.5, 8192)
print(f'{len(res)} parties en {time.perf_counter()-t0:.1f} s')
"
```

Attendu : un débit du même ordre que sans compteurs, soit environ 0,9 partie par seconde sur cette machine. Les compteurs sont sur un chemin partagé : s'ils coûtaient cher, ils dégraderaient ce qu'ils mesurent.

- [ ] **Step 10: Commit**

```bash
git add src/mcts.hpp src/mcts.cpp src/mcts_observe.cpp src/bindings.cpp CMakeLists.txt python_src/chess_engine.cp313-win_amd64.pyd python_src/tests/test_search_counters.py
git commit -m "Instrumente la recherche : inferences et succes de table"
```

---

### Task 3: `inspect_tree` et le passage de découverte

**Files:**
- Modify: `src/mcts_observe.cpp` (ajout de `inspect_tree`), `src/bindings.cpp` (binding de `TreeReport`)
- Test: `python_src/tests/test_tree_invariants.py` (créer)

**Interfaces:**
- Consomme : `SearchCounters` et `mcts_observe.cpp` de la tâche 2.
- Produit : `chess_engine.TreeReport` avec les propriétés en lecture seule `nodes`, `max_depth`, `violations` (entiers) et `messages` (liste de chaînes). `MCTS.inspect_tree() -> TreeReport`.

- [ ] **Step 1: Écrire le test qui échoue**

```python
"""Invariants de l'arbre de recherche.

Le piege principal du batching ne plante pas : select_leaf fait de l'expansion
paresseuse et continue de descendre apres un succes de table, donc une feuille
deja collectee pour le GPU peut recevoir des enfants pendant la meme collecte,
apres quoi expand_and_backup en creerait un second jeu. L'arbre serait corrompu
en silence. Ces controles sont le filet.
"""
import os
import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE / "python_src"))
os.add_dll_directory(str(RACINE / "python_src"))

import chess_engine

CHECKPOINT = (RACINE / "python_src" / "checkpoints"
              / "2026_04_23_23h25_iter316_unsupervised.pt")

pytestmark = pytest.mark.skipif(not CHECKPOINT.exists(),
                                reason="checkpoint absent")

TAILLE_TT = 8192

POSITIONS = [
    ("depart", None),
    ("kiwipete",
     "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("finale", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
]


@pytest.fixture(scope="module")
def evaluateur(tmp_path_factory):
    import puzzle_bench

    chemin, _ = puzzle_bench.resoudre_modele(
        CHECKPOINT, tmp_path_factory.mktemp("onnx"))
    return chess_engine.ONNXEvaluator(str(chemin), False)


def _plateau(fen):
    board = chess_engine.Chessboard()
    if fen is None:
        board.set_startup_pieces()
    else:
        board.load_fen(fen)
    return board


def test_un_arbre_vide_ne_signale_rien(evaluateur):
    mcts = chess_engine.MCTS(evaluateur, TAILLE_TT)

    rapport = mcts.inspect_tree()

    assert rapport.nodes == 0
    assert rapport.violations == 0


@pytest.mark.parametrize("nom,fen", POSITIONS)
def test_la_recherche_sequentielle_ne_viole_aucun_invariant(evaluateur, nom, fen):
    """Le passage de decouverte : la recherche actuelle fait reference."""
    mcts = chess_engine.MCTS(evaluateur, TAILLE_TT)
    mcts.step_analysis(_plateau(fen), 400, 1.4)

    rapport = mcts.inspect_tree()

    assert rapport.violations == 0, f"{nom} : {list(rapport.messages)}"
    assert rapport.nodes > 1
    assert rapport.max_depth >= 1


def test_les_visites_de_la_racine_egalent_les_simulations(evaluateur):
    """step_analysis developpe la racine sans backup (mcts.cpp:374-377), et
    chaque simulation remonte exactement une fois par la racine. C'est le
    controle qui detectera un virtual loss mal annule."""
    mcts = chess_engine.MCTS(evaluateur, TAILLE_TT)
    board = _plateau(None)

    mcts.step_analysis(board, 200, 1.4)
    stats = mcts.get_analysis_results()
    assert sum(s.visits for s in stats) == 200

    mcts.step_analysis(board, 150, 1.4)
    stats = mcts.get_analysis_results()
    assert sum(s.visits for s in stats) == 350


def test_le_rapport_grandit_avec_le_nombre_de_simulations(evaluateur):
    petit = chess_engine.MCTS(evaluateur, TAILLE_TT)
    petit.step_analysis(_plateau(None), 50, 1.4)

    grand = chess_engine.MCTS(evaluateur, TAILLE_TT)
    grand.step_analysis(_plateau(None), 400, 1.4)

    assert grand.inspect_tree().nodes > petit.inspect_tree().nodes
```

- [ ] **Step 2: Lancer le test et vérifier l'échec**

Run: `.venv/Scripts/python.exe -m pytest python_src/tests/test_tree_invariants.py -q`
Attendu : ÉCHEC, `AttributeError: 'chess_engine.MCTS' object has no attribute 'inspect_tree'`.

- [ ] **Step 3: Implémenter `inspect_tree` dans `src/mcts_observe.cpp`**

Ajouter en tête du fichier `#include <cmath>` et `#include <functional>`, puis :

```cpp
// Nombre maximum de messages conserves, sur le modele de MAX_PERFT_MESSAGES :
// eviter de noyer la sortie quand un bug se declenche sur des milliers de noeuds.
static constexpr size_t MAX_TREE_MESSAGES = 20;

static void ajouter_violation(TreeReport& rapport, const std::string& message) {
    rapport.violations++;
    if (rapport.messages.size() < MAX_TREE_MESSAGES) {
        rapport.messages.push_back(message);
    }
}

static void visiter(const MCTSNode* node, uint64_t profondeur, TreeReport& rapport) {
    rapport.nodes++;
    rapport.max_depth = std::max(rapport.max_depth, profondeur);

    if (node->is_terminal && !node->children.empty()) {
        ajouter_violation(rapport, "noeud terminal avec des enfants");
        return;
    }

    if (node->children.empty()) return;

    // 1. Aucun enfant duplique. C'est le controle qui attrape le piege du
    // batching : une feuille collectee qui recevrait un second jeu d'enfants.
    std::vector<int> vus;
    vus.reserve(node->children.size());
    uint64_t somme_visites = 0;
    float somme_priors = 0.0f;

    for (const auto& pair : node->children) {
        const int idx = pair.first;
        const MCTSNode* enfant = pair.second.get();

        if (std::find(vus.begin(), vus.end(), idx) != vus.end()) {
            ajouter_violation(rapport, "enfant duplique, move_idx " + std::to_string(idx));
        }
        vus.push_back(idx);

        // 2. move_idx dans les bornes de la policy.
        if (idx < 0 || idx > 4671) {
            ajouter_violation(rapport, "move_idx hors bornes : " + std::to_string(idx));
        }

        // 3. Coherence du pointeur parent.
        if (enfant->parent != node) {
            ajouter_violation(rapport, "pointeur parent incoherent, move_idx " + std::to_string(idx));
        }

        somme_visites += static_cast<uint64_t>(enfant->visit_count);
        somme_priors += enfant->prior;
    }

    // 4. Conservation des visites, sous forme encadree. L'encadrement est
    // impose par l'expansion paresseuse : selon qu'un noeud a ete developpe en
    // tant que feuille (defaut de table) ou traverse en creant ses enfants au
    // vol (succes de table), il a recu ou non une visite propre.
    const uint64_t visites = static_cast<uint64_t>(node->visit_count);
    if (visites < somme_visites || visites > somme_visites + 1) {
        ajouter_violation(rapport,
            "visites hors encadrement : noeud " + std::to_string(visites) +
            ", enfants " + std::to_string(somme_visites));
    }

    // 5. Priors sommant a 1, expand_node_single divisant par sum_legal.
    if (std::fabs(somme_priors - 1.0f) > 1e-3f) {
        ajouter_violation(rapport,
            "priors ne sommant pas a 1 : " + std::to_string(somme_priors));
    }

    for (const auto& pair : node->children) {
        visiter(pair.second.get(), profondeur + 1, rapport);
    }
}

TreeReport MCTS::inspect_tree() const {
    TreeReport rapport;
    if (!m_analysis_root) return rapport;

    visiter(m_analysis_root.get(), 0, rapport);
    return rapport;
}
```

Ajouter `#include <algorithm>` pour `std::find` et `std::max`.

- [ ] **Step 4: Ajouter le binding**

Dans `src/bindings.cpp`, après le bloc `py::class_<SearchCounters>` :

```cpp
    py::class_<TreeReport>(m, "TreeReport")
        .def_readonly("nodes", &TreeReport::nodes)
        .def_readonly("max_depth", &TreeReport::max_depth)
        .def_readonly("violations", &TreeReport::violations)
        .def_readonly("messages", &TreeReport::messages);
```

Et dans le bloc `py::class_<MCTS>`, après `.def("reset_counters", &MCTS::reset_counters)` :

```cpp
        .def("inspect_tree", &MCTS::inspect_tree)
```

- [ ] **Step 5: Compiler et lancer le passage de découverte**

Run: `"/c/Program Files/Microsoft Visual Studio/18/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe" --build build --config Release --target chess_engine`
Attendu : compilation sans erreur.

Run: `.venv/Scripts/python.exe -m pytest python_src/tests/test_tree_invariants.py -q`
Attendu : tous les tests passent.

**Si un invariant échoue, ne pas l'affaiblir pour faire passer le test.** C'est le passage de découverte : soit c'est un bug latent, soit le modèle mental de la spec est faux. Dans les deux cas, s'arrêter, comprendre lequel, et rapporter avant de continuer.

- [ ] **Step 6: Prouver que le contrôle d'enfants dupliqués mord**

Ajouter temporairement, à la fin de `MCTS::inspect_tree`, juste avant `return rapport;` : rien. À la place, dans `visiter`, remplacer la condition `if (std::find(vus.begin(), vus.end(), idx) != vus.end())` par `if (false)`.

Run: puis, dans un shell Python, forger un cas dupliqué est impossible depuis l'extérieur. Procéder autrement : rétablir la condition, et vérifier que le contrôle se déclenche en changeant temporairement `vus.push_back(idx);` en `vus.push_back(idx); vus.push_back(idx);`, ce qui fait voir un doublon à chaque enfant.

Run: `.venv/Scripts/python.exe -m pytest python_src/tests/test_tree_invariants.py -q` après recompilation.
Attendu : `test_la_recherche_sequentielle_ne_viole_aucun_invariant` échoue avec des messages « enfant duplique ». Rétablir ensuite, recompiler, et revérifier que tout passe.

- [ ] **Step 7: Lancer le filet complet**

Run: `.venv/Scripts/python.exe -m pytest python_src/tests -q`
Attendu : toute la suite passe.

Run: `./build/Release/chess_perft.exe bench --strict --check-fen`
Attendu : `Resultat : SUCCES`.

- [ ] **Step 8: Commit**

```bash
git add src/mcts_observe.cpp src/bindings.cpp python_src/chess_engine.cp313-win_amd64.pyd python_src/tests/test_tree_invariants.py
git commit -m "Ajoute inspect_tree et ses invariants d'arbre"
```

---

### Task 4: Le harnais de débit

**Files:**
- Create: `python_src/search_bench.py`
- Delete: `python_src/benchmark_release_debug.py`
- Test: `python_src/tests/test_search_bench.py` (créer)

**Interfaces:**
- Consomme : `puzzle_bench.resoudre_modele(chemin, dossier_onnx) -> (Path, dict)` ; `MCTS.get_counters()`, `MCTS.reset_counters()`, `MCTS.inspect_tree()` des tâches 2 et 3.
- Produit :
  - `POSITIONS: tuple[tuple[str, str | None], ...]`, les trois positions de référence
  - `Mesure` (dataclass gelée : `position: str`, `chemin: str`, `simulations: int`, `duree_s: float`, `nn_calls: int`, `tt_hits: int`, `tt_misses: int`, `terminal_hits: int`)
  - `sims_par_seconde(m: Mesure) -> float`, `inferences_par_seconde(m: Mesure) -> float`, `taux_table(m: Mesure) -> float`
  - `agreger(mesures: list[Mesure]) -> dict[tuple[str, str], dict]` groupant par (position, chemin) avec médiane et étendue
  - `mesurer_mcts_search(mcts, board, simulations) -> Mesure` et `mesurer_step_analysis(mcts, board, simulations) -> Mesure`
  - `TAILLE_TT = 8192`

`benchmark_release_debug.py` est supprimé : il appelle `chess_engine.MCTS(ONNX_PATH)` alors que le constructeur attend un `ONNXEvaluator`, et pointe vers un ONNX absent.

- [ ] **Step 1: Écrire les tests de logique pure**

```python
"""Logique pure du harnais de debit. Aucun moteur, aucun modele."""
import os
import sys

import pytest

RACINE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, RACINE)
os.add_dll_directory(RACINE)

from search_bench import (
    Mesure,
    agreger,
    inferences_par_seconde,
    sims_par_seconde,
    taux_table,
)


def _m(position="depart", chemin="mcts_search", simulations=400, duree_s=1.0,
       nn_calls=300, tt_hits=100, tt_misses=300, terminal_hits=0):
    return Mesure(position=position, chemin=chemin, simulations=simulations,
                  duree_s=duree_s, nn_calls=nn_calls, tt_hits=tt_hits,
                  tt_misses=tt_misses, terminal_hits=terminal_hits)


def test_les_trois_grandeurs_sont_distinctes():
    """Une simulation n'est pas une inference : elle peut s'arreter sur un
    noeud terminal ou sur un succes de table."""
    m = _m(simulations=400, duree_s=2.0, nn_calls=300, tt_hits=100, tt_misses=300)

    assert sims_par_seconde(m) == pytest.approx(200.0)
    assert inferences_par_seconde(m) == pytest.approx(150.0)
    assert taux_table(m) == pytest.approx(0.25)


def test_taux_table_sans_consultation_vaut_zero():
    assert taux_table(_m(tt_hits=0, tt_misses=0)) == 0.0


def test_duree_nulle_ne_divise_pas_par_zero():
    m = _m(duree_s=0.0)

    assert sims_par_seconde(m) == 0.0
    assert inferences_par_seconde(m) == 0.0


def test_agreger_groupe_par_position_et_chemin():
    mesures = [
        _m(position="depart", chemin="mcts_search", duree_s=1.0),
        _m(position="depart", chemin="mcts_search", duree_s=2.0),
        _m(position="depart", chemin="step_analysis", duree_s=1.0),
        _m(position="finale", chemin="mcts_search", duree_s=1.0),
    ]

    agr = agreger(mesures)

    assert set(agr) == {("depart", "mcts_search"), ("depart", "step_analysis"),
                        ("finale", "mcts_search")}
    assert agr[("depart", "mcts_search")]["passages"] == 2


def test_agreger_donne_mediane_et_etendue():
    """L'etendue est la raison d'etre des passages repetes : sans elle, on ne
    sait pas distinguer un gain de 5 pour cent d'un bruit de mesure."""
    mesures = [
        _m(duree_s=1.0),   # 400 sims/s
        _m(duree_s=2.0),   # 200 sims/s
        _m(duree_s=4.0),   # 100 sims/s
    ]

    a = agreger(mesures)[("depart", "mcts_search")]

    assert a["sims_par_seconde_median"] == pytest.approx(200.0)
    assert a["sims_par_seconde_min"] == pytest.approx(100.0)
    assert a["sims_par_seconde_max"] == pytest.approx(400.0)


def test_agreger_supporte_une_liste_vide():
    assert agreger([]) == {}
```

- [ ] **Step 2: Lancer et vérifier l'échec**

Run: `.venv/Scripts/python.exe -m pytest python_src/tests/test_search_bench.py -q`
Attendu : ÉCHEC, `ModuleNotFoundError: No module named 'search_bench'`.

- [ ] **Step 3: Implémenter la logique pure et la mesure**

```python
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

# Trois positions de reference. Un debit mesure sur la seule position de depart
# ne represente pas une partie : le nombre de coups legaux, le taux de succes de
# la table et la profondeur de l'arbre y sont atypiques. Les deux dernieres sont
# reprises des positions de reference du perft, donc deja verifiees legales.
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
                        simulations: int) -> Mesure:
    """Arbre neuf a chaque appel."""
    mcts = chess_engine.MCTS(evaluateur, TAILLE_TT)
    board = charger_position(fen)
    mcts.reset_counters()

    debut = time.perf_counter()
    mcts.mcts_search(board, simulations, 1.4, False)
    duree = time.perf_counter() - debut

    c = mcts.get_counters()
    return Mesure(nom, "mcts_search", simulations, duree,
                  c.nn_calls, c.tt_hits, c.tt_misses, c.terminal_hits)


def mesurer_step_analysis(evaluateur, fen: str, nom: str,
                          simulations: int) -> Mesure:
    """Le chemin reel du bot : arbre reutilise, racine deplacee."""
    mcts = chess_engine.MCTS(evaluateur, TAILLE_TT)
    board = charger_position(fen)
    mcts.reset_analysis()
    mcts.reset_counters()

    debut = time.perf_counter()
    mcts.step_analysis(board, simulations, 1.4)
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
```

- [ ] **Step 4: Supprimer le benchmark cassé**

```bash
git rm python_src/benchmark_release_debug.py
```

- [ ] **Step 5: Lancer et vérifier**

Run: `.venv/Scripts/python.exe -m pytest python_src/tests -q`
Attendu : toute la suite passe.

- [ ] **Step 6: Commit**

```bash
git add python_src/search_bench.py python_src/tests/test_search_bench.py
git commit -m "Mesure le debit de la recherche, trois grandeurs distinctes"
```

---

### Task 5: La CLI, le rapport et les invariants

**Files:**
- Modify: `python_src/search_bench.py`
- Modify: `python_src/tests/test_search_bench.py`

**Interfaces:**
- Consomme : tout ce que produit la tâche 4.
- Produit : `format_report(agr: dict, contexte: dict, invariants: list) -> str` et `main() -> int`. Les clés attendues de `contexte` sont `modele`, `iteration`, `global_step`, `passages`, `simulations`, `c_puct`. `invariants` est une liste de tuples `(position, nodes, max_depth, violations, messages)`.

- [ ] **Step 1: Écrire les tests du rapport**

```python
from search_bench import format_report

CONTEXTE = {
    "modele": "iter316.onnx",
    "iteration": 316,
    "global_step": 19415,
    "passages": 5,
    "simulations": 400,
    "c_puct": 1.4,
}


def _agr():
    from search_bench import agreger
    return agreger([_m(duree_s=1.0), _m(duree_s=1.1)])


def test_format_report_contient_le_contexte():
    texte = format_report(_agr(), CONTEXTE, [])

    assert "iter316.onnx" in texte
    assert "19415" in texte
    assert "simulations par seconde" in texte


def test_format_report_n_ecrit_jamais_noeuds_par_seconde():
    """Le perft mesure 1,6 million de noeuds par seconde, la recherche 375
    simulations par seconde : confondre les deux serait une erreur d'un facteur
    4000."""
    texte = format_report(_agr(), CONTEXTE, []).lower()

    assert "noeuds par seconde" not in texte
    assert "nps" not in texte


def test_format_report_ne_contient_pas_de_tiret_cadratin():
    assert "\u2014" not in format_report(_agr(), CONTEXTE, [])


def test_format_report_signale_les_violations():
    invariants = [("depart", 120, 5, 3, ["enfant duplique, move_idx 42"])]

    texte = format_report(_agr(), CONTEXTE, invariants)

    assert "enfant duplique" in texte
    assert "3" in texte


def test_format_report_est_muet_quand_aucune_violation():
    invariants = [("depart", 120, 5, 0, [])]

    texte = format_report(_agr(), CONTEXTE, invariants)

    assert "enfant duplique" not in texte
    assert "aucune violation" in texte.lower()


def test_format_report_produit_des_tables_markdown_valides():
    """Garde fou repris du banc de puzzles : un separateur dont le nombre de
    cellules ne correspond pas a l'en-tete casse le rendu en silence."""
    lignes = format_report(_agr(), CONTEXTE, []).split("\n")
    separateurs = [i for i, l in enumerate(lignes)
                   if set(l) <= set("|-") and "-" in l]

    assert separateurs
    for i in separateurs:
        cols = lignes[i].count("|") - 1
        assert lignes[i - 1].count("|") - 1 == cols, lignes[i - 1]
        j = i + 1
        while j < len(lignes) and lignes[j].startswith("|"):
            assert lignes[j].count("|") - 1 == cols, lignes[j]
            j += 1
```

- [ ] **Step 2: Lancer et vérifier l'échec**

Run: `.venv/Scripts/python.exe -m pytest python_src/tests/test_search_bench.py -q`
Attendu : ÉCHEC, `cannot import name 'format_report'`.

- [ ] **Step 3: Implémenter le rapport et la CLI**

À ajouter à `python_src/search_bench.py` :

```python
_EN_TETE = (
    "| Position | Chemin | passages | sims/s (med) | sims/s (min a max) "
    "| inferences/s (med) | taux table |\n"
    "|---|---|---|---|---|---|---|"
)


def format_report(agr: dict, contexte: dict, invariants: list) -> str:
    """Rapport markdown.

    Le mot « noeuds par seconde » est proscrit : le perft mesure la generation
    de coups a plus de 1,6 million de noeuds par seconde, la recherche tourne a
    375 simulations par seconde.
    """
    lignes = [
        "# Banc de recherche : resultats",
        "",
        f"Modele : `{contexte['modele']}`, iteration {contexte['iteration']}, "
        f"global_step {contexte['global_step']}",
        f"Protocole : {contexte['passages']} passages, "
        f"{contexte['simulations']} simulations, c_puct {contexte['c_puct']}, "
        "CPU, un seul processus",
        "",
        "Trois grandeurs distinctes. Les **simulations par seconde** mesurent le",
        "debit de la recherche. Les **inferences par seconde** comptent les appels",
        "reels au reseau : une simulation qui s'arrete sur un noeud terminal ou",
        "sur un succes de table n'en coute aucune. Le **taux de table** est la",
        "part des consultations reussies.",
        "",
        "## Debit",
        "",
        _EN_TETE,
    ]

    for (position, chemin) in sorted(agr):
        a = agr[(position, chemin)]
        lignes.append(
            f"| {position} | {chemin} | {a['passages']} "
            f"| {a['sims_par_seconde_median']:.1f} "
            f"| {a['sims_par_seconde_min']:.1f} a {a['sims_par_seconde_max']:.1f} "
            f"| {a['inferences_par_seconde_median']:.1f} "
            f"| {100 * a['taux_table_median']:.1f} % |"
        )

    lignes += ["", "## Invariants d'arbre", ""]
    if not invariants:
        lignes.append("Non verifies lors de ce passage.")
    else:
        lignes += [
            "| Position | noeuds | profondeur max | violations |",
            "|---|---|---|---|",
        ]
        total = 0
        for (position, nodes, max_depth, violations, messages) in invariants:
            total += violations
            lignes.append(
                f"| {position} | {nodes} | {max_depth} | {violations} |")
        lignes.append("")
        if total == 0:
            lignes.append("Aucune violation.")
        else:
            lignes.append(f"**{total} violations.** Premiers messages :")
            lignes.append("")
            for (position, _, _, violations, messages) in invariants:
                for message in messages:
                    lignes.append(f"- `{position}` : {message}")

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
    parser.add_argument("--invariants", action="store_true",
                        help="verifie les invariants d'arbre, sans mesurer le debit")
    parser.add_argument("--out-rapport", type=Path, default=None)
    args = parser.parse_args()

    onnx, meta = puzzle_bench.resoudre_modele(args.model, args.dossier_onnx)
    if not Path(onnx).exists():
        print(f"modele ONNX introuvable : {onnx}", file=sys.stderr)
        return 2

    evaluateur = chess_engine.ONNXEvaluator(str(onnx), False)

    # Rodage : la premiere inference initialise la session.
    chauffe = chess_engine.MCTS(evaluateur, TAILLE_TT)
    chauffe.mcts_search(charger_position(POSITIONS[0][1]), 8, args.c_puct, False)

    invariants = []
    if args.invariants:
        for nom, fen in POSITIONS:
            mcts = chess_engine.MCTS(evaluateur, TAILLE_TT)
            mcts.step_analysis(charger_position(fen), args.simulations,
                               args.c_puct)
            r = mcts.inspect_tree()
            invariants.append((nom, r.nodes, r.max_depth, r.violations,
                               list(r.messages)))
            etat = "OK" if r.violations == 0 else f"{r.violations} VIOLATIONS"
            print(f"  {nom:10} : {r.nodes} noeuds, profondeur {r.max_depth}, {etat}")

    # inspect_tree n'est jamais appele dans la boucle de mesure : son cout croit
    # avec la taille de l'arbre et fausserait la mesure qu'il protege.
    mesures = []
    if not args.invariants:
        for passage in range(args.passages):
            for nom, fen in POSITIONS:
                mesures.append(
                    mesurer_mcts_search(evaluateur, fen, nom, args.simulations))
                mesures.append(
                    mesurer_step_analysis(evaluateur, fen, nom, args.simulations))
            print(f"  passage {passage + 1}/{args.passages}", flush=True)

    agr = agreger(mesures)
    for (position, chemin), a in sorted(agr.items()):
        print(f"  {position:10} {chemin:14} : "
              f"{a['sims_par_seconde_median']:7.1f} sims/s, "
              f"{a['inferences_par_seconde_median']:7.1f} inferences/s, "
              f"table {100 * a['taux_table_median']:.1f} %")

    contexte = {
        "modele": Path(onnx).name,
        "iteration": meta.get("iteration"),
        "global_step": meta.get("global_step"),
        "passages": args.passages,
        "simulations": args.simulations,
        "c_puct": args.c_puct,
    }

    sortie = args.out_rapport or Path(
        f"../docs/superpowers/specs/{time.strftime('%Y-%m-%d')}"
        "-search-bench-resultats.md")
    sortie.parent.mkdir(parents=True, exist_ok=True)
    sortie.write_text(format_report(agr, contexte, invariants), encoding="utf-8")
    print(f"\nRapport : {sortie}")

    total_violations = sum(v for (_, _, _, v, _) in invariants)
    if total_violations:
        print(f"{total_violations} violations d'invariants", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Lancer et vérifier**

Run: `.venv/Scripts/python.exe -m pytest python_src/tests -q`
Attendu : toute la suite passe.

- [ ] **Step 5: Vérifier la barrière rapide de bout en bout**

```bash
cd python_src && ../.venv/Scripts/python.exe search_bench.py \
  --model checkpoints/2026_04_23_23h25_iter316_unsupervised.pt \
  --invariants --simulations 400 \
  --out-rapport ../docs/superpowers/specs/_essai.md
```

Attendu : trois lignes `OK`, code de sortie 0. Supprimer ensuite `docs/superpowers/specs/_essai.md`.

- [ ] **Step 6: Commit**

```bash
git add python_src/search_bench.py python_src/tests/test_search_bench.py
git commit -m "Ajoute la CLI et le rapport du banc de recherche"
```

---

### Task 6: Le passage de référence

**Files:**
- Create: `docs/superpowers/specs/2026-09-11-search-bench-resultats.md`

**Interfaces:**
- Consomme : `search_bench.main` de la tâche 5.
- Produit : le rapport de référence, commité. C'est lui que le diff git comparera après le batching.

- [ ] **Step 1: Lancer le passage de débit**

```bash
cd python_src && ../.venv/Scripts/python.exe search_bench.py \
  --model checkpoints/2026_04_23_23h25_iter316_unsupervised.pt \
  --simulations 400 --passages 5
```

Attendu : quelques minutes, et un débit de l'ordre de 375 simulations par seconde sur `mcts_search`, cohérent avec les mesures ad hoc de la session précédente.

- [ ] **Step 2: Lancer le passage d'invariants et fusionner dans le même rapport**

```bash
cd python_src && ../.venv/Scripts/python.exe search_bench.py \
  --model checkpoints/2026_04_23_23h25_iter316_unsupervised.pt \
  --invariants --simulations 400 \
  --out-rapport ../docs/superpowers/specs/_invariants.md
```

Reporter à la main la section « Invariants d'arbre » de `_invariants.md` dans le rapport de référence, puis supprimer `_invariants.md`. Les deux jambes sont lancées séparément parce qu'`inspect_tree` fausserait la mesure de débit.

- [ ] **Step 3: Contrôler la cohérence avant de conclure**

Vérifier dans le rapport :

- les inférences par seconde sont **inférieures ou égales** aux simulations par seconde sur chaque ligne, sinon il y a un bug de comptage ;
- le taux de table est strictement compris entre 0 et 100 pour cent, une valeur de 0 signalant que les compteurs ne sont pas branchés ;
- l'étendue min à max ne dépasse pas quelques pour cent de la médiane, sinon la machine était chargée et la mesure ne sert pas de référence ;
- zéro violation d'invariant sur les trois positions.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-09-11-search-bench-resultats.md
git commit -m "Ajoute la reference de debit de la recherche avant batching"
```

---

## Auto-revue

**Couverture de la spec.** Section 3, race `parse_position` : tâche 1. Section 4, les deux additions C++ : tâches 2 et 3, avec `mcts_observe.cpp` comme nouveau fichier et l'ajout aux sources CMake. Section 5, les trois grandeurs et la variance : tâche 4 pour le calcul, tâche 5 pour la restitution, tâche 6 pour le passage réel. Section 6, les invariants : tâche 3, y compris le passage de découverte et la consigne de ne pas affaiblir un invariant qui échoue. Section 7, sorties et barrières : tâche 5, avec la barrière rapide vérifiée de bout en bout. Section 8, critère d'acceptation du batching : hors périmètre de ce plan par construction, il concerne le chantier suivant. Section 10, risques : le contrôle du débit de self-play après instrumentation est l'étape 9 de la tâche 2.

**Scan des placeholders.** Aucun « TBD », aucun « gérer les cas limites », aucun renvoi du type « comme la tâche N ». Une réserve assumée à l'étape 6 de la tâche 3 : forger un arbre corrompu depuis l'extérieur est impossible, donc la preuve que le contrôle mord passe par une modification temporaire du code C++, décrite explicitement.

**Cohérence des types.** `Mesure` porte les mêmes huit champs en tâches 4 et 5. `agreger` renvoie partout un dictionnaire indexé par `(position, chemin)`. `TreeReport` expose `nodes`, `max_depth`, `violations` et `messages` en tâches 3 et 5, et le tuple `invariants` de la tâche 5 reprend ces champs dans cet ordre. `TAILLE_TT` vaut 8192 en tâches 2, 3 et 4.

**Point signalé à l'utilisateur, hors périmètre.** `uci.py:14` fixe `MODEL_PATH` à `C:\Users\M47h1\Documents\chess_cpp\...`, un chemin d'une autre machine, et `uci.py:29` demande `tt_size=4_000_000`, soit 3,9 Gio. Le bot n'est donc pas lançable ici en l'état. La tâche 1 rend la classe testable mais ne corrige ni le chemin ni la taille, qui ne relèvent pas de ce pipeline.
