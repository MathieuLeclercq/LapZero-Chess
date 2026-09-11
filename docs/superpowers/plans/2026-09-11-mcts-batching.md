# Batching MCTS avec virtual loss : plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Faire évaluer plusieurs positions par inférence dans un même arbre de recherche, au moyen d'un virtual loss à la manière de Leela Chess Zero.

**Architecture:** Un champ `n_in_flight` séparé sur `MCTSNode`, qui n'entre que dans le dénominateur du terme U de PUCT et laisse Q intact. Un noyau unique `MCTS::run_search` dans `src/mcts_batch.cpp`, que `step_analysis` et `mcts_search` appellent tous deux, et dont le paramètre `batch_size` vaut 0 pour la boucle séquentielle conservée. Les collisions sont détectées par `n_in_flight > 0` sur un nœud sans enfants.

**Tech Stack:** C++20 avec MSVC, pybind11, Python 3.13, onnxruntime 1.24.3, pytest, CMake.

Spec de référence : `docs/superpowers/specs/2026-09-11-mcts-batching-design.md`.

## Global Constraints

- **`n_in_flight` n'entre que dans le dénominateur du terme U.** `q_value()` continue d'utiliser `visit_count` seul. C'est le point qui distingue cette implémentation d'un incrément naïf de `visit_count`, lequel diluerait Q vers zéro et avantagerait les nœuds perdants.
- **`exploration_factor` du parent reste en `sqrt(visit_count)`**, sans `n_in_flight`. Délibéré : le virtual loss ne doit agir que sur le terme U des enfants. Variante à essayer plus tard seulement si la divergence est insuffisante.
- **Ne jamais réutiliser `backup()` pour poser ou annuler le virtual loss** : il alterne le signe en remontant (`mcts.cpp:71`), ce qui n'a aucun sens pour un compteur.
- **Après la collecte, le plateau n'est plus jamais nécessaire.** L'annulation du virtual loss, le backup et l'expansion depuis la capture remontent tous par les pointeurs `parent`.
- **`batch_size = 0` doit rester strictement la boucle actuelle**, inchangée, y compris dans son ordre d'opérations flottantes.
- **`tt_size` vaut 8192 dans les tests et le harnais**, jamais le défaut : une `TTEntry` pèse 1040 octets et le défaut de 2 097 143 entrées réserverait 2,03 Gio par instance.
- **Ne jamais écrire « nœuds par seconde »** dans le harnais ou les rapports. Les grandeurs sont **simulations par seconde**, **inférences par seconde** et **taux de succès de la table**.
- Le module compilé et ses DLL vivent dans `python_src/`. Tout script fait `os.add_dll_directory` sur ce répertoire avant `import chess_engine`.
- Le `.pyd` et `chess_engine.pyi` sont suivis par git : ils doivent être commités **avec** la source C++ correspondante.
- Pas de tiret cadratin dans le code, les commentaires ou les messages de commit. **Jamais de ligne `Co-Authored-By`.**
- `cmake` hors PATH : `"/c/Program Files/Microsoft Visual Studio/18/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe"`.
- Filet à chaque tâche : `.venv/Scripts/python.exe -m pytest python_src/tests -q` puis `./build/Release/chess_perft.exe bench --strict --check-fen`.

---

### Task 1: Le champ `n_in_flight` et la détection de collision

**Files:**
- Modify: `src/mcts.hpp` (champ de `MCTSNode`), `src/mcts.cpp:13-15` (constructeur), `src/mcts.cpp:18-25` (`ucb_score`), `src/mcts.cpp:83` (garde de collision dans `select_leaf`)
- Test: `python_src/tests/test_virtual_loss.py` (créer)

**Interfaces:**
- Consomme : rien.
- Produit : `MCTSNode::n_in_flight` de type `uint32_t`, initialisé à 0. `select_leaf` s'arrête sur un nœud tel que `children.empty() && n_in_flight > 0`. `TreeReport` gagne un champ `en_vol` (`uint64_t`) comptant les nœuds dont `n_in_flight != 0`.

Cette tâche ne change aucun comportement : rien n'incrémente encore `n_in_flight`, donc `1.0f + visit_count + 0` est bit à bit identique à `1.0f + visit_count`. L'étape 5 le vérifie sur les données plutôt que de s'en remettre au raisonnement.

- [ ] **Step 1: Capturer la référence AVANT toute modification**

```bash
cd python_src && ../.venv/Scripts/python.exe -c "
import os, json
os.add_dll_directory(os.getcwd())
import chess_engine
ev = chess_engine.ONNXEvaluator('checkpoints_onnx/2026_04_23_23h25_iter316_unsupervised.onnx', False)
POS = [('ouverture','r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4'),
       ('milieu','r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1'),
       ('finale','8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1')]
ref = {}
for nom, fen in POS:
    b = chess_engine.Chessboard(); b.load_fen(fen)
    pi = chess_engine.MCTS(ev, 8192).mcts_search(b, 400, 1.4, False)
    ref[nom] = [(i, pi[i]) for i in range(4672) if pi[i] > 0]
import pathlib
sortie = pathlib.Path(os.environ['TEMP']) / 'ref_pi_avant.json'
sortie.write_text(json.dumps(ref))
print('reference ecrite :', sortie, sum(len(v) for v in ref.values()), 'entrees')
"
```

Attendu : trois positions, quelques dizaines d'entrées non nulles chacune.

- [ ] **Step 2: Écrire le test de collision**

```python
"""Le virtual loss, approche LC0 : un compteur separe qui n'entre que dans le
denominateur du terme U.

Tant que rien ne l'incremente, ce champ ne doit rien changer. Les tests de
comportement reel arrivent avec la boucle batchee.
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
DEPART = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


@pytest.fixture(scope="module")
def evaluateur(tmp_path_factory):
    import puzzle_bench

    chemin, _ = puzzle_bench.resoudre_modele(
        CHECKPOINT, tmp_path_factory.mktemp("onnx"))
    return chess_engine.ONNXEvaluator(str(chemin), False)


def _plateau(fen=DEPART):
    board = chess_engine.Chessboard()
    board.load_fen(fen)
    return board


def test_aucun_noeud_ne_reste_en_vol_apres_une_recherche(evaluateur):
    """Un virtual loss non annule laisserait des noeuds en vol, ce qui
    fausserait le terme U de toutes les recherches suivantes."""
    mcts = chess_engine.MCTS(evaluateur, TAILLE_TT)
    mcts.step_analysis(_plateau(), 200, 1.4)

    assert mcts.inspect_tree().en_vol == 0


def test_les_visites_de_la_racine_egalent_les_simulations(evaluateur):
    """Rappel du filet existant : c'est le controle qui detecte un virtual loss
    mal annule, puisqu'il gonflerait ce compte."""
    mcts = chess_engine.MCTS(evaluateur, TAILLE_TT)
    board = _plateau()

    mcts.step_analysis(board, 200, 1.4)

    assert sum(s.visits for s in mcts.get_analysis_results()) == 200


def test_la_recherche_reste_deterministe(evaluateur):
    """Deux recherches identiques doivent donner exactement la meme
    distribution : sans Dirichlet, rien n'introduit d'alea."""
    board_a = _plateau()
    board_b = _plateau()

    pi_a = chess_engine.MCTS(evaluateur, TAILLE_TT).mcts_search(board_a, 200, 1.4, False)
    pi_b = chess_engine.MCTS(evaluateur, TAILLE_TT).mcts_search(board_b, 200, 1.4, False)

    assert list(pi_a) == list(pi_b)
```

- [ ] **Step 3: Lancer et vérifier l'échec**

Run: `.venv/Scripts/python.exe -m pytest python_src/tests/test_virtual_loss.py -q`
Attendu : ÉCHEC, `AttributeError: 'chess_engine.TreeReport' object has no attribute 'en_vol'`.

- [ ] **Step 4: Implémenter**

Dans `src/mcts.hpp`, ajouter à `MCTSNode`, après `float total_value;` :

```cpp
    // Virtual loss, approche LC0 : nombre de descentes en cours passant par ce
    // noeud. N'entre QUE dans le denominateur du terme U de ucb_score, jamais
    // dans q_value(), sans quoi Q se diluerait vers zero et avantagerait les
    // noeuds perdants.
    uint32_t n_in_flight;
```

Dans `src/mcts.hpp`, ajouter à `TreeReport`, après `uint64_t violations = 0;` :

```cpp
    uint64_t en_vol = 0;   // noeuds dont n_in_flight != 0 apres la recherche
```

Dans `src/mcts.cpp:13-15`, initialiser :

```cpp
MCTSNode::MCTSNode(float prior, int move_idx, MCTSNode* parent)
    : prior(prior), move_idx(move_idx), parent(parent),
    visit_count(0), total_value(0.0f), is_terminal(false), n_in_flight(0) {
}
```

Dans `src/mcts.cpp:22`, remplacer la ligne du terme U :

```cpp
    float u = exploration_factor * prior / (1.0f + visit_count + n_in_flight);
```

Dans `src/mcts.cpp`, dans `select_leaf`, juste après `if (node->children.empty()) {` (ligne 83) :

```cpp
        if (node->children.empty()) {
            // Collision : cette feuille est deja collectee par une descente
            // precedente du meme lot. Sans cette garde, l'expansion paresseuse
            // ci-dessous lui creerait des enfants, et le backup du lot lui en
            // creerait un second jeu. L'arbre serait corrompu en silence.
            if (node->n_in_flight > 0) break;

            uint64_t hash = board.getZobristHash();
```

Dans `src/mcts_observe.cpp`, dans `visiter`, juste après `rapport.nodes++;` :

```cpp
    if (node->n_in_flight != 0) rapport.en_vol++;
```

Dans `src/bindings.cpp`, ajouter au bloc `py::class_<TreeReport>` :

```cpp
        .def_readonly("en_vol", &TreeReport::en_vol)
```

- [ ] **Step 5: Compiler, lancer le filet, et prouver que rien n'a changé**

Run: `"/c/Program Files/Microsoft Visual Studio/18/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe" --build build --config Release --target chess_engine`
Attendu : compilation sans erreur.

Run: `.venv/Scripts/python.exe -m pytest python_src/tests -q`
Attendu : toute la suite passe.

Run: `./build/Release/chess_perft.exe bench --strict --check-fen`
Attendu : `Resultat : SUCCES`.

Puis la comparaison bit à bit contre la référence de l'étape 1 :

```bash
cd python_src && ../.venv/Scripts/python.exe -c "
import os, json, pathlib
os.add_dll_directory(os.getcwd())
import chess_engine
ref = json.loads((pathlib.Path(os.environ['TEMP']) / 'ref_pi_avant.json').read_text())
ev = chess_engine.ONNXEvaluator('checkpoints_onnx/2026_04_23_23h25_iter316_unsupervised.onnx', False)
POS = [('ouverture','r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4'),
       ('milieu','r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1'),
       ('finale','8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1')]
for nom, fen in POS:
    b = chess_engine.Chessboard(); b.load_fen(fen)
    pi = chess_engine.MCTS(ev, 8192).mcts_search(b, 400, 1.4, False)
    apres = [[i, pi[i]] for i in range(4672) if pi[i] > 0]
    assert apres == ref[nom], f'{nom} : la distribution a change'
print('identique bit a bit sur les trois positions')
"
```

Attendu : `identique bit a bit sur les trois positions`. Si ce contrôle échoue, ne pas continuer : le champ aurait un effet là où il ne doit pas en avoir.

- [ ] **Step 6: Commit**

```bash
git add src/mcts.hpp src/mcts.cpp src/mcts_observe.cpp src/bindings.cpp python_src/chess_engine.cp313-win_amd64.pyd python_src/chess_engine.pyi python_src/tests/test_virtual_loss.py
git commit -m "Ajoute n_in_flight et la detection de collision"
```

---

### Task 2: Le noyau `run_search`, sans encore batcher

**Files:**
- Create: `src/mcts_batch.cpp`
- Modify: `src/mcts.hpp` (déclaration de `run_search`), `src/mcts.cpp` (corps de `mcts_search` et `step_analysis`), `CMakeLists.txt:32-43` (sources de `chess_core`)

**Interfaces:**
- Consomme : `n_in_flight` de la tâche 1.
- Produit : `void MCTS::run_search(MCTSNode* root, Chessboard& board, int simulations, float c_puct, int batch_size)`, privée. Pour `batch_size == 0` elle exécute la boucle séquentielle actuelle, déplacée telle quelle. Les autres valeurs lèvent pour l'instant, la tâche 3 les implémente.

Cette tâche est une **restructuration sans changement de comportement** : la boucle est déplacée, pas réécrite.

- [ ] **Step 1: Capturer à nouveau la référence**

```bash
cd python_src && ../.venv/Scripts/python.exe -c "
import os, json, pathlib
os.add_dll_directory(os.getcwd())
import chess_engine
ev = chess_engine.ONNXEvaluator('checkpoints_onnx/2026_04_23_23h25_iter316_unsupervised.onnx', False)
POS = [('ouverture','r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4'),
       ('milieu','r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1'),
       ('finale','8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1')]
ref = {}
for nom, fen in POS:
    b = chess_engine.Chessboard(); b.load_fen(fen)
    pi = chess_engine.MCTS(ev, 8192).mcts_search(b, 400, 1.4, False)
    ref[nom] = [[i, pi[i]] for i in range(4672) if pi[i] > 0]
(pathlib.Path(os.environ['TEMP']) / 'ref_pi_t2.json').write_text(json.dumps(ref))
print('reference ecrite')
"
```

- [ ] **Step 2: Déclarer `run_search` dans `src/mcts.hpp`**

Dans la section `private` de `class MCTS`, après `std::pair<MCTSNode*, int> select_leaf(...);` :

```cpp
    // Noyau unique de recherche, defini dans mcts_batch.cpp.
    // batch_size == 0 : boucle sequentielle historique, conservee telle quelle.
    // batch_size >= 1 : boucle batchee avec virtual loss.
    void run_search(MCTSNode* root, Chessboard& board, int simulations,
                    float c_puct, int batch_size);
```

- [ ] **Step 3: Créer `src/mcts_batch.cpp` avec la boucle séquentielle déplacée**

```cpp
// Noyau de recherche : boucle sequentielle historique et boucle batchee.
//
// Separe de mcts.cpp, qui depasse 520 lignes, sur le modele de
// mcts_observe.cpp. Voir docs/superpowers/specs/2026-09-11-mcts-batching-design.md

#include "mcts.hpp"

#include <stdexcept>

// Boucle sequentielle : une inference par simulation. Deplacee telle quelle
// depuis mcts_search, sans modification de l'ordre des operations, parce
// qu'elle sert de reference au test d'equivalence de la boucle batchee.

void MCTS::run_search(MCTSNode* root, Chessboard& board, int simulations,
                      float c_puct, int batch_size) {
    if (batch_size != 0) {
        throw std::runtime_error(
            "run_search : seul batch_size = 0 est implemente pour l'instant");
    }

    for (int sim = 0; sim < simulations; sim++) {
        auto [node, moves_played] = select_leaf(root, board, c_puct);

        if (node->is_terminal) {
            m_terminal_hits.fetch_add(1, std::memory_order_relaxed);
            float value = 0.0f;
            if (board.checkThreefoldRepetition() ||
                board.getHalfMoveClock() >= 100 ||
                board.checkInsufficientMaterial()) {
                value = 0.0f;
            }
            else {
                value = board.isInCheck() ? -1.0f : 0.0f;
            }

            backup(node, value);
            for (int i = 0; i < moves_played; i++) board.undoMove();
            continue;
        }

        if (node->children.empty()) {
            float value = expand_node_single(node, board);
            backup(node, value);
        }

        for (int i = 0; i < moves_played; i++) {
            board.undoMove();
        }
    }
}
```

- [ ] **Step 4: Faire de `mcts_search` et `step_analysis` des enveloppes**

Dans `src/mcts.cpp`, remplacer le corps de la boucle de `mcts_search` (la boucle `for (int sim = 0; ...)` entière) par :

```cpp
    run_search(root.get(), board, num_simulations, c_puct, 0);
```

Faire de même dans `step_analysis` : remplacer sa boucle `for (int sim = 0; ...)` par un appel à `run_search(m_analysis_root.get(), board, num_simulations, c_puct, 0)`.

Attention à `step_analysis` : sa boucle prend `m_mutex` **par simulation**. L'appel à `run_search` doit donc être encadré par un seul `std::lock_guard` sur toute la durée, ce qui est plus sûr et non moins correct : la boucle batchée détiendra des pointeurs bruts et ne doit pas relâcher le verrou en cours de lot.

- [ ] **Step 5: Ajouter le fichier aux sources CMake**

Dans `CMakeLists.txt`, après la ligne `src/mcts_observe.cpp` :

```cmake
    src/mcts_batch.cpp
```

- [ ] **Step 6: Compiler et prouver que rien n'a changé**

Run: `"/c/Program Files/Microsoft Visual Studio/18/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe" --build build --config Release --target chess_engine`
Attendu : compilation sans erreur.

Run: `.venv/Scripts/python.exe -m pytest python_src/tests -q`
Attendu : toute la suite passe.

Puis la comparaison bit à bit, identique à l'étape 5 de la tâche 1 mais contre `ref_pi_t2.json` :

```bash
cd python_src && ../.venv/Scripts/python.exe -c "
import os, json, pathlib
os.add_dll_directory(os.getcwd())
import chess_engine
ref = json.loads((pathlib.Path(os.environ['TEMP']) / 'ref_pi_t2.json').read_text())
ev = chess_engine.ONNXEvaluator('checkpoints_onnx/2026_04_23_23h25_iter316_unsupervised.onnx', False)
POS = [('ouverture','r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4'),
       ('milieu','r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1'),
       ('finale','8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1')]
for nom, fen in POS:
    b = chess_engine.Chessboard(); b.load_fen(fen)
    pi = chess_engine.MCTS(ev, 8192).mcts_search(b, 400, 1.4, False)
    assert [[i, pi[i]] for i in range(4672) if pi[i] > 0] == ref[nom], nom
print('restructuration sans changement de comportement : verifie')
"
```

- [ ] **Step 7: Commit**

```bash
git add src/mcts.hpp src/mcts.cpp src/mcts_batch.cpp CMakeLists.txt python_src/chess_engine.cp313-win_amd64.pyd
git commit -m "Extrait la boucle de recherche dans un noyau run_search"
```

---

### Task 3: La boucle batchée

**Files:**
- Modify: `src/mcts_batch.cpp` (la boucle batchée), `src/mcts.hpp` (déclaration de `expand_and_backup_prepared`), `src/mcts.cpp` (corps de `expand_and_backup_prepared`)
- Test: `python_src/tests/test_virtual_loss.py` (compléter)

**Interfaces:**
- Consomme : `run_search` de la tâche 2, `n_in_flight` de la tâche 1.
- Produit : `void MCTS::expand_and_backup_prepared(MCTSNode* leaf, const std::vector<int>& legal_indices, uint64_t hash, const float* policy, float value)`, publique. `run_search` accepte désormais `batch_size >= 1`.

- [ ] **Step 1: Écrire `expand_and_backup_prepared` dans `src/mcts.cpp`**

Déclaration dans `src/mcts.hpp`, section `public`, après `void expand_and_backup(...)` :

```cpp
    // Variante de expand_and_backup qui ne touche pas au plateau : les donnees
    // dont elle a besoin ont ete capturees pendant la collecte, au moment ou le
    // plateau etait sur la feuille. Sans elle, il faudrait rejouer le chemin de
    // chaque feuille avant son backup, soit deux traversees supplementaires.
    void expand_and_backup_prepared(MCTSNode* leaf, const std::vector<int>& legal_indices,
                                    uint64_t hash, const float* policy, float value);
```

Corps dans `src/mcts.cpp`, juste après `expand_and_backup` :

```cpp
void MCTS::expand_and_backup_prepared(MCTSNode* leaf, const std::vector<int>& legal_indices,
                                      uint64_t hash, const float* policy, float value) {
    // Les feuilles terminales sont traitees pendant la descente, jamais
    // collectees : legal_indices n'est donc jamais vide ici.
    size_t tt_idx = hash % m_tt_size;
    TTEntry& tt = transposition_table[tt_idx];
    tt.hash = hash;
    tt.value = value;
    tt.policy_size = std::min((int)legal_indices.size(), TT_MAX_MOVES);

    float sum_legal = 0.0f;
    for (int k = 0; k < tt.policy_size; ++k) {
        int idx = legal_indices[k];
        float prob = policy[idx];
        tt.legal_policy[k] = { idx, prob };
        sum_legal += prob;
    }

    leaf->children.reserve(tt.policy_size);
    if (sum_legal > 0.0f) {
        for (int k = 0; k < tt.policy_size; ++k) {
            leaf->children.emplace_back(
                tt.legal_policy[k].first,
                std::make_unique<MCTSNode>(tt.legal_policy[k].second / sum_legal,
                                           tt.legal_policy[k].first, leaf));
        }
    }
    else {
        float uniform_prob = 1.0f / tt.policy_size;
        for (int k = 0; k < tt.policy_size; ++k) {
            leaf->children.emplace_back(
                tt.legal_policy[k].first,
                std::make_unique<MCTSNode>(uniform_prob, tt.legal_policy[k].first, leaf));
        }
    }

    backup(leaf, value);
}
```

- [ ] **Step 2: Écrire la boucle batchée dans `src/mcts_batch.cpp`**

Ajouter en tête du fichier `#include <vector>` et `#include <cstdint>`, puis remplacer le garde-fou `if (batch_size != 0) throw ...` par la boucle. Le corps complet de `run_search` devient :

```cpp
namespace {

// Pose et annule le virtual loss du noeud jusqu'a la racine. On ne peut pas
// reutiliser backup(), qui alterne le signe en remontant (mcts.cpp:71) : cela
// n'a aucun sens pour un compteur.
void poser_virtual_loss(MCTSNode* noeud) {
    for (MCTSNode* n = noeud; n != nullptr; n = n->parent) n->n_in_flight += 1;
}

void annuler_virtual_loss(MCTSNode* noeud) {
    for (MCTSNode* n = noeud; n != nullptr; n = n->parent) n->n_in_flight -= 1;
}

// Ce qu'on capture pendant que le plateau est sur la feuille. Apres la
// collecte, le plateau n'est plus jamais necessaire : l'annulation du virtual
// loss, le backup et l'expansion remontent tous par les pointeurs parent.
struct FeuilleCollectee {
    MCTSNode* noeud = nullptr;
    std::vector<int> coups_legaux;
    uint64_t hash = 0;
};

}  // namespace

void MCTS::run_search(MCTSNode* root, Chessboard& board, int simulations,
                      float c_puct, int batch_size) {
    if (batch_size == 0) {
        // Conserver ici, sans y toucher, le corps ecrit a l'etape 3 de la
        // tache 2 : la boucle for (int sim = 0; sim < simulations; sim++)
        // complete, suivie d'un return. Son ordre d'operations flottantes est
        // la reference du test d'equivalence a batch 1.
        return;
    }

    std::vector<FeuilleCollectee> lot;
    std::vector<float> tenseurs;
    std::vector<float> policies;
    std::vector<float> values;
    std::vector<float> tenseur_courant;

    int faites = 0;
    while (faites < simulations) {
        lot.clear();
        tenseurs.clear();

        const int reste = simulations - faites;
        const int cible = std::min(batch_size, reste);

        while ((int)lot.size() < cible) {
            auto [noeud, coups_joues] = select_leaf(root, board, c_puct);

            // Feuille terminale : traitee sur place, aucune inference.
            if (noeud->is_terminal) {
                m_terminal_hits.fetch_add(1, std::memory_order_relaxed);
                float value = 0.0f;
                if (board.checkThreefoldRepetition() ||
                    board.getHalfMoveClock() >= 100 ||
                    board.checkInsufficientMaterial()) {
                    value = 0.0f;
                }
                else {
                    value = board.isInCheck() ? -1.0f : 0.0f;
                }
                backup(noeud, value);
                for (int i = 0; i < coups_joues; i++) board.undoMove();
                faites++;
                if (faites >= simulations) break;
                continue;
            }

            // Collision : cette feuille est deja dans le lot. On n'insiste pas,
            // comme LC0 dont TryStartScoreUpdate renvoie false. La premiere
            // descente d'un lot ne peut jamais collisionner, puisque tous les
            // n_in_flight sont annules a la fin du lot precedent : la boucle ne
            // peut donc pas tourner a vide.
            if (noeud->children.empty() && noeud->n_in_flight > 0) {
                for (int i = 0; i < coups_joues; i++) board.undoMove();
                break;
            }

            // Succes de table : expand_node_single renvoie la valeur en cache
            // sans appeler le reseau.
            if (!noeud->children.empty()) {
                float value = expand_node_single(noeud, board);
                backup(noeud, value);
                for (int i = 0; i < coups_joues; i++) board.undoMove();
                faites++;
                if (faites >= simulations) break;
                continue;
            }

            // Vraie feuille : le plateau est SUR elle, on capture tout ce dont
            // le backup aura besoin.
            FeuilleCollectee f;
            f.noeud = noeud;
            f.coups_legaux = board.getLegalMoveIndices();
            f.hash = board.getZobristHash();

            if (f.coups_legaux.empty()) {
                noeud->is_terminal = true;
                m_terminal_hits.fetch_add(1, std::memory_order_relaxed);
                backup(noeud, board.isInCheck() ? -1.0f : 0.0f);
                for (int i = 0; i < coups_joues; i++) board.undoMove();
                faites++;
                if (faites >= simulations) break;
                continue;
            }

            board.getAlphaZeroTensor(tenseur_courant);
            tenseurs.insert(tenseurs.end(), tenseur_courant.begin(), tenseur_courant.end());

            poser_virtual_loss(noeud);
            lot.push_back(std::move(f));

            for (int i = 0; i < coups_joues; i++) board.undoMove();
        }

        if (lot.empty()) continue;

        const int taille = (int)lot.size();
        m_nn_calls.fetch_add(taille, std::memory_order_relaxed);
        m_tt_misses.fetch_add(taille, std::memory_order_relaxed);
        m_evaluator->evaluate_batch(tenseurs, policies, values, taille);

        for (int i = 0; i < taille; ++i) {
            annuler_virtual_loss(lot[i].noeud);
            expand_and_backup_prepared(lot[i].noeud, lot[i].coups_legaux,
                                       lot[i].hash, policies.data() + (size_t)i * 4672,
                                       values[i]);
            faites++;
        }
    }
}
```

Ajouter `#include <algorithm>` pour `std::min`.

Note sur les compteurs : `expand_node_single` incrémente déjà `m_nn_calls` et `m_tt_misses` sur son chemin, mais la boucle batchée ne passe pas par lui pour les vraies feuilles. C'est pourquoi elle les incrémente elle-même, d'un coup, avant l'inférence.

- [ ] **Step 3: Écrire les tests de la boucle batchée**

À ajouter à `python_src/tests/test_virtual_loss.py` :

```python
@pytest.mark.parametrize("taille", [1, 2, 8, 32])
def test_la_boucle_batchee_respecte_les_invariants(evaluateur, taille):
    mcts = chess_engine.MCTS(evaluateur, TAILLE_TT)

    mcts.step_analysis(_plateau(), 400, 1.4, taille)

    rapport = mcts.inspect_tree()
    assert rapport.violations == 0, list(rapport.messages)
    assert rapport.en_vol == 0, "virtual loss non annule"


@pytest.mark.parametrize("taille", [1, 2, 8, 32])
def test_le_compte_de_visites_de_la_racine_est_exact(evaluateur, taille):
    """C'est le controle qui detecte un virtual loss mal annule ou une
    simulation comptee deux fois."""
    mcts = chess_engine.MCTS(evaluateur, TAILLE_TT)

    mcts.step_analysis(_plateau(), 400, 1.4, taille)

    assert sum(s.visits for s in mcts.get_analysis_results()) == 400


def test_batch_1_est_identique_a_la_boucle_sequentielle(evaluateur):
    """Avec un seul element, la collecte se reduit a une descente, le virtual
    loss est pose puis annule sans que personne d'autre ne le voie, et le
    backup est le meme. C'est le critere d'acceptation du chantier."""
    sequentiel = chess_engine.MCTS(evaluateur, TAILLE_TT).mcts_search(
        _plateau(), 400, 1.4, False, 0)
    batche = chess_engine.MCTS(evaluateur, TAILLE_TT).mcts_search(
        _plateau(), 400, 1.4, False, 1)

    assert list(sequentiel) == list(batche)


def test_un_batch_plus_grand_reduit_le_nombre_d_inferences_par_seconde_pas_leur_total(evaluateur):
    """Le batching ne doit pas changer le nombre d'inferences, seulement leur
    regroupement. Un ecart signalerait des simulations perdues ou dupliquees."""
    petit = chess_engine.MCTS(evaluateur, TAILLE_TT)
    petit.step_analysis(_plateau(), 400, 1.4, 1)

    grand = chess_engine.MCTS(evaluateur, TAILLE_TT)
    grand.step_analysis(_plateau(), 400, 1.4, 32)

    a, b = petit.get_counters(), grand.get_counters()
    assert a.nn_calls + a.tt_hits + a.terminal_hits > 0
    assert abs(a.nn_calls - b.nn_calls) <= 400 * 0.5, (
        f"ecart trop grand : {a.nn_calls} contre {b.nn_calls}")
```

- [ ] **Step 4: Lancer et vérifier l'échec**

Run: `.venv/Scripts/python.exe -m pytest python_src/tests/test_virtual_loss.py -q`
Attendu : ÉCHEC, `step_analysis` n'accepte pas encore de quatrième argument. C'est la tâche 4 qui l'expose ; cette étape confirme simplement que les tests sont en place.

- [ ] **Step 5: Compiler et vérifier le filet**

Run: `"/c/Program Files/Microsoft Visual Studio/18/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe" --build build --config Release --target chess_engine`
Attendu : compilation sans erreur.

Run: `./build/Release/chess_perft.exe bench --strict --check-fen`
Attendu : `Resultat : SUCCES`.

- [ ] **Step 6: Commit**

```bash
git add src/mcts.hpp src/mcts.cpp src/mcts_batch.cpp python_src/chess_engine.cp313-win_amd64.pyd python_src/tests/test_virtual_loss.py
git commit -m "Implemente la boucle batchee avec virtual loss"
```

---

### Task 4: Exposer `batch_size` à Python

**Files:**
- Modify: `src/mcts.hpp` (signatures de `step_analysis` et `mcts_search`), `src/mcts.cpp` (les deux corps), `src/bindings.cpp:135-146`
- Test: `python_src/tests/test_virtual_loss.py` (déjà écrits en tâche 3)

**Interfaces:**
- Consomme : `run_search` complet de la tâche 3.
- Produit : `MCTS.step_analysis(board, num_simulations, c_puct=1.4, batch_size=32)` et `MCTS.mcts_search(board, num_simulations, c_puct=1.4, add_dirichlet=False, batch_size=32)` côté Python.

- [ ] **Step 1: Élargir les signatures C++**

Dans `src/mcts.hpp` :

```cpp
    void step_analysis(Chessboard& board, int num_simulations, float c_puct,
                       int batch_size = 32);
    std::vector<float> mcts_search(Chessboard& board, int num_simulations, float c_puct,
                                   bool add_dirichlet, int batch_size = 32);
```

Dans `src/mcts.cpp`, faire suivre les deux définitions et passer `batch_size` à `run_search` au lieu du `0` codé en dur.

- [ ] **Step 2: Élargir les bindings**

Dans `src/bindings.cpp`, remplacer les deux `.def` :

```cpp
        .def("mcts_search", &MCTS::mcts_search,
            py::call_guard<py::gil_scoped_release>(),
            py::arg("board"), py::arg("num_simulations"), py::arg("c_puct") = 1.4f,
            py::arg("add_dirichlet") = false, py::arg("batch_size") = 32)

        .def("step_analysis", &MCTS::step_analysis,
            py::call_guard<py::gil_scoped_release>(),
            py::arg("board"), py::arg("num_simulations"), py::arg("c_puct") = 1.4f,
            py::arg("batch_size") = 32)
```

- [ ] **Step 3: Compiler et lancer les tests**

Run: `"/c/Program Files/Microsoft Visual Studio/18/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe" --build build --config Release --target chess_engine`
Attendu : compilation sans erreur.

Run: `.venv/Scripts/python.exe -m pytest python_src/tests -q`
Attendu : toute la suite passe, y compris les tests de la tâche 3.

Run: `./build/Release/chess_perft.exe bench --strict --check-fen`
Attendu : `Resultat : SUCCES`.

- [ ] **Step 4: Prouver que la garde de collision mord**

Commenter la ligne `if (node->n_in_flight > 0) break;` ajoutée en tâche 1 dans `select_leaf`, recompiler, puis :

Run: `.venv/Scripts/python.exe -m pytest python_src/tests/test_virtual_loss.py -q`
Attendu : `test_la_boucle_batchee_respecte_les_invariants` échoue à taille 8 ou 32 avec des messages `enfant duplique`. Rétablir ensuite, recompiler, revérifier que tout passe.

Si ce test **ne** tombe **pas**, ne pas conclure que la garde est inutile : cela signifierait que le cas ne se produit pas sur cette position, et il faut alors augmenter le nombre de simulations à 2000 pour densifier la table avant de conclure.

- [ ] **Step 5: Commit**

```bash
git add src/mcts.hpp src/mcts.cpp src/bindings.cpp python_src/chess_engine.cp313-win_amd64.pyd python_src/chess_engine.pyi
git commit -m "Expose batch_size a Python sur les deux chemins de recherche"
```

---

### Task 5: Le balayage de débit dans le harnais

**Files:**
- Modify: `python_src/search_bench.py` (option `--batch-sizes`, colonne dans le rapport)
- Modify: `python_src/tests/test_search_bench.py`

**Interfaces:**
- Consomme : `batch_size` de la tâche 4.
- Produit : `Mesure` gagne un champ `batch_size: int`. `agreger` groupe par `(position, chemin, batch_size)`. La CLI accepte `--batch-sizes 1 2 4 8 16 32 64`.

- [ ] **Step 1: Écrire les tests**

À ajouter à `python_src/tests/test_search_bench.py`, et adapter le helper `_m` existant en lui ajoutant `batch_size=32` :

```python
def test_agreger_groupe_aussi_par_taille_de_batch():
    mesures = [
        _m(batch_size=1, duree_s=4.0),
        _m(batch_size=32, duree_s=1.0),
        _m(batch_size=32, duree_s=1.0),
    ]

    agr = agreger(mesures)

    assert set(agr) == {("depart", "mcts_search", 1), ("depart", "mcts_search", 32)}
    assert agr[("depart", "mcts_search", 32)]["passages"] == 2
    assert agr[("depart", "mcts_search", 1)]["sims_par_seconde_median"] == pytest.approx(100.0)
    assert agr[("depart", "mcts_search", 32)]["sims_par_seconde_median"] == pytest.approx(400.0)


def test_le_rapport_affiche_la_taille_de_batch():
    agr = agreger([_m(batch_size=1), _m(batch_size=32)])

    texte = format_report(agr, CONTEXTE, [])

    assert "batch" in texte.lower()
```

- [ ] **Step 2: Lancer et vérifier l'échec**

Run: `.venv/Scripts/python.exe -m pytest python_src/tests/test_search_bench.py -q`
Attendu : ÉCHEC, `Mesure.__init__() got an unexpected keyword argument 'batch_size'`.

- [ ] **Step 3: Implémenter**

Dans `python_src/search_bench.py`, ajouter le champ à `Mesure`, après `simulations: int` :

```python
    batch_size: int
```

Faire passer `batch_size` aux deux fonctions de mesure et aux `Mesure` qu'elles construisent :

```python
def mesurer_mcts_search(evaluateur, fen: str, nom: str, simulations: int,
                        c_puct: float = 1.4, batch_size: int = 32) -> Mesure:
    mcts = chess_engine.MCTS(evaluateur, TAILLE_TT)
    board = charger_position(fen)
    mcts.reset_counters()

    debut = time.perf_counter()
    mcts.mcts_search(board, simulations, c_puct, False, batch_size)
    duree = time.perf_counter() - debut

    c = mcts.get_counters()
    return Mesure(nom, "mcts_search", simulations, batch_size, duree,
                  c.nn_calls, c.tt_hits, c.tt_misses, c.terminal_hits)
```

et la même chose pour `mesurer_step_analysis` avec `mcts.step_analysis(board, simulations, c_puct, batch_size)`.

Dans `agreger`, remplacer la clé de groupement :

```python
        groupes.setdefault((m.position, m.chemin, m.batch_size), []).append(m)
```

Dans `_EN_TETE`, insérer une colonne après `Chemin` :

```python
_EN_TETE = (
    "| Position | Chemin | batch | passages | sims/s (med) | sims/s (min a max) "
    "| inferences/s (med) | taux table |\n"
    "|---|---|---|---|---|---|---|---|"
)
```

et dans `format_report`, dépaqueter la clé en trois et insérer `{batch}` dans la ligne produite.

Dans `main`, remplacer l'argument `--simulations` seul par l'ajout de :

```python
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[32],
                        help="tailles de batch a balayer ; 0 designe la boucle "
                             "sequentielle conservee")
```

et faire boucler la mesure sur `args.batch_sizes`.

- [ ] **Step 4: Lancer les tests**

Run: `.venv/Scripts/python.exe -m pytest python_src/tests -q`
Attendu : toute la suite passe.

- [ ] **Step 5: Commit**

```bash
git add python_src/search_bench.py python_src/tests/test_search_bench.py
git commit -m "Balaye les tailles de batch dans le harnais de debit"
```

---

### Task 6: La campagne de mesure

**Files:**
- Modify: `docs/superpowers/specs/2026-09-11-search-bench-resultats.md` (le diff git montre le gain)
- Create: `data/bench_results/2026_04_23_23h25_iter316_unsupervised_batche.csv`

**Interfaces:**
- Consomme : tout ce qui précède.
- Produit : la mesure du gain et la vérification de la qualité.

- [ ] **Step 1: Balayer les tailles de batch**

```bash
cd python_src && ../.venv/Scripts/python.exe search_bench.py \
  --model checkpoints/2026_04_23_23h25_iter316_unsupervised.pt \
  --simulations 400 --passages 5 --batch-sizes 0 1 2 4 8 16 32 64
```

Attendu : une courbe croissante puis saturante. La référence à battre est 286 à 299 simulations par seconde.

- [ ] **Step 2: Vérifier les invariants à la taille retenue**

```bash
cd python_src && ../.venv/Scripts/python.exe search_bench.py \
  --model checkpoints/2026_04_23_23h25_iter316_unsupervised.pt \
  --invariants --simulations 400 \
  --out-rapport ../docs/superpowers/specs/_invariants.md
```

Attendu : zéro violation sur les trois positions. Reporter la section dans le rapport de référence puis supprimer `_invariants.md`.

- [ ] **Step 3: Passer la barrière qualité**

```bash
cd python_src && ../.venv/Scripts/python.exe puzzle_bench.py \
  --model checkpoints/2026_04_23_23h25_iter316_unsupervised.pt \
  --out-csv ../data/bench_results/2026_04_23_23h25_iter316_unsupervised_batche.csv \
  --out-rapport ../docs/superpowers/specs/_puzzles_batche.md
```

Attendu : environ 11 min, zéro erreur de données.

- [ ] **Step 4: Comparer en apparié à la référence**

```bash
cd /c/Users/mlecl/Documents/LapZero-Chess && .venv/Scripts/python.exe -c "
import csv, math
ref = {r['ligne']: r for r in csv.DictReader(open('data/bench_results/2026_04_23_23h25_iter316_unsupervised.csv', encoding='utf-8'))}
new = {r['ligne']: r for r in csv.DictReader(open('data/bench_results/2026_04_23_23h25_iter316_unsupervised_batche.csv', encoding='utf-8'))}
communs = sorted(set(ref) & set(new), key=int)
b = sum(1 for k in communs if ref[k]['reussi_recherche']=='True' and new[k]['reussi_recherche']!='True')
c = sum(1 for k in communs if ref[k]['reussi_recherche']!='True' and new[k]['reussi_recherche']=='True')
n = b + c
chi2 = max(0.0, abs(b-c)-1.0)**2/n if n else 0.0
p = math.erfc(math.sqrt(chi2/2)) if n else 1.0
tr = sum(1 for k in communs if ref[k]['reussi_recherche']=='True')
tn = sum(1 for k in communs if new[k]['reussi_recherche']=='True')
print(f'puzzles communs : {len(communs)}')
print(f'sequentiel : {100*tr/len(communs):.1f} %   batche : {100*tn/len(communs):.1f} %')
print(f'sequentiel bon, batche mauvais : {b}')
print(f'sequentiel mauvais, batche bon : {c}')
print(f'McNemar : chi2 = {chi2:.2f}, p = {p:.4f}')
print()
# Le reseau seul ne depend pas de la recherche : il doit etre identique.
diff = [k for k in communs if ref[k]['reussi_reseau'] != new[k]['reussi_reseau']]
print(f'desaccords sur le reseau seul : {len(diff)}  (doit etre 0)')
"
```

Attendu : la colonne réseau seul identique, et un McNemar non significatif sur la colonne recherche. Un `p` inférieur à 0,05 avec `b > c` signalerait une **dégradation de la qualité par simulation**, ce qui invaliderait le chantier même si le débit a doublé.

- [ ] **Step 5: Mettre à jour le rapport de référence et commiter**

Régénérer le rapport de débit avec la taille de batch retenue, reporter la section des invariants, et ajouter une section de lecture indiquant le gain mesuré, la taille de batch choisie, et le résultat du McNemar qualité.

```bash
git add docs/superpowers/specs/2026-09-11-search-bench-resultats.md data/bench_results
git commit -m "Mesure le gain du batching et verifie la qualite"
```

---

## Auto-revue

**Couverture de la spec.** Section 3, périmètre : le noyau unique est en tâche 2, le `batch_size` par appel en tâche 4, la boucle séquentielle conservée en tâche 2. Section 4, architecture : tâche 2. Section 5, virtual loss : tâche 1 pour le champ et `ucb_score`, tâche 3 pour poser et annuler. Section 6, collisions : garde en tâche 1, preuve qu'elle mord en tâche 4 étape 4. Section 7, collecte et capture : tâche 3, avec `expand_and_backup_prepared`. Section 8, barrières : équivalence batch 1 en tâche 3, invariants et `en_vol` en tâches 1 et 3, balayage en tâche 5, banc de puzzles en tâche 6. Section 9, incertitudes : elles ne se lèvent que par la mesure de la tâche 6, et la réserve sur le cas froid y est rappelée.

**Scan des placeholders.** Une premiere version contenait un marqueur `static void boucle_sequentielle_impl() {}` a supprimer, et un `...` dans le bloc de la tache 3 : les deux ont ete remplaces, le premier par rien du tout, le second par une consigne explicite renvoyant au corps ecrit en tache 2. Aucun « TBD », aucun « gérer les cas limites », aucun renvoi du type « comme la tâche N » sans le code.

**Cohérence des types.** `n_in_flight` est `uint32_t` en tâches 1 et 3. `TreeReport.en_vol` est `uint64_t`, exposé sous le même nom en Python, utilisé en tâches 1 et 3. `expand_and_backup_prepared` a la même signature à cinq paramètres en tâches 1 et 3. `Mesure` gagne `batch_size: int` en tâche 5 et la clé d'agrégation passe partout à trois éléments. `batch_size` vaut 32 par défaut en C++ comme dans les bindings.

**Point d'attention pour l'implémenteur.** La tâche 2 change le verrouillage de `step_analysis` : la boucle prenait `m_mutex` par simulation, l'enveloppe le prend une fois pour toute la durée. C'est nécessaire, la boucle batchée détenant des pointeurs bruts pendant l'inférence, mais cela allonge la section critique. Sans conséquence puisque `uci.py` appelle déjà `stop_search()` avant toute modification de l'arbre depuis la correction de la race.
