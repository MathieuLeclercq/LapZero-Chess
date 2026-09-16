# MCTS multicœur par vagues : plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans pour exécuter ce plan en ligne, tâche par tâche, conformément à la préférence de l'utilisateur. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accélérer la préparation des lots de recherche UCI avec plusieurs workers CPU persistants, en conservant un arbre partagé correct et en mesurant le débit et la qualité avant/après.

**Architecture:** Chaque worker possède une copie complète du plateau et collecte des feuilles sous réservation atomique. Les hits TT peuvent publier des enfants pendant la collecte et poursuivre la descente ; les statistiques N/W restent constantes jusqu'à la barrière. Le coordinateur attend tous les collecteurs, appelle seul l'évaluateur, développe les feuilles réseau et effectue les backups.

**Tech Stack:** C++17, MSVC, std::thread/condition_variable, OpenMP existant pour le self-play, CMake/CTest, pybind11 3.0.3, Python 3.13, pytest, uv, ONNX Runtime 1.24.3.

**Spec:** `docs/superpowers/specs/2026-09-14-mcts-multicore-design.md`, révision du 2026-09-16. La demande de rédaction de ce plan valide le passage au plan de l'étape 1.

## Contraintes globales

- Ce plan couvre l'observabilité et les vagues, pas la production continue ni le softmax légal.
- Conserver `worker_count=1` par défaut et le noyau actuel dans ce cas, y compris `batch_size=0`.
- En multicœur : `worker_count >= 2`, `batch_size >= 1`, aucun thread recréé à chaque vague.
- Baseline TT h0 explicite, taille 8192 pour les tests et mesures isolées. Conserver les politiques -1 à 7 et les diagnostics existants.
- h0 reste une approximation choisie pour l'historique antérieur, pas une identité complète des tenseurs réseau.
- Conserver les 119 plans, le modèle, le softmax actuel, le décodage des coups et les règles du plateau.
- Ne jamais partager un `Chessboard` mutable, ni conserver une référence vers une entrée TT après déverrouillage.
- Q utilise seulement les visites terminées. Le virtual loss modifie uniquement `n_in_flight`.
- Exactement le budget demandé de backups par appel réussi ; zéro `pending` et zéro réservation restante au repos.
- Les workers ne font aucun backup ; les expansions TT publient une liste d'enfants construite entièrement avant visibilité.
- Préserver le self-play existant et tester les composants qu'il partage avec l'UCI.
- Tests C++ actifs en Release : utiliser un contrôle qui lève une exception, pas `assert()` supprimé par `NDEBUG`.
- Pas de nouvelle dépendance de test, de migration vers C++20, ni de changement du provider ONNX par défaut.
- Aucun commit de modèle, CSV volumineux ou binaire généré ; conserver les résultats bruts dans `out/multicore/`, déjà ignoré.
- Pas de ligne `Co-Authored-By` ni de tiret cadratin dans les textes ajoutés.

## Précisions d'implémentation issues de la lecture du code

1. `select_leaf()` développe actuellement un hit TT puis continue. Le remplacer par un backup immédiat modifierait la recherche et le sens du budget. Le nouveau collecteur conserve cette sémantique.
2. L'initialisation de racine peut appeler le réseau en plus du budget de simulations. Compter séparément cette inférence et les backups ; ne pas imposer `nn_calls == simulations`.
3. La spec mentionne un test où des workers descendent pendant une inférence bloquée. En phase par vagues, c'est précisément interdit. Le test vérifiera leur immobilité pendant l'inférence, et utilisera un crochet au moment de la réservation pour forcer les collisions pendant la collecte.
4. `search_bench.py` et `puzzle_bench.py` ont encore des défauts h1 malgré le défaut C++ h0. Les commandes de référence passent explicitement h0 ; les nouvelles CLI harmonisent leur défaut sur h0.
5. `--travailleurs` du banc puzzle désigne des processus indépendants avec évaluation CPU. Le nouveau `--search-workers` désignera les threads d'un arbre. Pour mesurer le gain UCI réel, utiliser un seul processus et le GPU.
6. Une durée `steady_clock` dans un worker est du temps écoulé dans ce thread, pas du temps CPU consommé. Les rapports feront cette distinction et nommeront l'inférence « temps d'appel évaluateur », sans prétendre mesurer uniquement les kernels GPU.

## Organisation des fichiers

| Fichiers | Responsabilité |
|---|---|
| `src/search_timing.hpp`, `src/search_timing.cpp` | Chronométrages optionnels et agrégation |
| `src/evaluator.hpp` | Contrat réseau injectable, sans dépendance ONNX |
| `src/evaluation_cache.hpp`, `src/evaluation_cache.cpp` | TT par snapshots et verrous rayés |
| `src/mcts_reservation.hpp`, `src/mcts_reservation.cpp` | Propriété `pending` et chemin réservé RAII |
| `src/search_executor.hpp`, `src/search_executor.cpp` | Threads persistants, réveil et rendez-vous C++17 |
| `src/mcts_wave.hpp`, `src/mcts_wave.cpp` | Contextes, collecte et coordinateur des vagues |
| `src/mcts.hpp`, `src/mcts.cpp`, `src/mcts_batch.cpp`, `src/mcts_observe.cpp` | Intégration et conservation du chemin actuel |
| `tests/cpp/test_support.hpp`, `tests/cpp/controlled_evaluator.hpp` | Vérifications Release et faux réseau |
| `tests/cpp/test_*.cpp` | Tests de composants, interleavings et intégration |
| `python_src/search_bench.py`, `python_src/puzzle_bench.py` | Mesures existantes étendues |
| `python_src/multicore_comparison.py` | Comparaison appariée et décision de non-infériorité |
| `src/bindings.cpp`, `python_src/uci.py` | Interface Python et activation finale mesurée |

Les structures de nœud restent dans `mcts.hpp`. Ne pas profiter du chantier pour découper tout le moteur. Les nouveaux headers utilisent des déclarations anticipées lorsque cela évite une dépendance circulaire.

## Commandes communes

Exécuter depuis la racine du dépôt. Au début de l'exécution, créer une branche de travail `codex/mcts-multicore-waves`, conserver les modifications étrangères éventuelles, et enregistrer le commit de départ. Ne pas lancer les campagnes maintenant : ce document en décrit l'exécution future.

```powershell
$cmakeExe = 'C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
$ctestExe = 'C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe'
$referenceModel = 'python_src/checkpoints_onnx/2026_04_23_23h25_iter316_unsupervised.onnx'
New-Item -ItemType Directory -Force out/multicore | Out-Null
& $cmakeExe -S . -B build
& $cmakeExe --build build --config Release --target chess_engine chess_perft
uv run pytest python_src/tests -q
& $ctestExe --test-dir build -C Release --output-on-failure
```

Pour chaque nouvelle cible CTest, copier `${ONNXRUNTIME_DLL}` à côté de l'exécutable par `POST_BUILD`. Un faux évaluateur n'utilise pas le GPU mais le linkage actuel peut encore demander la DLL principale ONNX. Ajouter `find_package(Threads REQUIRED)` et `Threads::Threads` lors de l'introduction du pool.

Chaque tâche suit le cycle test en échec, implémentation minimale, vérification ciblée, commit des seuls fichiers concernés. Les assertions ci-dessous illustrent les comportements à verrouiller ; les fonctions de test mentionnées sont à créer dans les fichiers indiqués.

## Tâche 1 : référence avant changement et chronométrage optionnel

**Fichiers :** créer `src/search_timing.hpp`, `src/search_timing.cpp`, `tests/cpp/test_search_timing.cpp`, `tests/cpp/test_support.hpp` ; modifier les quatre fichiers MCTS, `src/bindings.cpp`, `python_src/search_bench.py`, `python_src/tests/test_search_bench.py`, `CMakeLists.txt`.

**Interfaces produites :**

```cpp
enum class SearchPhase : size_t {
    Selection, TensorKey, TTProbeStore, TTWait, BoardCopy,
    BatchAssembly, Evaluator, Expansion, Backup, WorkerWait, Count
};
struct SearchTiming {
    bool enabled = false;
    uint64_t wall_ns = 0;
    std::array<uint64_t, static_cast<size_t>(SearchPhase::Count)> elapsed_ns{};
};
class PhaseTimer {
public:
    PhaseTimer(SearchTiming* timing, SearchPhase phase);
    ~PhaseTimer();
};
// MCTS, consultable et configurable entre deux appels :
void set_timing_enabled(bool enabled);
SearchTiming get_last_timing() const;
```

- [ ] **Capturer la référence avec le binaire de départ.** Exécuter la commande suivante avant de modifier le noyau. Enregistrer commit, hash SHA256 du modèle et du `.pyd`, GPU, CPU, versions et statut CUDA dans le rapport. Garder aussi trois distributions de visites et compteurs mono-worker pour les tests de compatibilité.

```powershell
uv run python python_src/search_bench.py --model $referenceModel --gpu --simulations 700 --passages 30 --batch-sizes 8 --cache-history-depths 0 --out-rapport out/multicore/before.md
```

- [ ] **Créer le support C++ et un test qui échoue.** `require(bool, const char*)` lève `runtime_error` si faux. Chaque exécutable attrape les exceptions dans `main()` et retourne 1. Ne pas introduire de framework. Vérifier les accumulateurs sans supposer une durée matérielle précise :

```cpp
SearchTiming timing;
{ PhaseTimer timer(nullptr, SearchPhase::Selection); }
require(timing.wall_ns == 0, "disabled timing changed");
timing.enabled = true;
{ PhaseTimer timer(&timing, SearchPhase::Selection); }
require(timing.elapsed_ns.size() == static_cast<size_t>(SearchPhase::Count),
        "phase storage mismatch");
```

Créer le CTest `search_timing`. Le premier build doit échouer sur les interfaces absentes.

- [ ] **Implémenter les timers.** Si le pointeur vaut `nullptr`, ne pas appeler l'horloge. Chaque thread écrit dans son propre `SearchTiming`. Séparer les scopes, notamment sélection et TT, pour éviter de facturer deux fois le même temps ; documenter les durées volontairement inclusives. La copie initiale du plateau et l'expansion de racine sont incluses dans `wall_ns`. `get_last_timing()` retourne un snapshot du dernier appel, pas un cumul historique.
- [ ] **Ajouter la mesure à la CLI.** `--timings`, `--out-json PATH`, `--repetitions N` (défaut 1), `--tt-size N` (défaut 8192). Conserver `--passages`. Le JSON contient les mesures individuelles et le contexte, pas seulement des moyennes. Ajouter un test avec un faux MCTS où `set_timing_enabled(True)` est effectivement appelé et les données retournées sont conservées.
- [ ] **Vérifier puis mesurer le surcoût désactivé.** Rebuild, `ctest -R search_timing`, `uv run pytest python_src/tests/test_search_bench.py -q`, puis répéter la référence avec les timers désactivés et un court passage activé. Exiger au plus 2 % de hausse de médiane désactivée ; si le bruit thermique est supérieur, alterner les passages avant de conclure. Aucun changement de décision ou de compteurs mono-worker.
- [ ] **Commit :** `Ajoute les chronometrages optionnels de recherche`.

## Tâche 2 : évaluateur injectable et plateaux privés vérifiés

**Fichiers :** créer `src/evaluator.hpp`, `tests/cpp/controlled_evaluator.hpp`, `tests/cpp/test_evaluator.cpp`, `tests/cpp/test_board_copy.cpp`, `tests/cpp/selfplay_phase_bench.cpp` ; modifier `src/onnx_evaluator.hpp`, `src/onnx_evaluator.cpp`, `src/mcts.hpp`, `src/mcts.cpp`, `src/bindings.cpp`, `CMakeLists.txt`.

**Interface produite :**

```cpp
class Evaluator {
public:
    virtual ~Evaluator() = default;
    virtual void evaluate_batch(const std::vector<float>& input,
        std::vector<float>& policies, std::vector<float>& values,
        int batch_size) = 0;
    void evaluate(const std::vector<float>& input,
                  std::vector<float>& policy, float& value);
};
// MCTS(Evaluator* evaluator, size_t tt_size, int cache_history_depth)
// ONNXEvaluator : public Evaluator
```

- [ ] **Écrire un faux réseau déterministe.** `ControlledEvaluator` dérive d'`Evaluator` et expose `batch_sizes`, `fail_on_call` (0 désactive), `before_evaluate` (`std::function<void()>`). Il retourne une policy uniforme de longueur `batch_size * 4672` et une value nulle par défaut, avec une valeur configurable pour tester les signes. Vérifier les dimensions de l'entrée. Le coordinateur est l'unique appelant, donc pas besoin de rendre arbitrairement la session réseau concurrente.

```cpp
ControlledEvaluator evaluator;
MCTS mcts(&evaluator, 8192, 0);
Chessboard board;
mcts.step_analysis(board, 17, 1.4f, 8);
auto report = mcts.inspect_tree();
require(report.violations == 0, "invalid reference tree");
require(!evaluator.batch_sizes.empty(), "network was not called");
```

- [ ] **Tester les copies sur des histoires légales concrètes.** Pour chaque fixture, copier puis jouer/annuler un coup sur la copie ; comparer FEN, Zobrist, tensor, clé h7, taille et contenu des historiques accessibles. Utiliser notamment les cas suivants et exiger que chaque `movePieceUCI()` retourne vrai :

```text
Départ : e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1
Départ : e2e4 a7a6 e4e5 d7d5 e5d6
FEN 7k/P7/8/8/8/8/8/7K w - - 0 1 : a7a8q
Départ : g1f3 g8f6 f3g1 f6g8
```

Exécuter `board_copy` et `evaluator_contract`. Les tests de copie doivent passer sans modification de `chessboard.cpp`.

- [ ] **Adapter uniquement la dépendance réseau.** Déplacer le wrapper `evaluate()` dans `Evaluator`, conserver le même softmax ONNX. Retirer l'allocateur `Ort::AllocatorWithDefaultOptions` inutilisé de `MCTS` si son absence d'utilisation est confirmée. Garder le constructeur Python acceptant `ONNXEvaluator*` via une lambda qui construit un `MCTS`, et ajouter `py::keep_alive<1, 2>()` pour que l'évaluateur survive au MCTS. Pas de faux réseau exposé en Python.
- [ ] **Valider l'équivalence.** Les suites `test_virtual_loss.py`, `test_search_counters.py`, `test_bench_engine.py` passent ; les distributions ONNX sauvegardées en tâche 1 restent conformes à la tolérance numérique existante. Capturer maintenant les distributions et compteurs du faux réseau avant les changements d'atomiques/cache pour les comparaisons exactes suivantes.
- [ ] **Capturer la référence CPU du chemin self-play partagé.** Créer la cible `selfplay_phase_bench` : huit racines indépendantes, même FEN de départ, 100 tours de collecte via `advance_to_leaf()` sous OpenMP8, puis évaluateur et `expand_and_backup()` séquentiels. Recréer les arbres et le MCTS entre les 30 répétitions, TT8192/h0, échauffement exclu. Mesurer séparément collecte et traitement ; imprimer durées brutes et compteurs en JSON. Le programme accepte `--fake` ou `--model PATH --gpu`. Exécuter les deux variantes ici, avant le cache et les atomiques, et conserver la sortie dans `out/multicore/selfplay-before-*.json`. Les tâches 8 et 10 réutilisent exactement cette cible ; aucun entraînement n'est lancé.
- [ ] **Commit :** `Rend l evaluateur MCTS injectable pour les tests`.

## Tâche 3 : cache partagé par snapshots stables

**Fichiers :** créer `src/evaluation_cache.hpp`, `src/evaluation_cache.cpp`, `tests/cpp/test_evaluation_cache.cpp` ; modifier `src/mcts.hpp`, `src/mcts.cpp`, `src/mcts_batch.cpp`, `CMakeLists.txt`.

**Interfaces produites :** déplacer `TTEntry`, `TTProbeStatus`, `TT_MAX_MOVES` dans le header de cache, et remplacer le pointeur de `TTProbe` :

```cpp
struct TTProbe {
    TTProbeStatus status = TTProbeStatus::MISS;
    float value = 0.0f;
    int policy_size = 0;
    std::array<std::pair<int, float>, TT_MAX_MOVES> legal_policy;
};
class EvaluationCache {
public:
    EvaluationCache(size_t size, int history_depth);
    TTProbe probe(const EvaluationCacheKey& key,
                  SearchTiming* timing = nullptr) const;
    void store(const EvaluationCacheKey& key, const std::vector<int>& legal,
               const float* policy, float value, SearchTiming* timing = nullptr);
};
```

- [ ] **Tests discriminants avant migration.** Prévoir 4096 bandes au plus (`min(size, 4096)`), taille nulle refusée. Avec une table de 3 entrées, les hash 1 et 4 ciblent la même entrée : exécuter deux writers et deux readers synchronisés au départ. Un hit doit rendre soit toute la policy A avec sa value A, soit toute la policy B avec sa value B, jamais un mélange. Garder un snapshot A puis écraser la table avec B, le snapshot A doit rester intact.
- [ ] **Tester les contextes.** Utiliser un même plateau avec demi-compteurs 0 et 99 et des policies sentinelles différentes. Une lecture d'une clé ne doit jamais accepter la policy de l'autre. Vérifier `RULE50_REJECT`, `CONTEXT_REJECT`, `HISTORY_REJECT` et `MISS` indépendamment. Conserver les tests Python de h0/h1/h3/h7/legacy.

```cpp
EvaluationCache cache(3, 0);
// key_a/key_b : même position_hash, contextes distincts construits par Chessboard.
cache.store(key_a, legal, policy_a.data(), 0.25f);
auto saved = cache.probe(key_a);
cache.store(key_b, legal, policy_b.data(), -0.75f);
require(saved.status == TTProbeStatus::HIT, "missing snapshot");
require(saved.value == 0.25f, "snapshot changed after store");
require(cache.probe(key_a).status == TTProbeStatus::RULE50_REJECT,
        "rule50 context leaked");
```

- [ ] **Implémenter le verrouillage.** `index = position_hash % size`, `stripe = index % stripe_count`. Le probe copie seulement `policy_size` éléments valides sous verrou. Aucun champ de l'entrée, pas même le hash, n'est lu avant le lock. Garder exactement l'ordre des rejets et la troncature existante à `TT_MAX_MOVES`.
- [ ] **Migrer tous les accès.** Les helpers `MCTS::probe_tt/store_tt` délèguent au cache ; remplacer chaque `probe.entry` par le snapshot. La classification et les compteurs restent dans MCTS. Le self-play profite de la même protection sans nouvel ordonnanceur.
- [ ] **Vérifier.** CTest `evaluation_cache`, puis `uv run pytest python_src/tests/test_evaluation_cache_key.py python_src/tests/test_tt_history_depth.py python_src/tests/test_search_counters.py -q`. Réexécuter la référence courte mono-worker pour repérer le coût des copies et locks.
- [ ] **Commit :** `Securise le cache d evaluation partage`.

## Tâche 4 : publication des nœuds et réservations sans fuite

**Fichiers :** créer `src/mcts_reservation.hpp`, `src/mcts_reservation.cpp`, `tests/cpp/test_reservation.cpp` ; modifier `src/mcts.hpp`, `src/mcts.cpp`, `src/mcts_batch.cpp`, `src/mcts_observe.cpp`, `src/selfplay_manager.cpp`, `src/bindings.cpp`, `CMakeLists.txt`.

**Interfaces produites :**

```cpp
enum class NodeState : uint8_t { Unexpanded, Pending, Expanded, Terminal };
// Dans MCTSNode : atomic<NodeState> state ; atomic<uint32_t> n_in_flight.
class PathReservation {
public:
    PathReservation() = default;
    PathReservation(PathReservation&&) noexcept;
    PathReservation& operator=(PathReservation&&) noexcept;
    PathReservation(const PathReservation&) = delete;
    ~PathReservation();
    void reserve(MCTSNode* node);
    bool try_claim(MCTSNode* node);
    void publish(NodeState state) noexcept;
    void release() noexcept;
private:
    std::vector<MCTSNode*> path_;
    MCTSNode* owned_pending_ = nullptr;
};
```

- [ ] **Tests unitaires.** Réserver racine/enfant, déplacer la garde, lever une exception puis constater les deux compteurs nuls. Faire concourir 8 gardes sur le même nœud avec une barrière de test C++17 : exactement une réussit le CAS, les autres n'annulent jamais la propriété gagnante. Vérifier que Q ne change pas pendant la réservation.

```cpp
MCTSNode root(0.0f);
{
    PathReservation a;
    a.reserve(&root);
    require(a.try_claim(&root), "claim failed");
    PathReservation b(std::move(a));
    require(root.n_in_flight.load() == 1, "move duplicated reservation");
}
require(root.n_in_flight.load() == 0, "reservation leaked");
require(root.state.load() == NodeState::Unexpanded, "pending leaked");
```

- [ ] **Implémenter les garanties d'exception.** `reserve()` ajoute d'abord le pointeur au vecteur, puis incrémente atomiquement : si l'allocation échoue, aucun compteur n'est perdu. `release()` décrémente chaque entrée exactement une fois, remet seulement son propre `Pending` non publié à `Unexpanded`, puis vide le chemin. `publish()` retire la propriété de la garde avant qu'elle ne puisse annuler un état publié. Ne pas créer une méthode « désarmer » qui oublierait la décrémentation.
- [ ] **Unifier les états.** Remplacer `is_terminal` par le test `state == Terminal` dans tous les consommateurs, pour éviter deux sources de vérité. Adapter aussi les chemins mono-worker et self-play. Ils publient `Expanded` après création des enfants et `Terminal` pour les terminaux, tout en gardant leur ordre de sélection.
- [ ] **Garantir la publication complète.** Construire les enfants dans un vecteur local, puis `swap` et store-release de l'état. Un lecteur fait load-acquire de l'état avant de lire `children`. Aucun enfant publié n'est modifié pendant la collecte. Ne pas allouer sous un verrou TT.
- [ ] **Renforcer l'observabilité.** `TreeReport` gagne `pending` et `root_visits`. `inspect_tree()` signale un `Pending` au repos, un `Unexpanded` avec enfants, un `Expanded` vide et les nombres non finis. En interne, un parcours prenant `const MCTSNode*` sert aussi aux racines locales de `mcts_search()` en tests. Garder l'encadrement actuel `sum(child.visits) <= visits <= sum(child.visits)+1`.
- [ ] **Vérifier.** CTest `reservation`, tests existants virtual loss/arbre/compteurs. Modifier temporairement la décrémentation ou le CAS et confirmer qu'un test échoue, puis retirer cette mutation avec `apply_patch`.
- [ ] **Commit :** `Formalise la propriete des feuilles et des reservations`.

## Tâche 5 : exécuteur persistant et rendez-vous C++17

**Fichiers :** créer `src/search_executor.hpp`, `src/search_executor.cpp`, `tests/cpp/test_search_executor.cpp` ; modifier `CMakeLists.txt`.

**Interface produite :**

```cpp
class SearchExecutor {
public:
    SearchExecutor();
    ~SearchExecutor();
    void run(size_t active_workers, const std::function<void(size_t)>& job);
    void shutdown() noexcept;
};
```

`run()` est synchrone côté coordinateur : un job par worker actif, tous terminés avant retour, puis relance de la première exception. Le callback n'est jamais conservé après retour. L'exécuteur ne connaît ni MCTS ni ONNX.

- [ ] **Tests de durée de vie.** Collecter les `std::thread::id` pour 8 jobs, puis 2, puis 8. Les deux premiers IDs doivent rester identiques, aucun job ne doit s'exécuter dans les workers inactifs. Vérifier que le destructeur termine sans travail supplémentaire. Les tests utilisent futures/CV avec timeout comme diagnostic, pas des sleeps supposés provoquer un ordre.

```cpp
SearchExecutor executor;
std::array<int, 8> calls{};
executor.run(8, [&](size_t id) { ++calls[id]; });
executor.run(2, [&](size_t id) { ++calls[id]; });
require(calls[0] == 2 && calls[1] == 2 && calls[7] == 1,
        "wrong active subset");
```

- [ ] **Tester une exception de worker.** Un callback lève ; les autres peuvent terminer ou constater l'arrêt, mais `run()` attend leur sortie avant de relancer. Un nouvel appel `run()` doit fonctionner. Échec de création partielle de threads : réveil/join des threads créés avant propagation.
- [ ] **Implémenter.** Mutex + CV, numéro de génération, compteur de jobs restants, `exception_ptr`, indicateur shutdown. Le pool ne grandit qu'au repos. Les workers attendent une nouvelle génération et `id < active_workers`. Capturer un ID, jamais une référence susceptible d'être invalidée par croissance d'un vecteur. Détruire/vider les captures de job au retour pour ne pas garder une ancienne racine vivante ou pendante.
- [ ] **Vérifier.** CTest `search_executor` avec timeout 30 s. Aucun polling actif entre appels. Pas d'utilisation de `std::barrier` ou `std::jthread`, indisponibles en C++17.
- [ ] **Commit :** `Ajoute les workers persistants de recherche`.

## Tâche 6 : collecteur parallèle et budgets de vague

**Fichiers :** créer `src/mcts_wave.hpp`, `src/mcts_wave.cpp`, `tests/cpp/test_wave_collection.cpp`, `tests/cpp/mcts_test_access.hpp` ; modifier `src/mcts.hpp`, `CMakeLists.txt`.

**Interfaces produites :**

```cpp
enum class LeafKind { Network, Terminal, Collision };
struct LeafWork {
    LeafKind kind = LeafKind::Collision;
    MCTSNode* node = nullptr;
    PathReservation reservation;
    EvaluationCacheKey key;
    std::vector<int> legal_moves;
    std::vector<float> tensor;
    float terminal_value = 0.0f;
};
struct WorkerContext {
    Chessboard board;
    std::vector<LeafWork> results;
    SearchTiming timing;
};
struct WaveTestHooks {
    std::function<void(MCTSNode*)> before_claim;
    std::function<void(MCTSNode*)> before_tt_publish;
};
// Membre privé MCTS, hooks=nullptr en production :
LeafWork collect_wave_leaf(MCTSNode* root, WorkerContext& context,
    float c_puct, const std::atomic<bool>& cancelled,
    const WaveTestHooks* hooks = nullptr);
```

`MCTSTestAccess`, seul friend de test, donne accès au collecteur/cache et à la racine pour les fixtures. Aucun pointeur d'arbre ni hook n'est exposé par pybind.

- [ ] **Forcer les collisions.** Deux collecteurs arrivent au `before_claim` du même nœud grâce à une barrière de test. Exiger une seule feuille Network, une Collision, zéro enfant dupliqué ; détruire les résultats et vérifier les compteurs/états restaurés. Ne pas demander deux appels réseau distincts si deux nœuds différents représentent la même position.
- [ ] **Tester la publication TT.** Injecter une entrée par le cache, bloquer son propriétaire avant publication. L'autre worker doit voir `Pending` et ne jamais parcourir une liste partielle. Après publication, une descente rejoint un descendant ; aucun backup ni incrément N/W n'a eu lieu. Tester un miss à demi-compteur différent et un terminal avant toute lecture de cache.
- [ ] **Implémenter le collecteur.** Réserver la racine, sélectionner avec le PUCT/FPU actuel, réserver l'enfant immédiatement, appliquer le coup sur le plateau privé. Vérifier répétition/50 coups/mat/pat/matériel suivant les conventions actuelles. Une garde locale d'annulation de coups ramène le plateau à sa profondeur initiale sur chaque retour et exception. Ne pas appeler `select_leaf()` actuel depuis les workers, puisqu'il écrit encore sans propriété exclusive.

```text
Expanded : choisir enfant, réserver, jouer, continuer.
Terminal : produire Terminal, sans tentative d'expansion.
Pending : produire Collision, libérer le chemin.
Unexpanded : CAS vers Pending ; échec = Collision.
  terminal selon plateau : publier Terminal, produire Terminal.
  hit TT : construire enfants, publier Expanded, poursuivre la même descente.
  miss TT : capturer coups, clé, tenseur ; transférer la garde dans Network.
```

- [ ] **Implémenter les permis bornés.** Chaque vague a `slots=min(batch_size, remaining)`. Un CAS décrémente `available` avant une tentative. Collision : rendre le slot ; résultat : le conserver jusqu'au commit par le coordinateur. Un compteur d'essais global limite les tentatives à `4*slots + worker_count`. Chaque worker peut enchaîner plusieurs tentatives. Les vectors de résultats sont locaux, puis fusionnés après rendez-vous par `(worker_id, ordinal_local)`. Cet ordre stable n'est pas une promesse de recherche déterministe.
- [ ] **Tester les bornes.** Avec 2 workers et batch 8, une fixture d'arbre déjà visité et suffisamment diversifié doit fournir plus de 2 feuilles. Avec un seul chemin jouable, le lot partiel doit progresser. À budget 3, jamais plus de 3 résultats acceptés. Au repos, une vague vide sans arrêt ni exception est une erreur de logique, pas une boucle `continue` infinie.
- [ ] **Vérifier.** CTest `wave_collection` sans modèle ; comparer chaque tensor capturé au tensor obtenu par rejeu du chemin avec le moteur lui-même. Inclure des transpositions et un historique réel pour vérifier que la capture ne vient pas du plateau racine remis en place.
- [ ] **Commit :** `Collecte les feuilles en parallele avec reservation precoce`.

## Tâche 7 : coordinateur, API et nettoyage de session

**Fichiers :** modifier `src/mcts_wave.cpp`, `src/mcts_wave.hpp`, `src/mcts.hpp`, `src/mcts.cpp`, `src/mcts_observe.cpp`, `src/bindings.cpp` ; créer `tests/cpp/test_wave_search.cpp`, `python_src/tests/test_multicore_api.py` ; modifier `CMakeLists.txt` et les stubs générés.

**Interfaces produites :**

```cpp
void step_analysis(Chessboard& board, int num_simulations, float c_puct,
                   int batch_size = 0, int worker_count = 1);
std::vector<float> mcts_search(Chessboard& board, int num_simulations,
    float c_puct, bool add_dirichlet, int batch_size = 0, int worker_count = 1);
// privé
void run_search_waves(MCTSNode* root, const Chessboard& board,
    int simulations, float c_puct, int batch_size, int worker_count);
```

- [ ] **Tests intégrés avant câblage.** Avec le faux réseau, tester workers `{2,4,8}`, batch `{1,8}`, budgets `{0,1,3,8,17,100}`. À chaque appel, `root_visits_after-root_visits_before == simulations`. Sur une racine terminale, les visites des enfants sont nulles : utiliser `root_visits`, pas leur somme. Vérifier plateau d'entrée intact, états au repos et légalité du meilleur coup lorsqu'il existe.

```cpp
ControlledEvaluator evaluator;
MCTS mcts(&evaluator, 8192, 0);
Chessboard board;
const auto fen = board.toFEN();
mcts.step_analysis(board, 17, 1.4f, 8, 4);
auto r = mcts.inspect_tree();
require(r.root_visits == 17, "wrong completed budget");
require(r.en_vol == 0 && r.pending == 0 && r.violations == 0,
        "session did not become quiescent");
require(board.toFEN() == fen, "input board changed");
```

- [ ] **Implémenter le coordinateur.** Le MCTS possède un exécuteur paresseux et des contextes persistants. En début d'appel, au repos, copier le plateau complet dans les seuls contextes actifs. Après `executor.run()`, assembler les Network, appeler l'évaluateur une fois s'il y en a, valider longueurs et valeurs finies, puis traiter les résultats dans l'ordre convenu. Construire les enfants hors du nœud ; publier ; backup avec alternance de signe existante ; libérer la garde. Le backup et la libération doivent être non lançants après publication.
- [ ] **Compter sans changer les conventions.** `nn_calls` compte les positions et `nn_batches` les appels physiques, y compris l'expansion initiale séparée. `terminal_hits` garde le sens existant : la découverte d'un mat/pat sur miss n'est pas un hit d'un terminal déjà connu. Ajouter `waves`, `leaf_collisions`, `completed_simulations` aux compteurs et leurs tests ; les réservations, collisions et publications TT ne gonflent jamais les simulations terminées.
- [ ] **Tester les erreurs et l'immobilité CPU pendant le réseau.** Faire échouer l'appel réseau 2, donc après l'initialisation de racine. Tester aussi une sortie de dimensions incorrectes. Après exception, zéro réservation/pending, arbre inspectable, nouvelle recherche possible. Bloquer le faux réseau sur CV : le compteur de descentes doit rester constant tant qu'il est bloqué. Utiliser les hooks de collecte pour les collisions, pas une fausse simultanéité CPU/GPU.
- [ ] **Nettoyer sans course.** En cas d'exception de worker, signaler l'annulation et attendre tous les callbacks avant de détruire les résultats détenus par le coordinateur. Chaque worker détruit lui-même sa garde encore locale. Une fois au repos, annuler les résultats non consommés et remettre leurs propriétaires `Pending` à `Unexpanded`. Une expansion alloue localement pour ne laisser aucun enfant partiel. Une exception ne revient pas sur les simulations déjà backupées, mais doit laisser un arbre valide.
- [ ] **Sérialiser la durée de vie.** Protéger les deux points d'entrée publics avec le mutex de session ; rendre ce mutex `mutable`. `update_root`, `reset_analysis`, `get_root_q`, `get_analysis_results`, `inspect_tree` utilisent le même verrou, sans récursion. Les workers ne le prennent jamais. Le destructeur ferme l'exécuteur avant la destruction de la racine. Après retour, vider les captures et pointeurs de la session même si les threads restent en vie.
- [ ] **Câbler pybind sans deadlock GIL.** Garder `gil_scoped_release` pour les recherches. Les getters/reset/update qui peuvent attendre le mutex libèrent aussi le GIL pendant l'attente : sinon un getter Python peut bloquer le retour du thread de recherche. Le binding du constructeur conserve `keep_alive`. Valider workers invalides et multicœur+batch0 avant toute mutation, conserver le comportement historique des appels sans argument workers.
- [ ] **Vérifier.** CTest `wave_search`, `uv run pytest python_src/tests/test_multicore_api.py python_src/tests/test_uci_race.py python_src/tests/test_virtual_loss.py python_src/tests/test_tree_invariants.py -q`. Un test C++ retient l'évaluateur et appelle `update_root` depuis un autre thread : le déplacement ne termine qu'après libération du réseau et retour au repos.
- [ ] **Commit :** `Integre la recherche MCTS par vagues`.

## Tâche 8 : stress et compatibilité du self-play

**Fichiers :** créer `tests/cpp/test_multicore_stress.cpp`, `tests/cpp/test_selfplay_shared_core.cpp` ; modifier `CMakeLists.txt`, et seulement si nécessaire `src/selfplay_manager.hpp/cpp` pour accepter `Evaluator*` au lieu d'`ONNXEvaluator*`.

**Interfaces consommées :** points d'entrée tâche 7, `Evaluator`, MCTS public de self-play. Aucune modification de l'ordonnancement OpenMP.

- [ ] **Cent recherches par configuration.** Les trois FEN de `search_bench.py`, workers `{2,4,8}`, batch `{1,8}`, 64 simulations. Alterner racines neuves et réutilisées, puis alterner workers `8 -> 2 -> 1 -> 4`. Contrôler invariants après chaque appel, histogramme des lots et budget réel. `mcts_search()` est couvert via le parcours interne de sa racine locale avant destruction.
- [ ] **Tester les cas terminaux.** Fixtures existantes des tests de finales : mat, pat, règle des 50 coups, répétition et matériel insuffisant. Ajouter le signe d'une value non nulle à profondeurs paires/impaires et la découverte d'un mat à plusieurs workers. Ne pas simplement vérifier que « ça ne plante pas ».
- [ ] **Tester le chemin self-play partagé.** Une instance MCTS, huit plateaux/racines distincts. Appeler `advance_to_leaf()` sous `#pragma omp parallel for num_threads(8)`, puis évaluer et `expand_and_backup()` séquentiellement comme le manager. Inclure des clés qui écrasent la même entrée TT. Vérifier toutes les racines après plusieurs tours. Avec l'évaluateur injectable, exécuter aussi `SelfPlayManager` sur deux parties, budgets 1/2, TT8192 : dimensions des données cohérentes, policies finies, aucune fuite. Conserver la borne existante de 300 plies.
- [ ] **Valider le filet complet une fois.** Rebuild Release de toutes les cibles, CTest complet, `uv run pytest python_src/tests -q`, `./build/Release/chess_perft.exe bench --strict --check-fen`. Ne pas relancer la campagne perft profonde ni 200000 positions de fuzz sans changement du plateau ou anomalie détectée.
- [ ] **Prouver que les gardes détectent des erreurs.** Neutraliser une seule fois la remise à zéro de `Pending`, vérifier l'échec du test d'exception, restaurer ; neutraliser le CAS de propriété, vérifier l'échec du test de collision, restaurer. Aucun invariant affaibli pour obtenir une suite verte.
- [ ] **Commit :** `Verrouille les invariants concurrents et le self play partage`.

## Tâche 9 : bancs comparables et non-infériorité

**Fichiers :** modifier `python_src/search_bench.py`, `python_src/puzzle_bench.py`, `python_src/bench_metrics.py` seulement pour le contexte de rapport, leurs tests ; modifier `src/evaluation_cache.hpp/cpp`, `src/mcts.hpp/cpp`, `src/bindings.cpp` pour le reset de cache au repos ; créer `python_src/multicore_comparison.py`, `python_src/tests/test_multicore_comparison.py`.

**CLI produites :**

```text
search_bench.py : --worker-counts 1 2 4 8, --timings,
                 --passages 5 --repetitions 6, --out-json PATH,
                 --tt-size 8192, --warmup-pool
puzzle_bench.py : --search-workers N, --gpu,
                 --search-seconds SECONDS (exclusif de --simulations explicite)
multicore_comparison.py : --baseline A.csv --candidate B.csv
                         --out-report R.md --seed 42
```

`--travailleurs` conserve son sens de nombre de processus. En mode GPU du banc puzzle, refuser `--travailleurs != 1` afin de ne pas multiplier les sessions GPU ; les workers de recherche sont internes au C++. Une erreur de puzzle doit produire un statut d'échec de campagne, pas un CSV partiellement comparé en silence.

- [ ] **Tests de propagation.** Un faux MCTS capture `(batch_size, worker_count)` et la profondeur h0 depuis chaque CLI. Vérifier `--search-workers=8` sans transformer les 8 en processus Python. Les groupes du banc de débit incluent workers et état froid/chaud du pool. La somme des temps des workers ne remplace jamais le temps mur.
- [ ] **Mesurer froid et chaud.** Pour le chemin UCI, conserver le MCTS et son pool entre les répétitions ; remettre l'arbre et la TT à froid hors chronométrage grâce à `void EvaluationCache::clear()`, exposé en `void MCTS::clear_evaluation_cache()`. Cette API prend le verrou de session et n'est utilisée qu'au repos, jamais pendant le self-play. Tester qu'un ancien hit devient un miss sans recréer les threads. Mesurer séparément le premier appel incluant la création des threads. Le chemin `mcts_search()` réutilise aussi l'objet MCTS, mais crée sa racine locale à chaque appel. Les resets ne reconstruisent pas le pool.
- [ ] **Ajouter la jambe GPU du banc puzzle.** Réutiliser le chargement d'historique et le scoring existants. Utiliser `ONNXEvaluator(onnx, True)` avec import torch pour les DLL comme le search bench. La policy brute peut garder la session CPU existante ; son coût est exclu du temps de recherche. À simulations égales, créer un MCTS neuf par puzzle. Pour le temps fixe, réutiliser le pool entre puzzles, effacer arbre et TT hors fenêtre, puis faire `step_analysis()` par tranches de 8 simulations jusqu'au délai. Compter le temps initial d'expansion de racine, enregistrer le temps réellement dépassé et les simulations terminées. Ne pas présenter un budget de simulations estimé comme une mesure à temps égal.
- [ ] **Figer le scoring.** Critère principal : premier coup de recherche égal au premier coup solution, colonne `reussi_recherche`. Conserver le même échantillon régulier stratifié, les IDs de lignes et l'historique ; aucune sélection de puzzles d'après le résultat d'une configuration. Ajouter workers, provider, hashes modèle/banc, budget et timings au contexte/sidecar de chaque CSV.
- [ ] **Comparer les paires strictement.** Refuser doublons, lignes manquantes, erreurs et métadonnées incompatibles. Ne pas prendre silencieusement l'intersection des CSV. Produire delta candidat-moins-référence, IC95 bootstrap apparié (20000 tirages, seed42), paires discordantes et McNemar via l'aide existante.

```python
# Contrat de la fonction pure à créer, taux exprimés en fractions.
def paired_interval(baseline: list[bool], candidate: list[bool],
                    seed: int = 42, draws: int = 20000) -> tuple[float, float, float]:
    import numpy as np
    if not baseline or len(baseline) != len(candidate):
        raise ValueError("paires vides ou de tailles differentes")
    delta = np.asarray(candidate, dtype=float) - np.asarray(baseline, dtype=float)
    rng = np.random.default_rng(seed)
    samples = np.empty(draws)
    for i in range(draws):
        samples[i] = delta[rng.integers(len(delta), size=len(delta))].mean()
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(delta.mean()), float(low), float(high)
```

Tester signes et reproductibilité avec 2 gains/1 perte, tout identique, tailles différentes et CSV avec une ligne retirée. Les tirages portent sur les paires complètes. Le bootstrap mesure l'incertitude sur les puzzles pour ces exécutions, pas toute la variabilité d'ordonnancement multicœur ; le rapport doit le préciser.

- [ ] **Définir les verdicts.** Campagne complète : `low >= -0.01` permet la non-infériorité. `high < -0.01` indique une régression. Entre les deux : résultat indéterminé, pas d'activation par défaut ni d'assouplissement du seuil. Sur les 500 puzzles de préfiltre, une régression claire arrête la campagne ; une incertitude large ne doit pas empêcher les 2500 nécessaires à la conclusion. Conserver la marge de -1 point de la spec, sans la confondre avec un pourcentage relatif.
- [ ] **Vérifier.** Tests de harnais, puis six puzzles et 32 simulations avec workers1/2, CPU puis GPU si disponible. Un faux chronomètre vérifie que le mode temps inclut le dernier appel complet et rapporte le dépassement. Pas de campagne longue dans cette tâche.
- [ ] **Commit :** `Etend les bancs au multicœur et aux comparaisons appariees`.

## Tâche 10 : mesures réelles, décision et activation UCI

**Fichiers :** créer `docs/superpowers/specs/2026-09-16-multicore-waves-results.md` ; modifier `python_src/uci.py`, `python_src/tests/test_uci_race.py`, `docs/backlog.md` pour marquer le batching déjà terminé et dater le résultat multicœur. Ne pas présenter la production continue comme réalisée.

**Livrable :** rapport avant/après vérifiable, réglage UCI justifié ou maintien explicite d'un worker.

- [ ] **Mesurer le débit, un seul processus GPU.** Le mode timers désactivés fournit les performances ; une courte campagne séparée `--timings` explique les coûts. Alterner l'ordre workers1/2/4/8 entre passages et échauffer chaque taille de lot. Les médianes et p95 sont calculés par position et point d'entrée, sur 30 observations par configuration.

```powershell
uv run python python_src/search_bench.py --model $referenceModel --gpu --simulations 700 --passages 5 --repetitions 6 --batch-sizes 8 --worker-counts 1 2 4 8 --cache-history-depths 0 --tt-size 8192 --warmup-pool --out-json out/multicore/after.json --out-rapport out/multicore/after.md
uv run python python_src/search_bench.py --model $referenceModel --gpu --simulations 700 --passages 1 --repetitions 3 --batch-sizes 8 --worker-counts 1 2 4 8 --cache-history-depths 0 --timings --warmup-pool --out-json out/multicore/phases.json --out-rapport out/multicore/phases.md
```

Comparer à la fois ancien binaire -> nouveau workers1, puis nouveau workers1 -> workers2/4/8. Régression d'infrastructure visible en mono-worker : corriger avant de conclure au gain multicœur. Rapporter aussi le coût de création du pool et de copie du plateau. La médiane doit s'améliorer de façon reproductible ; le p95 ne doit pas augmenter de plus de 5 % sur une des positions testées. Un p95 instable sur 30 valeurs exige de nouvelles observations, pas une certitude artificielle.

- [ ] **Vérifier l'absence de régression du chemin self-play.** Rejouer `selfplay_phase_bench` de la tâche 2 avec `--fake`, puis `--model $referenceModel --gpu`, et comparer aux JSON de référence conservés. Séparer le coût CPU des nouveaux atomiques/cache de celui du réseau. Ne pas lancer de boucle d'entraînement. Consigner toute régression reproductible et la résoudre avant activation UCI.
- [ ] **Préfiltre qualité sur 500 puzzles.** Référence workers1 contre workers8. Si workers2 ou 4 gagne en débit, le considérer comme candidat distinct et lui faire passer la même validation, sans supposer que le résultat workers8 le couvre.

```powershell
foreach ($searchWorkers in @(1, 8)) {
  uv run python python_src/puzzle_bench.py --model $referenceModel --banc data/puzzles_bench.txt --gpu --travailleurs 1 --search-workers $searchWorkers --batch-size 8 --cache-history-depth 0 --simulations 700 --limite 500 --out-csv "out/multicore/prefilter-w$searchWorkers.csv" --out-rapport "out/multicore/prefilter-w$searchWorkers.md"
}
uv run python python_src/multicore_comparison.py --baseline out/multicore/prefilter-w1.csv --candidate out/multicore/prefilter-w8.csv --out-report out/multicore/prefilter-comparison.md --seed 42
```

- [ ] **Campagne complète sur 2500 puzzles.** Après le préfiltre, annoncer la durée estimée à partir du débit réellement observé. Reprendre les commandes précédentes avec `--limite 2500` et préfixe `full-`, uniquement pour la référence et le candidat retenu. Ne pas réutiliser une ancienne campagne h1 ou CPU comme référence h0/GPU. Si l'IC est près du seuil, répéter le candidat multicœur avant de décider ; ne pas relancer tous les modes inutilement.
- [ ] **Mesure à temps égal.** Si la qualité à simulations égales passe, choisir un budget de temps égal à la médiane observée des 700 simulations mono-worker, puis exécuter la même sélection de 500 puzzles avec `--search-seconds` et un seul processus GPU. Workers1 et candidat reçoivent exactement ce budget. Rapporter résolutions, simulations, médiane/p95 de dépassement réel. Cela mesure un gain tactique pratique, pas un gain Elo établi.
- [ ] **Rédiger le rapport compact.** Inclure les hashes/commits, modèle, matériel/provider, h0/TT/batch/workers, avant/après avec dispersion, phases limitantes, collisions/remplissage, mémoire, tests exécutés, IC de qualité, verdict. Mentionner la réserve des TT hits en partie réelle et le coût d'inférence initial de racine. Les CSV restent dans `out/`, le rapport inclut suffisamment de données agrégées pour lire la décision.
- [ ] **Câbler UCI et fixer le défaut selon le verdict.** Ajouter `MCTS_WORKER_COUNT = 1` à côté de `MCTS_BATCH_SIZE = 8`, puis passer ce nombre à `step_analysis()`. Le remplacer par le candidat validé seulement si tous les critères sont franchis. Un test de transmission vérifie le cinquième argument réel ; conserver le test d'arrêt avant `parse_position()`.

```python
self.mcts.step_analysis(
    self.board, sims_to_do, 1.4, MCTS_BATCH_SIZE, MCTS_WORKER_COUNT)
```

- [ ] **Validation finale proportionnée.** Si seul le réglage UCI a changé depuis la suite complète, rejouer les tests UCI/API et un smoke `uci`, `isready`, `position startpos`, `go nodes`, `stop`, `quit` via un subprocess avec timeout. Vérifier bestmove légal, sortie propre, aucune recherche restante. Pas de connexion au compte Lichess pour cette validation.
- [ ] **Commit :** `Documente les gains multicœurs et configure le moteur UCI` si activé, sinon `Documente les mesures multicœurs et conserve le defaut mono worker`. Ne pas fusionner ni pousser sans demande correspondante.

## Ordre et couverture de la spec

```text
1 référence/timers -> 2 évaluateur -> 3 cache -> 4 réservations
-> 5 pool -> 6 collecte -> 7 coordinateur/API -> 8 stress
-> 9 bancs -> 10 mesures/décision
```

- Spec 2/3, plateaux et compatibilité : tâches 1, 2, 8.
- Spec 4, composants et API : tâches 2, 4, 5, 7.
- Spec 5, vagues et budgets : tâches 6, 7.
- Spec 7, TT : tâche 3, vérifiée également par 6 et 8.
- Spec 8, chronométrages : tâches 1, 9, 10.
- Spec 9/10, invariants et erreurs : tâches 4 à 8.
- Spec 11/12, mesures et sortie : tâches 9, 10.
- Spec 6, production continue : volontairement hors exécution ; le rapport fournit les mesures permettant de décider sa suite.

La fin de ce plan livre une recherche par vagues utilisable et testée, même si les mesures conduisent à conserver le réglage UCI mono-worker. Elle ne lance pas automatiquement l'étape de production continue.
