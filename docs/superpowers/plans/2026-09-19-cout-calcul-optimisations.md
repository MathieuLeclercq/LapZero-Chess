# Cout de calcul de la recherche : plan d'implementation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans pour executer ce plan en ligne, tache par tache, conformement a la preference de l'utilisateur. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduire le cout de l'appel a l'evaluateur (forme de lot, buffers, allocations), mesurer et regler la divergence de collecte, evaluer FP16, et quantifier le cas chaud, avec mesures avant/apres et validation qualite.

**Architecture:** Les taches 1 a 3 assainissent l'appel ONNX sans modifier la semantique de recherche : buffers persistants, lot de forme fixe, et banc hors arbre pour mesurer la vraie courbe du GPU. La tache 4 rend la divergence parametrable et la balaie aux tranches reelles du bot. La tache 5 evalue un export FP16. La tache 6 mesure le cas chaud. La tache 7 decide, documente et fixe les defauts. La production continue est hors perimetre : elle fera l'objet d'un plan separe, alimente par les mesures de la tache 7.

**Tech Stack:** C++17, MSVC, CMake/CTest, pybind11 3.0.3, Python 3.13, pytest, uv, ONNX Runtime 1.24.3 CUDA, PyTorch pour l'export.

**Spec:** `docs/superpowers/specs/2026-09-19-cout-calcul-cpu-gpu.md`. Mesures et decision du chantier precedent : `docs/superpowers/specs/2026-09-16-multicore-waves-results.md`.

**Statut d'execution (2026-09-19) :** taches 1 et 3 faites, plus l'extension du lot fixe au chemin mono. Tache 5 tranchee par la mesure : FP16 rejete. Taches 2, 4, 6 et 7 restantes. Resultats : `2026-09-19-cout-calcul-resultats.md`.

## Contraintes globales

- Conserver `worker_count=1` par defaut et le noyau sequentiel inchange.
- Les taches 1 a 3 ne changent pas la semantique de recherche : memes visites, memes compteurs, memes priors, memes coup principal. Le padding de lot ne doit pas modifier les resultats des positions reelles.
- Toute modification de la divergence (tache 4) ou de la precision du reseau (tache 5) exige une validation qualite par le banc de puzzles, prefiltre 500 puis campagne 2500 si le prefiltre franchit la non-inferiorite.
- Mesurer aux tranches reelles du bot, 64 et 20 simulations, en plus des appels longs de 700 simulations.
- Les valeurs absolues varient de 15 a 30 pour cent entre sessions. Toute comparaison avant/apres doit etre interleaved ou faite dans la meme session, avec ordre alterne.
- Ne pas committer de binaire genere, de modele ni de CSV volumineux. Les resultats bruts restent dans `out/multicore/`, deja ignore.
- Pas de ligne `Co-Authored-By` ni de tiret cadratin dans les textes ajoutes.
- Tests C++ actifs en Release : utiliser un controle qui leve une exception, pas `assert()` supprime par `NDEBUG`.
- Pas de nouvelle dependance sans justification. `onnxconverter-common` n'est ajoute qu'en tache 5, et seulement si l'export FP16 par torch ne suffit pas.

## Precisions d'implementation issues de la lecture du code

1. `ONNXEvaluator::evaluate_batch` (`src/onnx_evaluator.cpp:24-85`) cree un tenseur qui enveloppe le buffer d'entree, appelle `session->Run`, redimensionne `policies` et `values`, puis fait le softmax complet sur 4672 sorties par position.
2. `run_search_waves` (`src/mcts_wave.cpp:398-436`) declare `batch_input`, `policies` et `values` a chaque vague, donc une allocation par vague. Le chemin mono (`src/mcts_batch.cpp:87-94`) les declare une fois avant la boucle.
3. `Evaluator::evaluate` (`src/evaluator.hpp:14-20`) alloue un `std::vector<float> values(1)` a chaque appel, donc a chaque expansion de racine.
4. La forme du lot est variable : racine a 1 (`expand_node_single`), vagues a 1 a 8 selon les collisions et le budget restant. Le banc self-play, lui, evalue toujours 8 et coute 0.38 ms par position, contre 1.4 a 2 ms dans la recherche (spec section 6.1).
5. Les collisions de collecte sont comptees (`m_leaf_collisions`) mais leur cout CPU n'est pas chronometre separement. Le rapport multicœur mesure 3000 a 3300 collisions par tranche de 704 simulations en milieu avec 8 workers.
6. `c_puct` est deja un parametre. L'amplitude du virtual loss (`src/mcts_reservation.cpp:36`, `fetch_add(1)`), le coefficient FPU (`src/mcts.cpp:271`, `src/mcts_wave.cpp:219`, `0.30f`) et le facteur de tentatives (`src/mcts_wave.cpp:284`, `4 * slots + worker_count`) sont des constantes.
7. Le bot decoupe en tranches de `BATCH_SIZE = 64` (`python_src/uci.py:23` et `:358`), avec `MCTS_BATCH_SIZE = 8` et `MCTS_WORKER_COUNT = 8` (`python_src/uci.py:28` et `:34`).
8. `Mesure` (`python_src/search_bench.py:44-63`) ne porte pas `leaf_collisions` ; le compteur existe cote C++.

## Organisation des fichiers

| Fichiers | Responsabilite |
|---|---|
| `tests/cpp/evaluator_batch_bench.cpp` | banc hors arbre : courbe du lot, variantes de forme, timings internes |
| `src/onnx_evaluator.hpp`, `src/onnx_evaluator.cpp` | timings internes optionnels, padding, buffers |
| `src/evaluator.hpp` | contrat sans allocation par appel |
| `src/mcts.hpp`, `src/mcts.cpp` | buffers de session, option de lot fixe, parametres de divergence |
| `src/mcts_wave.cpp`, `src/mcts_batch.cpp` | usage des buffers, padding, compteurs |
| `src/mcts_reservation.hpp`, `src/mcts_reservation.cpp` | amplitude du virtual loss |
| `src/bindings.cpp` | exposition minimale des nouveaux reglages |
| `python_src/search_bench.py`, `python_src/tests/test_search_bench.py` | variantes, tranches, collisions |
| `python_src/hot_tree_bench.py`, `python_src/tests/test_hot_tree_bench.py` | cas chaud UCI |
| `python_src/puzzle_bench.py` | export FP16, CLI de conversion |
| `CMakeLists.txt` | cible du banc et copie des DLL |
| `docs/superpowers/specs/2026-09-19-cout-calcul-resultats.md` | mesures finales et decision |

## Commandes communes

Executer depuis la racine du depot. Creer une branche de travail `codex/cout-calcul-optimisations` au debut de l'execution et enregistrer le commit de depart.

```powershell
$cmakeExe = 'C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
$ctestExe = 'C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe'
$referenceModel = 'python_src/checkpoints_onnx/2026_04_23_23h25_iter316_unsupervised.onnx'
New-Item -ItemType Directory -Force out/multicore | Out-Null
& $cmakeExe -S . -B build
& $cmakeExe --build build --config Release
uv run pytest python_src/tests -q
& $ctestExe --test-dir build -C Release --output-on-failure
```

Chaque nouvelle cible CTest copie `${ONNXRUNTIME_DLL}` a cote de l'executable par `POST_BUILD`. Le banc GPU copie aussi les DLL des providers, comme `selfplay_phase_bench`. Chaque tache suit le cycle test en echec, implementation minimale, verification ciblee, commit des seuls fichiers concernes.

## Tache 1 : banc evaluateur hors arbre et timings internes

**Fichiers :** creer `tests/cpp/evaluator_batch_bench.cpp` ; modifier `src/onnx_evaluator.hpp`, `src/onnx_evaluator.cpp`, `CMakeLists.txt`.

**Interfaces produites :**

```cpp
struct EvaluatorTiming {
    std::uint64_t run_ns = 0;      // session->Run
    std::uint64_t softmax_ns = 0;  // boucle C++ sur les 4672 sorties
};
// ONNXEvaluator
void set_timing_enabled(bool enabled);
EvaluatorTiming get_last_timing() const;
```

CLI du banc : `--fake` ou `--model PATH [--gpu]`, `--batches 1 2 4 8 16 32 64 128 256`, `--repetitions N`, `--rounds N` (vagues par repetition), `--variable` (alterne 1 a 8), `--out-json PATH`. Sortie JSON par lot : `ms_appel`, `ms_position`, `run_ns`, `softmax_ns`.

- [ ] **Ecrire le test en echec.** Le CTest `evaluator_batch_bench_smoke` lance `--fake --batches 1 4 8 --repetitions 2 --rounds 2` et verifie par `PASS_REGULAR_EXPRESSION` la presence des trois lots, des compteurs et du champ de timing. Le premier build doit echouer sur la cible absente.
- [ ] **Implementer les timings.** Deux `steady_clock` dans `evaluate_batch`, uniquement si `m_timing_enabled`. Cout nul quand desactive, ce qui est le defaut de production.
- [ ] **Implementer le banc.** Tenseurs aleatoires de taille fixe par lot, echauffement, repetitions, puis mode `--variable` qui alterne 1 a 8 comme le fait la recherche. Avec `--fake`, utiliser `ControlledEvaluator` pour valider le harnais sans modele.
- [ ] **Verifier.** Rebuild, `ctest -R evaluator_batch_bench`, puis campagne GPU reelle : `--model $referenceModel --gpu --batches 1 2 4 8 16 32 64 128 256 --repetitions 30 --out-json out/multicore/evaluator-batch-gpu.json`, et la meme en CPU. Repondre a la question de la spec : le cout par position chute-t-il entre 8 et 128, ou le GPU est-il deja a son regime.
- [ ] **Commit :** `Mesure la courbe du lot de l evaluateur hors arbre`.

## Tache 2 : buffers persistants et suppression des allocations par appel

**Fichiers :** modifier `src/evaluator.hpp`, `src/onnx_evaluator.hpp`, `src/onnx_evaluator.cpp`, `src/mcts.hpp`, `src/mcts_wave.cpp`, `src/mcts_batch.cpp`, `tests/cpp/test_evaluator.cpp`.

**Interfaces produites :**

```cpp
// Evaluator : plus d'allocation par appel
void evaluate(const std::vector<float>& input, std::vector<float>& policy,
              std::vector<float>& values_scratch, float& value);
// MCTS : buffers de session, dimensionnes une fois par appel
std::vector<float> m_wave_input;
std::vector<float> m_wave_policies;
std::vector<float> m_wave_values;
```

- [ ] **Ecrire le test en echec.** Un faux evaluateur expose la capacite des buffers recus. Apres une recherche a batch 8, la capacite de `policies` doit rester superieure ou egale a `8 * 4672` et le second appel ne doit pas reduire la capacite. Le test rougit sur l'interface actuelle, qui alloue par appel.
- [ ] **Implementer.** Deplacer `batch_input`, `policies` et `values` de `run_search_waves` vers des membres de `MCTS`, reserves une fois a `batch_size * 119 * 64` et `batch_size * 4672`, redimensionnes par `assign` ou `resize` sans reallocation. Faire de meme pour le scratch de `Evaluator::evaluate`, qui ne doit plus creer de `values(1)` par appel.
- [ ] **Verifier.** Tous les tests C++ et Python existants, puis mesure interleaved avec le banc de recherche : variante actuelle contre variante buffers, aux tranches 64 et 20 et a 700 simulations, workers 8, pool chaud. Objectif : aucune regression de mediane, aucune hausse du p95 de latence au-dela de 5 pour cent, aucun changement de compteurs ni de visites.
- [ ] **Commit :** `Reutilise les buffers d evaluation entre les vagues`.

## Tache 3 : lot de forme fixe et variantes d'appel

**Fichiers :** modifier `src/mcts.hpp`, `src/mcts.cpp`, `src/mcts_wave.cpp`, `tests/cpp/test_wave_search.cpp`, `python_src/search_bench.py`, `python_src/tests/test_search_bench.py`.

**Interfaces produites :**

```cpp
// MCTS
void set_fixed_batch(bool enabled);  // defaut false
bool fixed_batch() const;
// search_bench.py
// --fixed-batch, --io-binding si la variante est implementee
```

Principe du padding : quand une vague collecte `network_count` feuilles reelles avec `network_count < batch_size`, le coordinateur duplique le dernier tenseur jusqu'a `batch_size`, appelle l'evaluateur avec `batch_size`, et n'utilise que les `network_count` premieres sorties. Les compteurs gardent leur sens : `nn_calls` compte les positions reelles, `nn_batches` les appels physiques.

- [ ] **Ecrire le test en echec.** Avec un faux evaluateur qui enregistre les tailles de lot, une recherche de 17 simulations a batch 8 doit produire trois appels de taille 8 (8, 8, 1 padde a 8). Les visites, les priors, le nombre de noeuds et les compteurs doivent etre identiques a la version non paddee, au bit pres pour les positions reelles.
- [ ] **Implementer le padding.** Etendre le controle de dimensions de `run_search_waves` (`src/mcts_wave.cpp:422`) a `batch_size * 4672` quand le mode est actif. Laisser la racine a batch 1 : deux formes seulement, 1 et 8, au lieu de 1 a 8.
- [ ] **Mesurer les variantes.** Dans la meme session et en alternance : actuelle, buffers seuls, lot fixe, puis lot fixe plus IO binding si implemente. Aux tranches 64 et 20 et a 700 simulations, workers 8, pool chaud. Critere d'adoption : mediane en hausse d'au moins 10 pour cent contre la variante buffers, p95 de latence sous plus 5 pour cent, aucun changement de resultat de recherche.
- [ ] **Verifier.** `ctest -R wave_search`, suite Python complete, puis la campagne de variantes et son JSON dans `out/multicore/fixed-batch-variants.json`.
- [ ] **Commit :** `Padde les lots d evaluation a une forme fixe` si la variante gagne, sinon `Documente le rejet du lot de forme fixe` et retrait du code.

## Tache 4 : parametres de divergence et balayage

**Fichiers :** modifier `src/mcts.hpp`, `src/mcts.cpp`, `src/mcts_wave.cpp`, `src/mcts_reservation.hpp`, `src/mcts_reservation.cpp`, `src/bindings.cpp`, `python_src/search_bench.py`, `python_src/tests/test_search_bench.py`, `tests/cpp/test_wave_collection.cpp`.

**Interfaces produites :**

```cpp
struct SearchTuning {
    float virtual_loss = 1.0f;        // unites par descente
    float fpu_reduction = 0.30f;      // coefficient du terme FPU
    int collision_attempt_factor = 4; // 4 * slots + workers
};
void set_tuning(const SearchTuning& tuning);
SearchTuning get_tuning() const;
```

CLI du banc : `--virtual-loss`, `--fpu`, `--collision-attempts`. `Mesure` gagne `leaf_collisions` et le JSON l'expose.

- [ ] **Ecrire les tests en echec.** Les defauts ne changent rien : une recherche avec `set_tuning(SearchTuning{})` produit les memes compteurs qu'avant. Une amplitude de 2 se propage bien a `n_in_flight` (verifiable via `inspect_tree().en_vol` en cours de collecte avec les hooks de test). Un coefficient FPU negatif est refuse.
- [ ] **Implementer.** Remplacer les constantes par les champs, en gardant les valeurs actuelles par defaut. `PathReservation::reserve` prend l'amplitude en parametre ou la lit dans le MCTS appelant.
- [ ] **Verifier.** Tests cibles, puis balayage croise amplitude {1, 2, 3} x c_puct {1.0, 1.4, 2.0} x FPU {0.2, 0.3, 0.5}, a 700 simulations et aux tranches 64 et 20, workers 8, pool chaud. Rapporter remplissage, collisions, mediane et p95 de latence, temps evaluateur. Criteres : mediane en hausse d'au moins 5 pour cent aux tranches reelles, p95 sous plus 5 pour cent, collisions en baisse.
- [ ] **Validation qualite.** Chaque configuration gagnante passe le prefiltre 500 puzzles contre le reglage actuel, puis la campagne 2500 si la non-inferiorite est proche du seuil. Aucune modification de defaut sans ce passage.
- [ ] **Commit :** `Rend la divergence de collecte parametrable`.

## Tache 5 : export FP16 et comparaison

**Fichiers :** modifier `python_src/puzzle_bench.py`, `src/onnx_evaluator.cpp` si necessaire, `python_src/tests/test_bench_engine.py`, `CMakeLists.txt` si un test C++ est ajoute.

- [ ] **Ecrire le test en echec.** Un test verifie que la conversion produit un fichier `.fp16.onnx` lisible et que `ONNXEvaluator` l'ouvre sans erreur. Avec le modele de reference, comparer policy et value a la version FP32 sur quelques positions : tolerance a fixer, par exemple 1e-3 en absolu sur la policy renormalisee et 1e-2 sur la value.
- [ ] **Implementer la conversion.** D'abord par export torch en half. Si le graphe produit n'est pas supporte par le provider CUDA, ajouter `onnxconverter-common` et convertir l'ONNX existant. Le fichier FP16 reste dans `python_src/checkpoints_onnx/`, ignore par git.
- [ ] **Mesurer.** Banc evaluateur (tache 1) et banc de recherche, FP32 contre FP16, a forme fixe et aux tranches reelles. Puis prefiltre 500 puzzles, et campagne 2500 si la non-inferiorite est proche du seuil.
- [ ] **Commit :** `Evalue un export FP16 du reseau` si le gain est reel, sinon `Documente le rejet de l export FP16`.

## Tache 6 : cas chaud de l'arbre UCI

**Fichiers :** creer `python_src/hot_tree_bench.py`, `python_src/tests/test_hot_tree_bench.py` ; modifier `python_src/tests/test_bench_engine.py` si le modele est partage.

Objectif : quantifier la fraction de simulations qui evite le reseau en partie reelle, et donc la loi d'Amdahl sur tout le chantier.

- [ ] **Ecrire le test en echec.** La logique pure d'agregation, sans moteur, sur des mesures factices : taux de hits, simulations par coup, nps par coup, part de l'expansion de racine.
- [ ] **Implementer le banc.** Jouer une sequence de coups avec un `MCTS` persistant : `step_analysis` par tranches de 64 puis `update_root`, sur une ouverture, un milieu et une finale. Rapporter par coup : simulations, duree, hits et rejets TT, cout de l'expansion de racine, nps.
- [ ] **Verifier.** Test pur, puis execution courte reelle et JSON dans `out/multicore/hot-tree.json`.
- [ ] **Commit :** `Mesure le cas chaud de l arbre UCI`.

## Tache 7 : mesures finales, decision et rapport

**Fichiers :** creer `docs/superpowers/specs/2026-09-19-cout-calcul-resultats.md` ; modifier `docs/superpowers/specs/2026-09-19-cout-calcul-cpu-gpu.md` (statut et renvoi), `python_src/uci.py` et ses tests si un defaut change, `docs/backlog.md`.

- [ ] **Mesurer l'etat final.** Meme session et ordre alterne : banc evaluateur, banc de recherche workers 1/2/4/8 avec pool chaud, aux tranches 64 et 20 et a 700 simulations, et banc de puzzles pour tout reglage retenu. Conserver les JSON dans `out/multicore/`.
- [ ] **Decider des defauts.** Lot fixe, FP16, reglages de divergence. Chaque changement de defaut doit avoir franchi mediane, p95 et qualite. Sinon, le defaut reste inchange et la variante est documentee comme rejetee.
- [ ] **Statuer sur la production continue.** Avec les couts mesures (collecte par feuille, evaluateur par lot, remplissage), estimer le gain d'une file d'inference et decider si elle merite son propre plan. Ne pas l'implementer ici.
- [ ] **Rediger le rapport.** Inclure les commits, le modele et son hash, les reglages, les courbes avant/apres avec dispersion, la part du softmax, les collisions, le cas chaud, les tests executes et les decisions. Distinguer les mesures interleaved des mesures entre sessions.
- [ ] **Validation finale.** Suite CTest et Python complete, smoke UCI en sous-processus (`uci`, `isready`, `position startpos`, `go nodes`, `stop`, `quit`), bestmove legal, aucune recherche restante.
- [ ] **Commit :** `Documente les optimisations du cout de calcul` si des defauts changent, sinon `Documente les mesures du cout de calcul`.

## Ordre et couverture de la spec

```text
1 banc hors arbre -> 2 buffers -> 3 forme fixe -> 4 divergence
-> 5 FP16 -> 6 cas chaud -> 7 mesures et decision
```

- Spec 6 et 6.1, cout par appel et ecart self-play : taches 1 a 3.
- Spec 8.1, assainir l'appel : taches 2 et 3.
- Spec 8.2, remplissage et divergence : tache 4.
- Spec 8.4, cas chaud : tache 6.
- Spec 8.5, reseau et quantification : tache 5.
- Spec 9, tests T1 a T6 : taches 1, 3, 4, 5, 6.
- Spec 11 et 12, plafond realiste : taches 1, 6 et 7.
- Spec 8.2, production continue : volontairement hors execution ; la tache 7 fournit la decision et le plan separe eventuel.

Le plan livre un evaluateur assaini, des reglages de divergence mesures et un verdict chiffre sur FP16 et la production continue, sans changer les defauts UCI sans preuve.
