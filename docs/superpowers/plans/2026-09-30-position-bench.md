# Banc externe de positions : Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. L'exécution directe est recommandée pour ce chantier ; le choix de l'utilisateur reste prioritaire.

**Goal:** Remplacer l'ancre Stockfish périodique par un banc externe figé qui mesure policy, value et recherche, avec une durée cible d'environ cinq minutes.

**Architecture:** Un constructeur hors ligne extrait des parties de broadcasts, annote chaque coup légal avec Stockfish puis publie un dataset immuable. Un évaluateur utilise une seule session ONNX GPU : lots de positions pour policy/value, puis positions MCTS séquentielles avec un objet MCTS persistant. Les résultats par position sont conservés pour comparer les checkpoints et alimenter W&B.

**Tech Stack:** Python 3.13, NumPy, python-chess, zstandard, requests, W&B, C++17, pybind11, ONNX Runtime 1.24.3, pytest et CTest. Dépendances déjà déclarées dans le projet.

**Spec:** [Conception validée, révisée le 30 septembre](C:/Users/mlecl/Documents/LapZero-Chess/docs/superpowers/specs/2026-09-29-position-bench-design.md).

Statut : plan proposé, aucune tâche d'implémentation réalisée.
Base relue : commit `18963ea`, plus la clarification utilisateur du 30 septembre sur la durée souple.

## Global Constraints

- « cible d'environ 300 secondes, chargement et agrégation compris, sans limite dure » ; l'écriture locale des résultats est également incluse.
- « toutes les 4 itérations par défaut, réglable par `--eval-every` ».
- 10 000 positions, une seule par partie ; 256 positions MCTS à 384 simulations.
- Policy/value : lots de 1 024, 119 plans, historique intégral rejoué depuis startpos.
- MCTS : `c_puct=1.4`, batch 8 fixe, 8 workers, virtual loss 2, FPU 0.30, facteur de tentatives de collision 4.
- Les positions MCTS se succèdent sur un seul objet MCTS, avec racine neuve, TT froide et compteurs réinitialisés. Session GPU et pool de threads conservés.
- Aucun Dirichlet, aucun échantillonnage des coups ; aucune adaptation automatique du budget ou du nombre de positions au temps écoulé.
- Cinq minutes ne déclenche ni arrêt ni invalidation. Une tendance vers huit ou neuf minutes justifie une analyse de coût.
- Sources : broadcasts externes, août 2026 puis juillet et juin, Elo des deux joueurs >= 2000, bots conservés.
- Exclure les 105 000 PGN du préentraînement, les données supervisées/fine-tuning, puzzles, self-play et parties de LapZero des sources du banc.
- Annotation : Stockfish non bridé, 1 thread, Hash 128 Mio, WDL activée, pas de Syzygy, état de recherche réinitialisé, budgets en noeuds.
- Criblage 50 000 noeuds ; réserve 12 500 positions ; 200 000 noeuds par coup légal ; audit de 5 % à 400 000.
- Score principal : regret attendu de la policy sur l'espérance WDL ; value scalaire dans `[-1,1]`, MCTS et centipions publiés séparément. Aucun Elo proxy ni composite.
- Tout changement de dataset produit une nouvelle version. Tout changement des paramètres de mesure produit un nouveau protocole, sans mélanger les comparaisons.
- Windows, Python >= 3.13 ; conserver les dépendances existantes et le module C++ de recherche.
- Pas de tiret cadratin ni de ligne de co-auteur dans les textes et commits.

## Review Focus

1. Une archive récente peut contenir une vieille partie, une partie tronquée ou un doublon d'un autre broadcast : contrôler la provenance réelle et la ligne complète avant extraction (tâche 3).
2. FEN identique, historique différent ; double poussée de pion sans prise en passant légale ; répétitions et sous-promotions : reconstruire exactement l'entrée réseau et tous les coups légaux (tâches 1 et 3).
3. Coup des noirs, mat et égalité WDL arrondie : conserver le bon point de vue et choisir la référence centipions sur les scores bruts (tâches 1 et 4).
4. Inférence de fin de lot, sortie non finie, CUDA indisponible ou données modifiées : échouer explicitement, sans publier une mesure incomplète ni changer de backend à l'insu de l'utilisateur (tâches 2 et 6).
5. Reprise d'entraînement, résultat précédent incompatible ou passage de plus de 300 secondes : comparer les bonnes paires, garder la cadence globale et publier tous les résultats complets (tâches 7 et 8).

## Points du code à respecter

| Point relu | Conséquence pour le plan |
|---|---|
| `src/mcts.cpp`, `mcts_search()` | Racine locale créée/détruite à chaque appel ; pas besoin de recréer le MCTS pour isoler les positions. |
| `src/mcts.cpp`, `clear_evaluation_cache()` et `src/mcts.hpp` | La remise à froid de la TT conserve le pool ; l'appeler au repos avant chaque recherche. |
| `src/mcts_wave.cpp`, `run_search_waves()` | Les huit workers collectent dans un arbre partagé ; le coordinateur infère puis effectue les backups. |
| `src/search_executor.cpp` | Pool persistant créé paresseusement, détruit avec le MCTS. Aucun pool Python autour des 256 recherches. |
| `src/onnx_evaluator.cpp` | Une session possède les compteurs non atomiques ; aucun appel concurrent sur le même évaluateur. `evaluate_batch()` fournit actuellement des probabilités, pas les logits. |
| `src/bindings.cpp` | Le constructeur ONNX est exposé, mais pas l'inférence directe. Ajouter un binding ciblé pour policy/value. |
| `pyproject.toml` et environnement local | Python ORT 1.24.3 expose Azure/CPU uniquement, vérifié le 30 septembre. Le runtime C++ téléchargé par CMake dispose du fournisseur CUDA. Utiliser ce dernier pour les deux phases. |
| `python_src/puzzle_bench.py` | Réutiliser les fonctions de résolution/export de modèle ; ne pas réutiliser sa fabrique qui crée un MCTS par position à budget fixe. |
| `python_src/bench_metrics.py` | Réutiliser le codage UCI/index ; la limitation historique à 128 coups mentionnée dans ce fichier ne doit pas devenir un filtre du nouveau banc. |
| `python_src/train_self_play.py` | Le checkpoint et son ONNX existent avant l'évaluation. L'assert sur Stockfish est actuellement inconditionnel. |
| `python_src/run_selftrain.py` et bloc principal de `train_self_play.py` | Les deux valeurs effectives de cadence sont actuellement 8, malgré le défaut 4 de `pipeline()`. Corriger les deux lanceurs. |
| `sync_to_sim.ps1` | Le transfert utilise une liste explicite de scripts et dossiers ; ajouter le banc et ses modules à cette liste lors de l'intégration. |

## Structure des fichiers

Les chemins ci-dessous sont relatifs à `C:/Users/mlecl/Documents/LapZero-Chess`.

| Fichier | Responsabilité |
|---|---|
| Créer `python_src/position_bench_metrics.py` | Contrats purs, validation des enregistrements/manifeste, métriques et bootstrap. Aucun import de moteur, torch ou W&B. |
| Créer `python_src/position_bench_sources.py` | Archives, PGN, provenance, rejeu C++, candidats et identité réseau. |
| Créer `python_src/position_bench_teacher.py` | Processus Stockfish, annotation, reprise et audit. |
| Créer `python_src/build_position_bench.py` | CLI, sélection par quotas, publication immuable. |
| Créer `python_src/position_bench.py` | CLI d'évaluation, lots policy/value, recherche séquentielle, mesures de durée. |
| Créer `python_src/position_bench_results.py` | NPZ/sidecars, comparaison, journal W&B et rapports. |
| Modifier `src/onnx_evaluator.hpp`, `src/onnx_evaluator.cpp`, `src/bindings.cpp` | Accès direct aux logits/value avec la session existante. |
| Régénérer `python_src/chess_engine.pyi` | Stub issu du build, pas de modification manuelle. |
| Modifier `python_src/train_self_play.py`, `python_src/run_selftrain.py` | Appel périodique, configuration, prévalidation, logs. |
| Modifier `.gitignore`, `README.md`, `sync_to_sim.ps1` | Fichiers de travail, mode d'emploi et transfert des dépendances. |
| Créer `python_src/tests/test_position_bench_*.py` | Tests purs, sources, Stockfish simulé, interface ONNX, runtime et intégration. |
| Créer `tests/data/position_bench/` | Petits PGN synthétiques et données de test. Aucun poids ni PGN externe massif. |
| Créer `data/position_bench/v1/` | Dataset final, manifeste et attribution, seulement après qualification. |

Ce découpage spécialise les trois points d'entrée de la spec pour éviter un constructeur monolithique. Il ne change pas leurs responsabilités publiques. Le MCTS, le self-play C++ et le banc de puzzles gardent leurs algorithmes.

## Contrats partagés

Définir les contrats JSON avec `TypedDict` dans `position_bench_metrics.py`, accompagnés de validations exécutées à la lecture. Les tuples ci-dessous sont sérialisés en tableaux JSON.

| Type | Champs obligatoires |
|---|---|
| `SourceInfo` | `source_id`, `url`, `month`, `archive_sha256`, `license`, `attribution` |
| `Position` | `position_id`, `game_id`, `game_fingerprint`, `source_id`, `event_id`, `white_id`, `black_id`, `white_elo`, `black_elo`, `white_type`, `black_type`, `game_date`, `start_fen`, `moves_uci`, `fen`, `ply`, `turn`, `halfmove_clock`, `repetition_count`, `phase`, `legal_indices` |
| `MoveLabel` | `uci`, `index`, `wdl` (W,D,L), `score` (espérance S), `cp` ou `mate` (l'autre vaut null), `depth`, `nodes` |
| `AnnotatedPosition` | Tous les champs de `Position`, plus `labels`, `s_best`, `wdl_bucket` |
| `Manifest` | `schema_version=1`, `dataset_version`, `dataset_sha256`, `training_forbidden=true`, `sources`, `counts`, `quotas`, `stockfish`, `audit`, `search_ids`, `protocol_defaults`, `builder_revision` |
| `SearchConfig` | dataclass gelée : `simulations=384`, `batch_size=8`, `workers=8`, `fixed_batch=True`, `virtual_loss=2`, `fpu=0.30`, `collision_attempts=4`, `c_puct=1.4`, `tt_size=8192`, `cache_history_depth=0` |
| `EvalConfig` | dataclass gelée : `search: SearchConfig`, `policy_batch_size=1024`, `target_s=300.0`, `use_gpu=True`, `bootstrap_samples=10000`, `bootstrap_seed=20260930` |
| `RawResult` | `position_ids`, `legal_offsets`, `legal_indices`, `policy_probs`, `values`, `search_ids`, `search_offsets`, `search_indices`, `search_probs`, `search_counters`, `timings` |
| `EvalReport` | `status`, `dataset_version`, `dataset_sha256`, `model_sha256`, `protocol_id`, `completed_positions`, `completed_search_positions`, `duration_s`, `timings`, `metrics`, `comparison`, `result_path`, `error` |

Règles communes : identifiants uniques ; coups légaux triés par index ; en cas d'égalité de policy ou de visites, plus petit index. Aucune valeur NaN/Inf en JSON ; utiliser null pour un diagnostic non défini. Joueurs : `human`, `bot` ou `unknown`, ce dernier évite de déduire une identité absente du PGN. Taille de TT 8192 et politique h0 reprises des bancs existants, enregistrées explicitement.

`protocol_id` hache une représentation JSON canonique des paramètres de mesure, de la version des métriques, du backend, du binaire `chess_engine` et du runtime ONNX. Le hash du modèle est enregistré séparément pour permettre la comparaison de checkpoints. La cible de durée et les chemins de sortie ne changent pas le protocole scientifique.

## Commandes de validation

Depuis la racine du dépôt, dans PowerShell :

```powershell
$benchPython = Join-Path (Get-Location) '.venv/Scripts/python.exe'
$benchCmake = 'C:/Program Files/Microsoft Visual Studio/18/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe'
$benchCtest = Join-Path (Split-Path $benchCmake) 'ctest.exe'
& $benchPython -m pytest -p no:cacheprovider python_src/tests/test_position_bench_metrics.py -q
& $benchCmake --build build --config Release
& $benchCtest --test-dir build -C Release --output-on-failure
```

Ces chemins sont ceux du checkout relu. Dans un autre checkout, reprendre l'interpréteur du build CMake et vérifier la disponibilité du module cp313. Utiliser les mécanismes d'approbation du shell si l'interpréteur ou `.git` est bloqué par le sandbox.

Pour chaque tâche logicielle : ajouter le test ciblé, constater son échec fonctionnel, implémenter, vérifier le passage puis committer uniquement les fichiers de la tâche. Une erreur d'import d'une dépendance existante n'est pas une validation rouge du comportement. Les exemples ci-dessous désignent les cas minimaux à écrire, complétés par les cas explicitement énumérés.

## Tâche 1 : contrats et métriques pures

**Files:** créer `python_src/position_bench_metrics.py` et `python_src/tests/test_position_bench_metrics.py`.

**Interfaces produites :** `wdl_score(wdl: tuple[int,int,int]) -> float`, `wdl_value(wdl) -> float`, `validate_position(record: dict) -> None`, `validate_manifest(manifest: dict) -> None`, `score_distribution(probabilities: np.ndarray, labels: list[MoveLabel]) -> dict`, `value_metrics(predicted: np.ndarray, targets: np.ndarray) -> dict`, `aggregate(rows: list[dict]) -> dict`. Définir également les types du tableau précédent.

- [ ] Écrire les cas analytiques, puis lancer le fichier de test :

```python
def test_echelle_wdl_et_value():
    assert wdl_score((300, 600, 100)) == pytest.approx(0.6)
    assert wdl_value((300, 600, 100)) == pytest.approx(0.2)

def test_regret_pondere():
    labels = [
        dict(uci="a2a3", index=1, wdl=(600, 400, 0), score=.8,
             cp=120, mate=None, depth=20, nodes=200000),
        dict(uci="b2b3", index=2, wdl=(300, 600, 100), score=.6,
             cp=40, mate=None, depth=20, nodes=200000),
        dict(uci="c2c3", index=3, wdl=(0, 400, 600), score=.2,
             cp=-120, mate=None, depth=20, nodes=200000),
    ]
    m = score_distribution(np.array([.5, .3, .2]), labels)
    assert m["expected_regret"] == pytest.approx(.18)
    assert m["argmax_regret"] == 0
    assert m["near_best_mass"] == pytest.approx(.5)
```

- [ ] Implémenter WDL et regret en float64, avec validation avant calcul : W,D,L entiers positifs ou nuls, somme strictement positive, scores cohérents ; distribution finie, positive ou nulle et de somme 1 à `1e-6` près. Ne pas réparer une distribution invalide silencieusement.

```python
scores = np.asarray([label["score"] for label in labels], dtype=np.float64)
regrets = scores.max() - scores
expected_regret = float(probabilities @ regrets)
chosen = int(np.argmax(probabilities))  # labels déjà triés par index
```

- [ ] Tester puis calculer les masses aux frontières inclusives `<=0.02` et `>=0.20`, avec une tolérance numérique fixe `1e-12`. Tester MAE, RMSE, biais et Pearson ; Pearson null si moins de deux valeurs ou variance nulle.

- [ ] Tester le diagnostic cp lorsque plusieurs coups ont S=1 mais des cp différents : choisir le meilleur selon le score cp/mat, pas selon l'argmax d'une WDL saturée. L'ordre est mat perdant, cp, mat gagnant ; les distances de mat servent uniquement à départager des mats. Si meilleur ou choisi est un mat, cp regret null. Médiane et p90 utilisent `np.quantile(..., method="linear")`, seuils 20/50/100 inclusifs, couverture comptée sur toutes les positions. Cas zéro position admissible : statistiques null, couverture 0.

- [ ] Tester la validation de schéma, les annotations manquantes/dupliquées, les deux couleurs, un index >=4672 et une position ayant plus de 128 coups légaux. Agréger globalement, par phase, catégorie WDL et type du joueur au trait, avec effectifs de chaque tranche. Ne pas moyenner les corrélations calculées sur des sous-groupes.

- [ ] Relancer le fichier ciblé, puis commit `Ajoute les contrats et metriques du banc de positions`.

## Tâche 2 : exposer l'inférence brute de la session C++

**Files:** modifier `src/onnx_evaluator.hpp`, `src/onnx_evaluator.cpp`, `src/bindings.cpp` ; créer `python_src/tests/test_position_bench_onnx.py` ; régénérer `python_src/chess_engine.pyi`.

**Interface produite :** `ONNXEvaluator.predict_batch(states: ndarray[float32, N,119,8,8]) -> tuple[ndarray[float32,N,4672], ndarray[float32,N]]`, logits bruts et value. Appel synchrone, sans concurrence avec MCTS.

- [ ] Générer dans le test un minuscule modèle ONNX déterministe avec `onnx.helper`, sans checkpoint : entrée aplatie puis Slice des 4672 premiers éléments pour `policy`, ReduceMean des plans pour `value`, batch dynamique. Forcer un opset et une IR pris en charge par le runtime du dépôt (opset 18, IR 10). Exemple de contrat :

```python
def test_logits_bruts_et_derniere_ligne(tiny_onnx):
    evaluator = chess_engine.ONNXEvaluator(str(tiny_onnx), False)
    states = np.zeros((3, 119, 8, 8), dtype=np.float32)
    states[2].fill(.25)
    logits, values = evaluator.predict_batch(states)
    assert logits.shape == (3, 4672)
    assert values.shape == (3,)
    np.testing.assert_array_equal(logits[0], np.zeros(4672))
    np.testing.assert_array_equal(logits[2], np.full(4672, .25))
    assert values[2] == pytest.approx(.25)
```

La fixture `tiny_onnx(tmp_path)` construit, vérifie avec `onnx.checker` et sauvegarde ce graphe. Tous ses tenseurs internes sont nommés et les dimensions N sont symboliques.

- [ ] Ajouter une méthode C++ brute et factoriser le `session->Run` existant dans une fonction privée recevant un booléen de post-traitement. Conserver le softmax global, ses compteurs et timings dans le chemin MCTS existant. Éviter d'ajouter une copie des logits au chemin MCTS.

```cpp
void predict_batch(const std::vector<float>& input,
                   std::vector<float>& logits,
                   std::vector<float>& values, int batch_size);
// Fonction privée commune ; probabilities=true pour evaluate_batch().
void evaluate_impl(const std::vector<float>& input,
                   std::vector<float>& policy,
                   std::vector<float>& values, int batch_size,
                   bool probabilities);
```

- [ ] Dans le binding, vérifier N>0, le type float32, les dimensions et la contiguïté avant de copier vers le buffer C++. Libérer le GIL seulement autour du calcul, puis construire des tableaux NumPy propriétaires après sa reprise. Vérifier tailles et valeurs finies des sorties avant de les lire. Refuser explicitement entrées mal formées et sorties d'un modèle incompatible.

- [ ] Tester les lots 1, 8 et 3, une entrée non contiguë, le mauvais dtype, NaN et une mauvaise forme de sortie. Comparer les priors légaux de MCTS avec le softmax masqué des logits sur le petit modèle. Tester qu'une erreur laisse la session utilisable à l'appel suivant. Réutiliser les tests de contrat/timing existants pour éviter une régression du self-play.

- [ ] Construire en Release ; exécuter `test_position_bench_onnx.py`, `test_bench_engine.py`, puis CTest `-R 'evaluator_contract|softmax_rows|wave_search|search_timing'`. Les tests avec le petit ONNX doivent passer sans GPU ni checkpoint. Commit `Expose les logits batches du runtime ONNX C++`.

## Tâche 3 : archives externes et positions avec historique

**Files:** créer `python_src/position_bench_sources.py`, `python_src/tests/test_position_bench_sources.py`, `tests/data/position_bench/broadcasts.pgn` ; modifier `.gitignore` pour `data/position_bench_work/` et `/position_bench_results/`.

**Interfaces consommées :** `Position`, `SourceInfo`, `validate_position` (pour les enregistrements annotés en aval), `bench_metrics.uci_to_index`, `bench_metrics.index_to_uci`.

**Interfaces produites :** `read_games(path: Path, source: SourceInfo) -> Iterator[chess.pgn.Game]`, `extract_candidates(games: Iterable[chess.pgn.Game], source: SourceInfo) -> list[Position]`, `replay_position(record: Position) -> chess_engine.Chessboard`, `position_identity(board) -> str`, `download_archive(url: str, destination: Path, expected_sha256: str | None) -> SourceInfo`. Les compteurs de rejet sont disponibles dans le bilan de l'extraction.

- [ ] Préparer des PGN synthétiques de >=16 demi-coups avec commentaires et variantes, deux joueurs >=2000, un tag BOT ; dériver les rejets Elo 1999/absent, variante non standard, `SetUp`, `FEN`, coup illégal, résultat `*`, date ancienne et partie trop courte. Absence du tag Variant signifie Standard selon PGN ; ne pas rejeter tous les broadcasts pour ce seul tag absent.

```python
def test_rejeu_garde_la_case_en_passant():
    ref = chess.Board()
    ref.push_uci("e2e4")
    record = {"start_fen": chess.STARTING_FEN, "moves_uci": ["e2e4"],
              "fen": ref.fen(en_passant="fen")}
    board = replay_position(record)
    assert board.to_fen() == record["fen"]
```

Ce test de rejeu n'exige que les trois champs qu'il lit ; les tests d'extraction valident le contrat complet. Tester aussi deux séquences de développement aboutissant à la même FEN mais avec des tenseurs historiques différents.

- [ ] Lire `.pgn.zst` en streaming, sans téléchargement massif lors des tests. Écriture de téléchargement dans un `.part`, hash en streaming, publication seulement après contrôle. Sur interruption, ne jamais présenter le `.part` comme une archive valide. Les fichiers téléchargés restent dans `data/position_bench_work/archives/`.

- [ ] Lire uniquement la table broadcasts de la [base officielle Lichess](https://database.lichess.org/#broadcasts) et enregistrer URL, mois, licence CC BY-SA 4.0, attribution et hash. Lire le mois entier avant de compter 20 000 parties valides ; un arrêt au 20 000e PGN ferait dépendre le résultat de l'ordre de l'archive. Commencer en août puis remonter selon la spec. Vérifier la date de la partie : après février 2026 ; date absente ou incohérente signalée et exclue. Ne jamais descendre dans les archives de préentraînement pour remplir un quota.

- [ ] Parser toute la ligne principale avant extraction ; rejeter `game.errors`, résultat incomplet, partie illégale. Pour les identités joueur/événement, préférer identifiant FIDE/broadcast stable puis nom normalisé et contexte événement. Stocker la règle de normalisation. Ne pas convertir un joueur sans métadonnée en humain certain. Une identité joueur/événement inexploitable doit être comptée comme rejet, pas diluée dans des identifiants aléatoires.

- [ ] Calculer `game_fingerprint` depuis startpos et la séquence UCI complète, indépendamment des commentaires, en-têtes de diffusion ou ordre des archives. Conserver un `game_id` de source stable ou SHA-256 du PGN canonique, puis dédupliquer aussi par fingerprint. Vérifier les exclusions de corpus par leur provenance et fingerprint quand les fichiers supervisés locaux sont disponibles ; enregistrer ce qui a été contrôlé dans le manifeste de travail.

- [ ] Pour chaque partie valide, parcourir les positions non terminales ; déterminer les phases selon la spec et choisir au plus un candidat par phase par hash `game_id | ply | lapzero-position-bench-v1`. Rejouer le candidat dans le C++, comparer la FEN aux six champs avec `python-chess.fen(en_passant="fen")` et vérifier le bijection UCI/index, y compris sous-promotions, roques et prise en passant.

```python
tensor = np.asarray(board.get_alphazero_tensor(), dtype="<f4", order="C")
indices = np.asarray(sorted(board.get_legal_move_indices()), dtype="<u4")
key = board.get_evaluation_cache_key(0)
repetitions = int(key.repetition_category) + 1  # catégorie C++ 0, 1, 2
suffix = json.dumps([int(board.turn), board.half_move_clock, repetitions],
                    separators=(",", ":")).encode("ascii")
identity = hashlib.sha256(tensor.tobytes() + indices.tobytes() + suffix).hexdigest()
```

Stocker un domaine et les longueurs dans la sérialisation canonique définitive pour rendre ses frontières explicites. La clé h0 seule ne remplace pas le tenseur historique.

- [ ] Tester que mélanger les parties, les archives et les commentaires ne modifie pas l'ensemble final des candidats ni leurs identités réseau ; collisions entre candidats résolues par un ordre canonique `(game_id, ply, source_id)`. Lancer `test_position_bench_sources.py`, `test_move_coding.py`, `test_evaluation_cache_key.py`. Commit `Extrait les positions externes avec leur historique reel`.

## Tâche 4 : annotations Stockfish reprenables

**Files:** créer `python_src/position_bench_teacher.py`, `python_src/tests/test_position_bench_teacher.py`.

**Interfaces consommées :** `Position`, `MoveLabel`, `wdl_score`, sources et séquences UCI de tâche 3.

**Interfaces produites :** `screen_position(engine, position: Position) -> float`, `annotate_move(engine, position: Position, uci: str, nodes: int) -> MoveLabel`, `annotate_position(engine, position: Position, nodes: int) -> AnnotatedPosition`, `audit_labels(base: list[AnnotatedPosition], deeper: list[AnnotatedPosition]) -> dict` ; `AnnotationStore(work_dir: Path, config: dict)` avec `get(key)`, `put(key, result)` et vérification d'identité de configuration.

- [ ] Écrire un faux moteur qui enregistre options, pile de coups, limite et root_moves puis renvoie un `PovScore` et un `PovWdl`. Vérifier notamment le trait noir :

```python
def test_score_du_point_de_vue_des_noirs():
    info = {"score": chess.engine.PovScore(chess.engine.Cp(80), chess.WHITE),
            "wdl": chess.engine.PovWdl(chess.engine.Wdl(600, 300, 100), chess.WHITE)}
    assert info["score"].pov(chess.BLACK).score() == -80
    wdl = info["wdl"].pov(chess.BLACK)
    assert (wdl.wins, wdl.draws, wdl.losses) == (100, 300, 600)
    assert wdl_score((wdl.wins, wdl.draws, wdl.losses)) == .25
```

Compléter par un test qui appelle réellement `annotate_move()` sur le faux moteur, pour couvrir l'assemblage de l'étiquette et son index.

- [ ] Construire un plateau python-chess avec sa pile intégrale, puis analyser à la racine :

```python
engine.configure({"Clear Hash": None})
info = engine.analyse(board, chess.engine.Limit(nodes=nodes),
                      root_moves=[chess.Move.from_uci(uci)], game=object())
score = info["score"].pov(board.turn)
wdl = info["wdl"].pov(board.turn)
```

`game=object()` réinitialise également l'état entre recherches. Cette utilisation de la pile et de `root_moves` est documentée par [python-chess](https://python-chess.readthedocs.io/en/latest/engine.html#chess.engine.Protocol.analyse). Ne pas analyser une FEN enfant avec le mauvais point de vue ni recalculer la WDL par `Score.wdl()` avec un modèle différent.

- [ ] Configurer chaque processus une fois : `Threads=1`, `Hash=128`, `UCI_LimitStrength=False`, `UCI_ShowWDL=True`, `SyzygyPath=""`, MultiPV simple et aucun ponder. Vérifier la présence des options requises. Enregistrer nom UCI, hash du binaire et toutes les NNUE utilisées. Pour un réseau embarqué, enregistrer son nom et extraire ses octets via la commande d'export supportée par ce binaire pour calculer le hash ; si l'identification n'est pas possible, arrêter le build avec une erreur précise plutôt qu'inventer un hash de réseau. Les NNUE externes sont hashées directement.

- [ ] Valider chaque résultat : PV commence par le coup imposé, annotations complètes, WDL présente et valide, score exact sans borne, profondeur et noeuds présents. Retenter une seule fois sur un nouveau processus en cas d'annotation invalide ou crash ; le second échec arrête la construction. Toujours fermer les processus créés, y compris si un autre worker échoue. Tests mats gagnants/perdants, absence WDL et PV incorrecte.

- [ ] Utiliser un cache de travail SQLite standard, avec une seule connexion d'écriture détenue par le coordinateur ; les workers renvoient leurs résultats. Clé : identité position, coup, budget, hash Stockfish/NNUE/options, version du schéma. Une transaction par résultat complet ; ne pas stocker les étiquettes partielles comme terminées. Des changements de configuration de reprise sont refusés ; une extension explicite à un autre mois enrichit le manifeste de sources sans réinterpréter le contenu déjà hashé.

- [ ] Tester reprise après interruption, refus d'un autre moteur/budget pour la même étape et résultats indépendants du nombre de workers. Ajouter un smoke test réel optionnel via variable `POSITION_BENCH_STOCKFISH`, distinct de la suite rapide. Lancer `test_position_bench_teacher.py`. Commit `Ajoute l annotation Stockfish reprenable`.

## Tâche 5 : quotas, audit et publication du banc

**Files:** créer `python_src/build_position_bench.py`, `python_src/tests/test_position_bench_build.py`. Le chargeur de lecture du dataset sera créé en tâche 7, distinct de la publication du constructeur.

**Interfaces produites :** `select_positions(records: list[AnnotatedPosition], quotas: dict, *, salt: str, event_cap: int, player_cap: int) -> list[AnnotatedPosition]`, `select_search_ids(records: list[AnnotatedPosition]) -> list[str]`, `write_dataset(records, manifest: Manifest, output: Path) -> Path`, `main(argv: list[str] | None = None) -> int`.

- [ ] Créer un fixture synthétique équilibré, avec plusieurs candidats par partie, des événements/joueurs répétés, des couleurs alternées et des scores près des frontières. Vérifier les neuf quotas, une position par game fingerprint, plafonds événement/joueur et couleurs 45 % minimum. Les petits tests utilisent des quotas réduits injectés ; un manifeste de fixture est explicitement non productif et refusé par l'intégration d'entraînement.

```python
def test_selection_ne_depend_pas_de_l_ordre(annotated_pool, small_quotas):
    options = dict(salt="lapzero-position-bench-v1-final",
                   event_cap=100, player_cap=20)
    a = select_positions(annotated_pool, small_quotas, **options)
    b = select_positions(list(reversed(annotated_pool)), small_quotas, **options)
    assert [p["position_id"] for p in a] == [p["position_id"] for p in b]
    assert len({p["game_fingerprint"] for p in a}) == len(a)
```

- [ ] Implémenter un parcours déterministe des neuf cases, en privilégiant le plus grand déficit relatif, égalités départagées ouverture/milieu/finale puis disputée/avantage/décisive. Dans chaque case, l'ordre des candidats est le hash demandé par la spec. Refuser un candidat qui dépasse un plafond ou fait dépasser 55 % du total pour une couleur. Vérifier toutes les contraintes à la fin. Une sélection épuisée rapporte les cases manquantes et relance avec le mois précédent ; ne pas présenter cet échec glouton comme une preuve mathématique d'impossibilité.

- [ ] Sélectionner la réserve de 12 500 après criblage avec quotas de la spec multipliés par 1.25. Annoter cette réserve puis reclasser avec les scores finaux. Si les neuf quotas définitifs deviennent insuffisants, ajouter les candidats du mois précédent autorisé et refaire la sélection de réserve, en réutilisant les annotations déjà obtenues. Tester une position qui change de catégorie après analyse profonde.

- [ ] Pour le sous-banc, arrondir par plus grands restes les neuf quotas à 256, égalités départagées dans l'ordre des cases : ouverture `(26,15,10)`, milieu `(77,46,31)`, finale `(26,15,10)`. Hash indépendant `position_id | lapzero-position-bench-v1-search`, puis ordre explicite des identifiants dans le manifeste.

- [ ] Auditer les 500 premières positions triées par un sel `lapzero-position-bench-v1-audit`. Vérifier le regret du meilleur coup de base, départagé par index, sous les labels à 400 000, et la moyenne de `abs(S*_base-S*_profond)` ; pas de compensation entre variations positives et négatives. Conditions : >=95 % avec regret <=0.02, variation moyenne <=0.01. Si échec, reconstruire toutes les étiquettes de la réserve à 400 000, refaire la sélection, puis auditer à 800 000 pour réellement tester la stabilité de ce nouveau budget. Si ce second audit échoue, ne pas publier v1 ; conserver le diagnostic. Ce contrôle explicite le cas non détaillé dans la spec, sans augmenter silencieusement le budget à l'infini.

- [ ] Écrire le JSONL canonique dans un fichier temporaire, trié, compressé avec paramètres zstd fixes ; calculer le hash des octets compressés ; publier dataset, README et manifeste en dernier. `write_dataset()` refuse d'écraser une version existante différente. Une date de fabrication variable reste dans le manifeste et ne doit pas modifier les labels/ordres/hashes du JSONL.

- [ ] Exposer les sous-commandes `extract`, `screen`, `annotate`, `finalize`, `build`. Options : `--work-dir`, `--output`, `--months`, `--stockfish`, `--workers`, `--resume`. Ajouter `annotate --pilot-count 20` pour l'essai sur les premiers candidats triés par hash, enregistré dans un espace de travail marqué `pilot` ; ce mode ne publie jamais de v1. `extract` écrit `candidates.jsonl.zst` et son manifeste de sources dans le work-dir pour le mode profile. Les filtres/quota/budgets de v1 sont centralisés et hashés. `build` orchestre les étapes, `finalize` effectue sélection et audit ; les sources locales doivent disposer d'un manifeste d'archives externes, aucun parcours opportuniste des PGN du dépôt.

- [ ] Tester refus d'une archive corrompue, version immutable, réserve épuisée, audit qui échoue puis labels reconstruits, ordre identique malgré l'achèvement variable des workers. Lancer `test_position_bench_build.py` et `test_position_bench_teacher.py`. Commit `Construit et verifie le banc externe versionne`.

## Tâche 6 : évaluation GPU complète et recherches séquentielles

**Files:** créer `python_src/position_bench.py`, `python_src/tests/test_position_bench_runtime.py` ; réutiliser `replay_position()` et le binding de tâche 2.

**Interfaces consommées :** `AnnotatedPosition`, `Manifest`, `SearchConfig`, `EvalConfig`, `RawResult`, `score_distribution()`, `ONNXEvaluator.predict_batch()`, `replay_position()`.

**Interfaces produites :** `evaluate_positions(records: list[AnnotatedPosition], search_ids: list[str], evaluator, config: EvalConfig, *, mcts_factory, clock=time.perf_counter) -> RawResult`, `profile_positions(records: list[Position], evaluator, config: EvalConfig, *, mcts_factory, clock=time.perf_counter) -> dict`. L'agrégation et les fichiers de résultats complets sont ajoutés en tâche 7.

- [ ] Créer un faux évaluateur dont `predict_batch()` dépend de la ligne d'entrée, et un faux MCTS qui vérifie les réglages, remet ses compteurs à zéro et renvoie une distribution légale avec exactement 384 simulations. Tester 10 positions avec batch 4 : tailles d'appels `[4,4,2]`, aucune perte/duplication au dernier lot et un seul objet évaluateur.

```python
class RecordingMCTS:
    def __init__(self, evaluator, tt_size, cache_history_depth):
        self.evaluator = evaluator
        self.calls = []
        self.completed = 0
        self.cache_cold = False

    def set_fixed_batch(self, enabled):
        assert enabled is True

    def set_tuning(self, virtual_loss, fpu_reduction, collision_attempt_factor):
        assert (virtual_loss, fpu_reduction, collision_attempt_factor) == (2, .30, 4)

    def clear_evaluation_cache(self):
        self.cache_cold = True

    def reset_counters(self):
        self.completed = 0

    def mcts_search(self, board, simulations, c_puct, add_dirichlet, batch, workers):
        assert self.cache_cold and self.completed == 0
        assert (simulations, c_puct, add_dirichlet, batch, workers) == (384, 1.4, False, 8, 8)
        self.calls.append(board.to_fen())
        self.cache_cold = False
        self.completed = simulations
        pi = np.zeros(4672)
        pi[min(board.get_legal_move_indices())] = 1
        return pi

    def get_counters(self):
        return SimpleNamespace(completed_simulations=self.completed,
                               nn_calls=384, nn_batches=48, tt_hits=0,
                               tt_misses=384, leaf_collisions=0)
```

Créer la liste des instances dans la fabrique de test ; vérifier qu'il n'y en a qu'une et que son évaluateur est exactement celui des inférences directes. Les fausses 48 inférences servent uniquement au contrat de compteurs, pas à prédire le remplissage réel.

- [ ] Préparer chaque lot via le C++ et comparer FEN, identité réseau et ensemble exact des annotations avant l'inférence. Pas de cache persistant des tenseurs en v1 : les 10 000 historiques doivent être validés à chaque passage. Garder en mémoire un seul lot de tenseurs et les distributions légales compactes ; reconstruire les 256 plateaux à la phase MCTS pour éviter de conserver 10 000 arbres/historiques C++.

```python
logits, values = evaluator.predict_batch(states)
for row, legal in zip(logits, legal_lists, strict=True):
    selected = np.asarray(row[legal], dtype=np.float64)
    selected -= selected.max()
    probs = np.exp(selected)
    probs /= probs.sum()
```

Vérifier toutes les sorties finies et la value dans `[-1,1]` avec tolérance `1e-6`, sans renormaliser ni écrêter une sortie value invalide. Le masque porte sur tous les coups légaux.

- [ ] Une fois policy/value terminées, conserver le même évaluateur et créer un seul MCTS, puis parcourir les identifiants dans l'ordre du manifeste :

```python
s = config.search
mcts = mcts_factory(evaluator, s.tt_size, s.cache_history_depth)
mcts.set_fixed_batch(s.fixed_batch)
mcts.set_tuning(s.virtual_loss, s.fpu, s.collision_attempts)
for position_id in search_ids:
    board = replay_position(by_id[position_id])
    mcts.clear_evaluation_cache()
    mcts.reset_counters()
    pi = mcts.mcts_search(board, s.simulations, s.c_puct, False,
                          s.batch_size, s.workers)
    counters = mcts.get_counters()
    if counters.completed_simulations != s.simulations:
        raise ValueError("budget MCTS incomplet")
```

Ne pas découper les 384 simulations en plusieurs appels à `mcts_search()` : chacun recréerait sa racine. `step_analysis()` peut cumuler des tranches sur son arbre, mais n'est pas nécessaire ici. Pas de thread Python par position, de file GPU partagée entre plusieurs recherches ni de modification du noyau MCTS.

- [ ] Vérifier la légalité, la taille et la normalisation des distributions, ainsi que le FEN et le tenseur avant/après MCTS sur les tests réels. Tester séquence A,B,A : TT remise à froid ; égalité des deux A requise en mono-worker de test, invariants et budget exact requis à huit workers. Ne pas réclamer une identité bit à bit entre deux recherches multicœurs.

- [ ] Mesurer `load_s`, `replay_s`, `policy_s`, `mcts_setup_s`, `mcts_s`, `aggregate_s`, `write_s`, plus durée totale et durées par position MCTS. Chronométrer la création initiale de session/pool, ne pas la cacher dans un warm-up hors budget. L'expansion racine reste un appel de taille 1 ; le padding à 8 concerne les vagues, pas tous les appels ONNX.

- [ ] Tester avec une fausse horloge avançant au-delà de 300 secondes : tous les identifiants sont encore traités et le résultat complet est conservé. Une exception au milieu produit un statut d'échec sans scores agrégés ; `KeyboardInterrupt` se propage après nettoyage au repos, puis la boucle d'entraînement s'arrête. Aucune API d'annulation ou processus de surveillance n'est ajouté.

- [ ] Ajouter `profile_positions()` pour mesurer le parcours direct et les recherches sur des candidats non annotés. Ce mode est étiqueté `profile`, ne calcule pas de score et ne crée pas une version du banc. Il permet de chronométrer 32 positions puis un échantillon plus large avant la longue annotation ; conserver des historiques réels et les différentes phases.

- [ ] Exécuter `test_position_bench_runtime.py`, `test_multicore_api.py` et CTest `-R 'wave_search|multicore_stress|search_executor'`. Commit `Evalue les positions avec une session GPU et un MCTS persistants`.

## Tâche 7 : fichiers de résultats, comparaison et CLI

**Files:** créer/compléter `python_src/position_bench_results.py`, `python_src/tests/test_position_bench_results.py` ; compléter `position_bench.py` et `position_bench_metrics.py`.

**Interfaces produites :**

- `load_dataset(path: Path, *, production: bool = True) -> tuple[Manifest, list[AnnotatedPosition]]` dans results, sans charger de modèle.
- `save_result(raw: RawResult, report: EvalReport, output_dir: Path, checkpoint_stem: str) -> Path` et `load_result(path: Path) -> tuple[RawResult, EvalReport]`.
- `compare_results(current: RawResult, previous: RawResult, current_meta: dict, previous_meta: dict) -> dict` et `paired_bootstrap(delta: np.ndarray, *, samples=10000, seed=20260930) -> tuple[float,float]`.
- `wandb_metrics(report: EvalReport) -> dict` et `find_previous(output_dir: Path, dataset_sha256: str, protocol_id: str, iteration: int) -> Path | None`.
- `evaluate_checkpoint(onnx_path: Path, bench_path: Path, output_dir: Path, config: EvalConfig, *, iteration: int | None = None, global_step: int | None = None, previous: Path | None = None) -> EvalReport` dans `position_bench.py`.

- [ ] Tester le chargement avant création de l'évaluateur : hash incorrect, version inconnue, dataset tronqué, 9 999 positions, index légal manquant, doublons d'identifiants ou liste de recherche différente de 256. Le mode fixture n'est utilisé que par les tests et interdit dans `pipeline()`.

Les imports de `torch`, `chess_engine` et du résolveur de modèle sont différés jusqu'après les contrôles de fichiers et de schéma. Une commande de comparaison pure reste utilisable sans poids ni session ONNX. `load_dataset()` valide le contenu sérialisé ; le rejeu sémantique et l'identité réseau sont contrôlés avant chaque inférence par `evaluate_positions()`.

- [ ] Stocker les données légales sous forme de tableaux aplatis avec offsets, valeurs/scalar et compteurs. `np.load(..., allow_pickle=False)` partout. Le sidecar contient les hashes complets du modèle, dataset, module C++, runtime, protocole, paramètres effectifs, matériel, version des métriques et identifiants iteration/global_step. Le NPZ et le sidecar portent chacun une vérification de cohérence ; aucun simple nom de checkpoint ne sert d'identité.

```python
np.savez_compressed(handle,
    position_ids=np.asarray(raw["position_ids"], dtype="U64"),
    legal_offsets=np.asarray(raw["legal_offsets"], dtype=np.int64),
    legal_indices=np.asarray(raw["legal_indices"], dtype=np.int32),
    policy_probs=np.asarray(raw["policy_probs"], dtype=np.float32),
    values=np.asarray(raw["values"], dtype=np.float32),
    search_ids=np.asarray(raw["search_ids"], dtype="U64"),
    search_offsets=np.asarray(raw["search_offsets"], dtype=np.int64),
    search_indices=np.asarray(raw["search_indices"], dtype=np.int32),
    search_probs=np.asarray(raw["search_probs"], dtype=np.float32))
```

Compléter ce contenu avec les compteurs et durées du contrat `RawResult`. Enregistrer le hash du NPZ dans le sidecar. Écrire dans des fichiers temporaires, fermer puis publier, sidecar valide en dernier. Ne pas écraser un résultat différent portant le même nom ; les répétitions de qualification utilisent des dossiers distincts.

- [ ] Calculer les agrégats depuis la représentation brute qui sera sauvegardée, afin qu'une réagrégation retrouve les mêmes nombres à `1e-6` près. Pour la comparaison, apparier par ID, vérifier ensembles complets, hashes et protocole ; permuter l'ordre de lignes doit être inoffensif. Refuser les doublons et toute comparaison de versions différentes. Pour un premier passage, `comparison=null`.

```python
def test_bootstrap_d_un_delta_constant():
    delta = np.full(100, -.01)
    low, high = paired_bootstrap(delta)
    assert low == pytest.approx(-.01)
    assert high == pytest.approx(-.01)
    assert high < 0
```

Implémenter les 10 000 tirages par blocs de 128 pour éviter une matrice de 100 millions d'indices permanente. Pour chaque tirage, échantillonner les mêmes indices de la paire via le vecteur des différences. Quantiles 2.5/97.5 %, graine fixe ; amélioration uniquement si borne haute <0. Rapporter amélioré/inchangé/dégradé avec tolérance `1e-12` sur le regret. L'intervalle porte sur les positions du banc ; il ne prouve pas un gain Elo ni l'indépendance statistique parfaite de parties issues d'un même événement.

- [ ] Exposer `evaluate`, `compare`, `profile`, `validate` et `reaggregate` dans la CLI. `profile` reçoit `--candidates`, `--model`, `--search-count` (32 au pilote) et `--output-dir` ; la sélection est déterministe par phase et hash, enregistrée dans son rapport. Le mode normal consomme l'ONNX déjà exporté par l'entraînement. Pour un `.pt` manuel, appeler `puzzle_bench.resoudre_modele()` dans un dossier de cache nommé par le SHA-256 du checkpoint pour éviter son cache par stem ; inclure le temps d'export dans le rapport de ce mode manuel. Tous les chemins par défaut se résolvent depuis `__file__`, indépendamment du cwd.

```powershell
& $benchPython python_src/position_bench.py validate --bench data/position_bench/v1
& $benchPython python_src/position_bench.py evaluate --model python_src/checkpoints_onnx/2026_04_30_09h53_iter436_unsupervised.onnx --bench data/position_bench/v1 --output-dir position_bench_results/qualification-1
```

Le chemin du modèle est un exemple existant dans la configuration de transfert du dépôt, à vérifier avant utilisation. L'absence du modèle doit produire une erreur, pas un export/repli arbitraire. Pour `compare` et `reaggregate`, les fichiers restent locaux et aucun moteur n'est initialisé.

- [ ] `wandb_metrics()` fournit tous les noms de la section 8.3 de la spec. Ajouter `delta_policy_regret_low95`, `delta_policy_regret_high95`, `completed_search_positions`, `target_s`, `over_target`, les timings de phases et `protocol_id`. Les tranches utilisent `eval/position/phase/<phase>/...`, `wdl/<bucket>/...`, `player/<type>/...`, avec leur `count`. Garder les clés simples de v1 et ajouter une copie sous `eval/position/v1/<protocol_id>/...` pour des séries comparables ; une autre version utilise son propre préfixe. Pas d'Elo, pas de synthèse policy/value.

- [ ] Tester que `status="ok"` à 340 s conserve toutes les métriques et pose `over_target=True`, tandis que `status="error"` ou `"interrupted"` ne fournit que contexte, statut, durée et compteurs de progression. Un diagnostic null est absent du payload numérique W&B et reste null dans le JSON. La durée totale concerne le calcul et l'écriture locale, pas le temps de synchronisation distant de W&B.

- [ ] Exécuter `test_position_bench_results.py` et `test_position_bench_metrics.py`, puis un aller-retour complet sur fixture avec faux moteur. Commit `Sauvegarde et compare les evaluations de positions`.

## Tâche 8 : intégration au self-play et cadence

**Files:** modifier `python_src/train_self_play.py`, `python_src/run_selftrain.py` ; créer `python_src/tests/test_position_bench_training.py`.

**Interfaces consommées :** `evaluate_checkpoint()`, `load_dataset()`, `wandb_metrics()`, `find_previous()`.

**Interfaces produites :** `pipeline(..., eval_every=4, position_bench_path=None, eval_search_workers=8, eval_target_s=300.0, eval_output_dir=None)` ; `analyser_arguments(argv=None)` pour rendre le CLI testable. Les autres paramètres self-play/training restent présents. `eval_every=0` désactive explicitement le banc ; les valeurs négatives sont rejetées. `None` pour les chemins devient respectivement `<racine>/data/position_bench/v1` et `<racine>/position_bench_results`, jamais un chemin dérivé du cwd.

- [ ] Écrire un test qui exécute une boucle de six itérations à partir d'une reprise d'iteration 3, avec génération, entraînement, export, sauvegarde et W&B remplacés par des doubles. Vérifier des appels d'évaluation aux iterations 4 et 8, après sauvegarde/export, et transmission de l'ONNX correspondant. Tester cadence 0 et cadence invalide.

```python
def test_cadence_globale_apres_reprise():
    iterations = range(3, 9)
    evaluated = [i + 1 for i in iterations if (i + 1) % 4 == 0]
    assert evaluated == [4, 8]
```

Ce cas fixe l'attendu ; le test de `pipeline()` doit exercer les appels réels via les doubles, et pas seulement recopier cette expression.

- [ ] Ajouter `--position-bench`, `--eval-search-workers`, `--eval-target-seconds`, `--eval-output-dir`, et changer le défaut `--eval-every` à 4. Le bloc principal de `train_self_play.py` adopte la même cadence et transmet les options. Si on ajoute un parseur à ce deuxième point d'entrée, ne pas importer ou exécuter celui de `run_selftrain.py` pour éviter la circularité.

- [ ] Remplacer le bloc d'ancrage par l'appel au banc, après checkpoint :

```python
if eval_every > 0 and (iteration + 1) % eval_every == 0:
    report = evaluate_checkpoint(
        Path(current_onnx_path), position_bench_path, eval_output_dir,
        EvalConfig(search=SearchConfig(workers=eval_search_workers),
                   target_s=eval_target_s),
        iteration=iteration + 1, global_step=global_step,
        previous=previous_result)
    journal.update(wandb_metrics(report))
```

Résoudre `previous_result` depuis le dernier résultat complet compatible d'une iteration strictement antérieure, y compris après redémarrage. Un `previous` explicitement demandé mais incompatible doit être signalé ; la recherche automatique peut ignorer les incompatibles et signaler qu'aucune référence n'a été trouvée.

- [ ] Retirer l'import inconditionnel d'`evaluate_against_anchor`, le calcul Elo et l'assert d'existence Stockfish du pipeline par défaut. Garder `stockfish_player.py` et sa fonction d'ancrage utilisables manuellement. Pour les anciens paramètres Python `eval_stockfish_every` et paramètres Stockfish, prévoir une transition explicite : alias de cadence avec avertissement ; paramètres Stockfish explicitement fournis refusés avec message renvoyant à l'outil manuel, jamais réinterprétés silencieusement. Les anciens drapeaux CLI Stockfish affichent aussi une erreur de migration compréhensible.

- [ ] Prévalider dataset/manifest avant la première génération si le banc est activé. En son absence, arrêter tôt avec la commande de construction et l'option `--eval-every 0`, sans substitution par des parties Stockfish. Les erreurs de données ou d'évaluation doivent être loguées puis remontées ; le checkpoint déjà sauvegardé reste disponible. Un dépassement de la cible est un passage réussi et laisse l'entraînement continuer.

- [ ] Ajouter la configuration effective et les hashes au run W&B. Regrouper le journal self-play et le rapport d'évaluation dans le même `wandb.log(..., step=global_step)` par iteration, en tenant compte des logs de minibatches : vérifier que W&B reçoit effectivement le point au bon step, sans second envoi rejeté à un step déjà committé. Une exécution avec `--wandb-mode disabled` doit conserver les fichiers locaux. Libérer l'évaluateur du banc et son MCTS dans le bloc de nettoyage avant la génération suivante ; conserver ces objets seulement à l'intérieur d'un passage.

- [ ] Tester absence de Stockfish, absence du dataset avant génération, erreurs après checkpoint, reprise de comparaison, configuration workers journalisée, bonne association iteration/global_step et dépassement souple à 340 s. Exécuter `test_position_bench_training.py` et `test_selfplay_diag.py`. Commit `Remplace l ancre periodique par le banc de positions`.

## Tâche 9 : qualification réelle, distribution et documentation

**Files:** compléter `README.md` et `sync_to_sim.ps1` ; créer `data/position_bench/v1/README.md`, `manifest.json` et éventuellement `positions.jsonl.zst` ; créer `docs/superpowers/specs/2026-09-30-position-bench-resultats.md` après les mesures. Les fichiers de travail et résultats détaillés restent ignorés.

**Livrable :** banc fabriqué et audité, rapport de durée/répétabilité, documentation des commandes, intégration prête à utiliser. Aucune affirmation sur les performances n'est faite avant ces mesures.

- [ ] Dès que les tâches 2 et 3 et le mode profile de tâche 6 sont disponibles, extraire un petit échantillon externe et chronométrer 32 recherches à 384 simulations, réparties entre phases. Mesurer séparément replay, inférence directe et MCTS. Estimer le coût de 256 positions avec `256 * moyenne(durée_par_position)` et vérifier le coût direct par lots 1024 ; cette projection reste provisoire. Si le coût paraît régulièrement proche de 8-9 minutes, optimiser les coûts d'orchestration identifiés avant de figer v1. Ne pas baisser automatiquement les budgets validés.

- [ ] Lancer ensuite un pilote d'annotation sur 20 positions, tous coups légaux, et contrôler le protocole réel Stockfish, les mats, les hashes des réseaux et la reprise. Mesurer sa durée pour estimer honnêtement le temps de fabrication : 12 500 positions multipliées par leur nombre moyen de coups légaux à 200 000 noeuds représentent un travail hors ligne potentiellement long. La cible cinq minutes ne s'applique pas à cette fabrication.

```powershell
& $benchPython python_src/build_position_bench.py build --months 2026-08 2026-07 2026-06 --stockfish python_src/lichess_bot/stockfish/stockfish-windows-x86-64-universal.exe --workers 4 --work-dir data/position_bench_work --output data/position_bench/v1 --resume
```

Les quatre workers concernent uniquement la construction CPU hors ligne. Vérifier le chemin moteur local avant la commande ; cette valeur ne crée aucun processus Stockfish pendant l'évaluation récurrente. Le constructeur peut poursuivre les mois précédents autorisés si les quotas échouent, sans descendre jusqu'à février.

- [ ] Vérifier le manifeste complet : dates effectives, sources exclues, quotas/caps, 10 000 identités uniques, 256 IDs de recherche, WDL/cp/mat par coup légal, audit réussi, attribution et `training_forbidden=true`. Réexécuter la finalisation à partir du cache de travail dans un autre dossier et vérifier l'identité des labels et du fichier compressé.

- [ ] Sur la machine d'entraînement de référence, lancer trois évaluations complètes consécutives du même ONNX, dans des répertoires distincts et avec le contexte de VRAM de la boucle PyTorch documenté. Rien ne doit exécuter un autre benchmark GPU en parallèle. Conserver matériel, driver, versions et empreintes, durée totale, phases, moyenne/p95 par position MCTS et volume de sorties. Un passage à 305 ou 340 s n'est pas invalidé ; examiner les coûts si les passages approchent régulièrement 480-540 s. Ne pas prétendre garantir cinq minutes avant ce test.

- [ ] Comparer les trois sorties policy : mêmes argmax, probabilities à `1e-7`, métriques à `1e-6`. Pour MCTS, rapporter le bruit observé des regrets entre répétitions, sans le qualifier de négligeable avant mesure. Si les tolérances policy échouent, identifier la cause et l'exposer dans le rapport ; ne pas arrondir les résultats pour faire passer la validation.

- [ ] Réaliser un essai d'intégration court avec doublures de self-play/training puis un passage réel d'évaluation à partir d'un checkpoint, W&B en mode offline. Vérifier les courbes et les steps réellement présents dans les logs locaux. La validation n'exige pas de lancer une campagne d'entraînement longue ni d'envoyer des données en ligne.

- [ ] Exécuter la suite Python existante et les nouveaux tests, puis les tests C++ Release. Vérifier les échecs/skips plutôt que de les assimiler à des réussites. Les trois fichiers de diagnostic non suivis présents avant ce chantier ne font pas partie des changements et ne doivent pas être ajoutés au commit.

```powershell
& $benchPython -m pytest -p no:cacheprovider python_src/tests -q
& $benchCtest --test-dir build -C Release --output-on-failure
git diff --check
```

- [ ] Documenter construction/reprise, évaluation manuelle, comparaison, réagrégation, cadence, désactivation et interprétation des courbes dans README. Conserver les bornes de la mesure : accord à Stockfish sur positions externes, pas Elo. Mettre la licence/attribution et l'interdiction d'entraînement dans le README du dataset.

- [ ] Mesurer la taille du fichier compressé. Sous 100 Mio, committer le dataset avec son manifeste ; au-delà, conserver le manifeste dans Git et préparer l'artefact W&B hashé, dont l'envoi suit les autorisations de l'environnement. Le chargeur lit un fichier local et ne télécharge pas implicitement un artefact pendant un passage. Fournir une commande explicite de récupération via le SDK W&B puis vérification du hash.

- [ ] Mettre à jour la liste de transfert `sync_to_sim.ps1` : les six modules créés, `puzzle_bench.py`, `bench_metrics.py`, les dépendances déjà transférées (`move_coding.py`, `model.py`, `lib.py`), le module `.pyd` reconstruit et les DLL ONNX correspondantes, puis le dossier `data/position_bench/v1`. Les fichiers nécessaires au banc activé sont obligatoires et leur absence interrompt le transfert, contrairement aux éléments anciens optionnels. Relire les commandes de transfert sans lancer de synchronisation distante non demandée.

- [ ] Produire le rapport avec observations et limitations mesurées. Commit ciblé `Qualifie et documente le banc externe de positions`. Si l'artefact final ou la machine cible n'est pas disponible, qualifier seulement ce qui a été réellement vérifié et laisser les cases correspondantes ouvertes.

## Ordre d'exécution et couverture

Les tâches 1 à 8 construisent une fonctionnalité testable sur fixtures. La tâche 9 qualifie les données et performances réelles. Son essai précoce intervient dès que ses dépendances existent ; l'annotation complète attend cette estimation de coût.

| Exigence de la spec | Tâches |
|---|---|
| Source externe, >=2000, bots, provenance et exclusions | 3, 5, 9 |
| Historique complet, FEN, identité réseau et coups légaux | 1, 3, 6 |
| WDL/cp/mat, tous les coups, reprise et stabilité | 4, 5, 9 |
| Quotas exacts, plafonds, couleurs et sous-banc fixe | 5, 9 |
| Hashes, immutabilité, attribution, distribution | 3, 5, 7, 9 |
| GPU, policy/value batchées et MCTS séquentiel persistant | 2, 6 |
| Cible cinq minutes souple, pas de score partiel | 6, 7, 8, 9 |
| Policy/value/cp/MCTS, comparaisons appariées et W&B | 1, 7, 8 |
| Intégration cadence 4 et Stockfish manuel préservé | 8 |
| Tests purs, moteur réel, répétabilité et suite existante | 1 à 9 |

Les seuils temporels de l'ancienne spec sont remplacés par la clarification utilisateur. Les détails ajoutés ici (TT 8192 h0, binding de logits, cache de travail, comportement d'un second audit) rendent l'implémentation explicite ; ils sont consignés dans les métadonnées, sans changement caché du protocole entre checkpoints.

## Relecture du plan

- [x] Les sections de la spec ont une tâche associée.
- [x] Les interfaces utilisées entre tâches sont définies, y compris les fichiers de résultats et les réglages MCTS.
- [x] Le plan inclut les entrées et défaillances du Review Focus dans les tests des tâches concernées.
- [x] La boucle MCTS respecte le code relu : un appel complet par position, pool persistant et table remise à froid.
- [x] La cible de durée est souple ; aucun arrêt à cinq minutes, aucune réduction adaptative du banc.
- [x] Les commandes de qualification sont distinguées des contrôles réalisés lors de la rédaction.

Méthode d'exécution recommandée : réalisation directe dans cette conversation, tâche par tâche, car les contrats de données, résultats et intégration sont étroitement liés. L'utilisateur peut choisir une exécution avec sous-agents lors de la revue du plan.
