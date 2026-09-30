# Banc externe de positions : état d'implémentation

Date : 2026-09-30
Plan : `docs/superpowers/plans/2026-09-30-position-bench.md` (tâches 1 à 8)
Conception : `docs/superpowers/specs/2026-09-29-position-bench-design.md`

## Statut

Le banc est implémenté et testé sur fixtures, sans qualification réelle : le
PC d'entraînement, le binaire Stockfish de qualification et le réseau Lichess
n'étaient pas disponibles dans cet environnement. Les cases ouvertes de la
tâche 9 sont listées plus bas ; aucune affirmation de performance n'est faite
avant leurs mesures.

## Vérifié

- Tâche 1 : contrats JSON, WDL, regrets (attendu, argmax, masses, centipions),
  value (MAE, RMSE, biais, Pearson), agrégations globales et par phase,
  catégorie WDL et type de joueur. 37 tests.
- Tâche 2 : `ONNXEvaluator.predict_batch` rend les logits bruts et la valeur
  d'un lot, refuse dtype, contiguïté, forme et sorties non finies, et reste
  utilisable après un rejet. 11 tests plus `evaluator_contract` et
  `softmax_rows` verts.
- Tâche 3 : lecture PGN et PGN.zst en flux, téléchargement `.part` avec hash,
  filtres de parties, identités joueur et événement, phases, sélection par
  hash, rejeu C++ avec FEN, bijection UCI/index et identité réseau. Le test de
  stabilité mélange les parties et retire les commentaires et variantes sans
  changer le résultat. 14 tests.
- Tâche 4 : annotation Stockfish reprenable, point de vue du trait, refus des
  scores bornés, WDL invalide, PV incohérente et réseaux non identifiables ;
  cache SQLite qui refuse une autre configuration. 24 tests.
- Tâche 5 : sélection gloutonne par plus grand déficit relatif, neuf quotas,
  plafonds événement et joueur, borne de 55 % par couleur, sous-banc de 256 par
  plus grands restes (26, 15, 10, 77, 46, 31, 26, 15, 10), audit d'échantillon,
  publication JSONL/zstd triée et manifeste immuable. 17 tests.
- Tâche 6 : lots policy de 4, 4 et 2 sur dix positions, réglages transmis au
  MCTS unique, budget exact de 384 simulations, refus des données incohérentes
  et des sorties hors `[-1, 1]`, cible de durée sans effet sur le travail,
  mode `profile` sans étiquettes. 16 tests.
- Tâche 7 : chargement contrôlé du dataset, sauvegarde NPZ et sidecar avec
  hashes croisés, comparaison appariée par identifiant avec bootstrap à graine
  fixe, noms W&B de la section 8.3 et tranches, refus des datasets ou
  protocoles différents. 20 tests.
- Tâche 8 : cadence 4 sur reprise (évaluations aux itérations 4 et 8), cadence
  0, cadence négative refusée, migration des paramètres et drapeaux Stockfish,
  prévalidation du banc avant la première génération. 12 tests.
- Suites complètes : 476 tests pytest verts, 1 saut (smoke test Stockfish
  opt-in par `POSITION_BENCH_STOCKFISH`), 19 tests CTest Release verts,
  ruff et pyright propres sur les fichiers ajoutés.

## Ouvert

- [ ] Construire `data/position_bench/v1` sur une machine avec réseau et
      Stockfish : extraction, criblage, annotation de la réserve, audit, puis
      vérification du manifeste (10 000 positions, 256 identifiants, WDL et
      score par coup légal, `training_forbidden`, attribution).
- [ ] Auditer un pilote d'annotation de 20 positions et mesurer honnêtement le
      coût de fabrication avant l'annotation complète.
- [ ] Chronométrer trois passages complets consécutifs du même ONNX sur la
      machine d'entraînement de référence, avec la décomposition `replay_s`,
      `policy_s`, `mcts_setup_s`, `mcts_s`, `aggregate_s`, `write_s`, et la
      moyenne et le p95 par position MCTS. Un passage à 305 ou 340 s reste
      valide ; viser une analyse si la durée tend vers 8 ou 9 minutes.
- [ ] Vérifier la répétabilité : mêmes argmax, probabilités à `1e-7`,
      métriques à `1e-6` sur les trois passages ; rapporter le bruit des
      regrets MCTS sans le qualifier avant mesure.
- [ ] Vérifier la taille du fichier compressé : committer le dataset sous
      100 Mio, sinon publier un artefact W&B hashé et le documenter.
- [ ] Essai d'intégration court avec `run_selftrain.py` (W&B en mode offline)
      pour contrôler les points réellement écrits au bon `global_step`.
- [ ] Renseigner `README.md` du dataset (provenance, attribution, interdiction
      d'entraînement) une fois `v1` publié.
- [ ] Après qualification, décider du remplacement des chiffres de référence
      du banc de puzzles pour les décisions qualité.
