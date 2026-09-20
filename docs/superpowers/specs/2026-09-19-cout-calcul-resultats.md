# Cout de calcul de la recherche : resultats de l'assainissement

Date : 2026-09-19

Statut : taches 1, 3 et 5 du plan `2026-09-19-cout-calcul-optimisations.md` executees ;
taches 2, 4 et 6 restantes ; le lot fixe est active dans `uci.py`.

Modele : `2026_04_23_23h25_iter316_unsupervised.onnx`.

Materiel : RTX PRO 2000 Blackwell Laptop 8 Gio, 16 coeurs logiques.

Spec : `2026-09-19-cout-calcul-cpu-gpu.md`.

## 1. Resume

Le decrochage entre le banc self-play (0.38 ms par position) et la recherche (1.4 a
2 ms par position) venait du **changement de forme du lot** a chaque appel ONNX Runtime.
En lots alternee 1 a 8, `session->Run` coute 13.4 ms quel que soit le lot, contre 2.7 a
3.7 ms a forme fixe. Le moteur padde desormais les vagues et les lots mono avec le dernier
tenseur, ignore les sorties dupliquees, et la recherche est inchangee.

Gains mesures en A/B interleaved, 700 simulations, lot 8, pool chaud :

| position | chemin | w8 avant | w8 apres | gain | w1 avant | w1 apres | gain |
|---|---|---|---|---|---|---|---|
| ouverture | step_analysis | 1035 | 2068 | x2.00 | 1018 | 2049 | x2.01 |
| milieu | step_analysis | 485 | 1542 | x3.18 | 465 | 1497 | x3.22 |
| finale | step_analysis | 767 | 1958 | x2.55 | 670 | 1843 | x2.75 |

Le p95 de latence est divise par 2 a 3 (rapports 0.30 a 0.56), et le debit du bot est
monte de 500 a 850 vers 1 500 a 2 100 simulations par seconde.

## 2. Tache 1 : banc evaluateur hors arbre

Nouvelle cible `evaluator_batch_bench` (`tests/cpp/evaluator_batch_bench.cpp`), avec
`--fake`, `--model PATH`, `--gpu`, `--batches`, `--repetitions`, `--rounds`, `--variable`
et timings internes separes (`run_ns`, `softmax_ns`), desactives par defaut en production.

Lots fixes, GPU, medianes de 30 repetitions :

| lot | ms par appel | ms par position | run ONNX |
|---|---|---|---|
| 1 | 9.75 | 9.75 | 4.95 |
| 2 | 6.53 | 3.26 | 4.23 |
| 4 | 5.32 | 1.33 | 3.71 |
| 8 | 3.3 a 5.1 | 0.41 a 0.64 | 2.7 a 3.7 |
| 16 | 5.83 | 0.36 | 3.95 |
| 32 | 5.86 | 0.18 | 4.15 |
| 64 | 8.79 | 0.14 | 6.16 |
| 128 | 13.18 | 0.10 | 9.77 |
| 256 | 41.43 | 0.16 | 18.12 |

Le cout par position continue de baisser jusqu'a 128, mais l'arbre ne fournit que 7
feuilles par vague. Le softmax C++ vaut 0.02 a 0.10 ms par appel, soit 0.2 a 2 pour cent :
il n'est pas un levier. Le CPU, lui, coute environ 3 ms par position quel que soit le lot.

En lots alternee (`--variable`), chaque appel coute 13.5 ms et le Run 13.4 ms, a toutes
les tailles. A/B interleaved serre : 13.03 et 13.04 ms en variable contre 2.87 et 3.65 ms
a lot fixe 8, soit un facteur 4.

## 3. Tache 3 : lot de forme fixe

Implementation : `MCTS::set_fixed_batch(bool)`, defaut faux. Le coordinateur duplique le
dernier tenseur jusqu'a `batch_size`, appelle l'evaluateur avec la forme fixe et
n'utilise que les `network_count` premieres sorties. Les compteurs gardent leur sens :
`nn_calls` compte les positions reelles, `nn_batches` les appels. La racine reste au lot 1,
donc deux formes seulement, 1 et 8.

L'extension au chemin mono (`mcts_search`) donne les memes gains, ce qui accelere le banc
de puzzles et les tournois.

Activation : `MCTS_FIXED_BATCH = True` dans `uci.py`, a cote de `MCTS_BATCH_SIZE` et
`MCTS_WORKER_COUNT`. Tests : `test_fixed_batch_pads_wave_calls_and_keeps_budget`,
`test_fixed_batch_pads_mono_batch`, `test_le_constructeur_active_le_lot_fixe_par_defaut`.

## 4. Tache 5 : export FP16 rejete

Conversion `onnxruntime.transformers.float16.convert_float_to_float16` avec
`keep_io_types=True`, puis A/B interleaved sur le banc evaluateur, lots 1, 8 et 32. Le
FP16 est **plus lent** de 18 a 28 pour cent selon le lot. Rejet sans campagne qualite :
aucun code de production ajoute, modele FP16 conserve hors depot dans
`python_src/checkpoints_onnx/`.

## 5. Tache 6 : cas chaud, et correction d'une reserve

Parties auto-jouees, ouverture et finale, 12 coups, 700 simulations par coup, tranches de
64, lot fixe (`out/multicore/hot-tree.json`) :

- debit stable de 1 750 a 2 240 simulations par seconde apres le premier coup, qui paie
  l'initialisation de la session ONNX ;
- **environ 700 appels reseau pour 700 simulations a chaque coup**, alors que les hits de
  table montent de 8 a 81 pour cent.

Un hit de table materialise les enfants depuis la table et la descente continue vers une
feuille reseau. Il n'evite donc pas l'inference et la reserve Amdahl des rapports
precedents est sans objet. Les gains mesures s'appliquent tels quels en partie reelle ;
seuls les terminaux evitent le reseau.

## 6. Verification

- 15/15 CTest, dont le smoke du banc evaluateur et les deux tests de padding.
- 256/256 pytest, dont les tests de propagation du lot fixe et du UCI.
- A/B interleaved pour toutes les comparaisons de debit, meme session, ordre alterne.

## 7. Taches restantes

- Tache 2, buffers persistants : la mesure a montre que le facteur dominant etait la
  forme du lot, pas les allocations. Faible valeur attendue desormais.
- Tache 4, balayage de divergence : les collisions sont maintenant un cout relatif plus
  visible, mais le balayage et ses campagnes qualite restent a faire.
- Tache 6, cible commitee `hot_tree_bench` : la mesure a ete faite par script ; la cible
  testable reste a ecrire.
- Tache 7 complete : mesures finales, decision sur les reglages et verdict sur la
  production continue.

## 8. Commits de ce chantier

- `4161cd5` Mesure la courbe du lot de l evaluateur hors arbre
- `29f9458` Padde les lots d evaluation a une forme fixe
- `74122ac` Active le lot fixe dans le moteur UCI
- `6606d97` Padde aussi les lots du chemin mono
- `4461d80` Consigne la cause du decrochage et le gain du lot fixe
