# Banc de puzzles : resultats

Modele : `2026_04_23_23h25_iter316_unsupervised.onnx`, iteration 316, global_step 19415
Banc : `..\data\puzzles_bench.txt`, bras avec historique
Recherche : 700 simulations, c_puct 1.4, batch 8, 16 travailleurs
Duree : 10.7 min

Seul le PREMIER coup est score, celui ou il y a une tactique a trouver.
C'est aussi le seul coup que le self-play traite specialement, donc le
banc mesure ce que l'entrainement optimise.

La colonne reseau seul est une inference sans aucune recherche : c'est la
policy brute, la grandeur qui s'effondrait sur les puzzles prives
d'historique. La colonne recherche est le meme reseau avec le MCTS.

## Global

| | n | Reseau seul % | Recherche % | p med. du bon coup | part de visites med. |
|---|---|---|---|---|---|
| global | 2500 | 45.7 (43.7 a 47.6) | 76.8 (75.1 a 78.4) | 0.286 | 0.957 |

## Par tranche de rating

| | n | Reseau seul % | Recherche % | p med. du bon coup | part de visites med. |
|---|---|---|---|---|---|
| 1000-1449 | 625 | 60.0 (56.1 a 63.8) | 90.7 (88.2 a 92.8) | 0.439 | 0.977 |
| 1450-1899 | 625 | 45.1 (41.3 a 49.0) | 78.9 (75.5 a 81.9) | 0.271 | 0.963 |
| 1900-2349 | 625 | 38.4 (34.7 a 42.3) | 71.8 (68.2 a 75.2) | 0.221 | 0.937 |
| 2350-2800 | 625 | 39.2 (35.4 a 43.1) | 65.8 (62.0 a 69.4) | 0.206 | 0.873 |

## Par theme tactique

Un puzzle portant plusieurs themes compte dans chacun : la ventilation
est multi-etiquettes et ne somme pas au total.

| | n | Reseau seul % | Recherche % | p med. du bon coup | part de visites med. |
|---|---|---|---|---|---|
| fork | 574 | 45.3 (41.3 a 49.4) | 82.1 (78.7 a 85.0) | 0.274 | 0.961 |
| sacrifice | 523 | 21.6 (18.3 a 25.3) | 48.6 (44.3 a 52.8) | 0.063 | 0.334 |
| pin | 438 | 37.2 (32.8 a 41.8) | 70.3 (65.9 a 74.4) | 0.202 | 0.923 |
| mateIn2 | 288 | 53.8 (48.0 a 59.5) | 85.8 (81.3 a 89.3) | 0.389 | 0.976 |
| deflection | 273 | 39.6 (33.9 a 45.5) | 72.9 (67.3 a 77.8) | 0.248 | 0.930 |
| discoveredAttack | 270 | 37.4 (31.9 a 43.3) | 70.0 (64.3 a 75.2) | 0.170 | 0.924 |
| attraction | 246 | 28.9 (23.6 a 34.8) | 56.5 (50.3 a 62.6) | 0.094 | 0.733 |
| mateIn1 | 200 | 67.0 (60.2 a 73.1) | 95.0 (91.0 a 97.3) | 0.548 | 0.980 |
| hangingPiece | 167 | 80.8 (74.2 a 86.1) | 95.2 (90.8 a 97.6) | 0.731 | 0.986 |
| mateIn3 | 112 | 41.1 (32.4 a 50.3) | 62.5 (53.3 a 70.9) | 0.144 | 0.914 |
| skewer | 86 | 41.9 (32.0 a 52.4) | 74.4 (64.3 a 82.5) | 0.228 | 0.959 |
| trappedPiece | 58 | 56.9 (44.1 a 68.8) | 84.5 (73.1 a 91.6) | 0.337 | 0.959 |
| capturingDefender | 57 | 31.6 (21.0 a 44.5) | 75.4 (62.9 a 84.8) | 0.159 | 0.921 |
| doubleCheck | 26 | 34.6 (19.4 a 53.8) | 61.5 (42.5 a 77.6) | 0.146 | 0.943 |
| interference | 22 | 59.1 (38.7 a 76.7) | 90.9 (72.2 a 97.5) | 0.426 | 0.976 |
| xRayAttack | 21 | 28.6 (13.8 a 50.0) | 71.4 (50.0 a 86.2) | 0.226 | 0.957 |

## Reseau seul contre recherche, comparaison appariee

Les deux colonnes portent sur les memes puzzles, donc McNemar
s'applique. La case qui compte est la premiere : si elle est grosse, la
recherche detruit des tactiques que la policy voyait deja, ce qui est un
diagnostic tout autre qu'un reseau faible.

- Reseau bon, recherche mauvaise : **46**
- Reseau mauvais, recherche bonne : **824**
- McNemar : chi2 = 693.94, p = 6.21e-153

## Comparaison appariee au chemin sequentiel

La reference batch 0 et ce passage portent sur les memes 2 500 puzzles.

- Batch 0 : **1 917 / 2 500** (76,68 %)
- Batch 8 : **1 920 / 2 500** (76,80 %)
- Batch 0 bon, batch 8 mauvais : **15**
- Batch 0 mauvais, batch 8 bon : **18**
- McNemar exact bilateral : **p = 0,73**

Le batch 8 ne montre donc aucune degradation mesurable a budget de recherche
constant. La policy brute est identique sur les 2 500 puzzles et aucune erreur
de donnees n'a ete relevee.
