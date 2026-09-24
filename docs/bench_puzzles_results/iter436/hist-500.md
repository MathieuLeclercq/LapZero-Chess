# Banc de puzzles : resultats

Modele : `2026_04_30_09h53_iter436_unsupervised.onnx`, iteration None, global_step None
Banc : `data\puzzles_bench.txt`, bras avec historique
Recherche : 700 simulations, c_puct 1.4, batch 8, TT h0, 16 travailleurs, 1 workers de recherche, CPU
Duree : 5.1 min

Seul le PREMIER coup est score, celui ou il y a une tactique a trouver.
C'est aussi le seul coup que le self-play traite specialement, donc le
banc mesure ce que l'entrainement optimise.

La colonne reseau seul est une inference sans aucune recherche : c'est la
policy brute, la grandeur qui s'effondrait sur les puzzles prives
d'historique. La colonne recherche est le meme reseau avec le MCTS.

## Global

| | n | Reseau seul % | Recherche % | p med. du bon coup | part de visites med. |
|---|---|---|---|---|---|
| global | 500 | 47.2 (42.9 a 51.6) | 74.4 (70.4 a 78.0) | 0.287 | 0.956 |

## Par tranche de rating

| | n | Reseau seul % | Recherche % | p med. du bon coup | part de visites med. |
|---|---|---|---|---|---|
| 1000-1449 | 125 | 60.8 (52.0 a 68.9) | 88.0 (81.1 a 92.6) | 0.485 | 0.979 |
| 1450-1899 | 125 | 50.4 (41.8 a 59.0) | 81.6 (73.9 a 87.4) | 0.305 | 0.963 |
| 1900-2349 | 125 | 42.4 (34.1 a 51.2) | 70.4 (61.9 a 77.7) | 0.246 | 0.944 |
| 2350-2800 | 125 | 35.2 (27.4 a 43.9) | 57.6 (48.8 a 65.9) | 0.165 | 0.689 |

## Par theme tactique

Un puzzle portant plusieurs themes compte dans chacun : la ventilation
est multi-etiquettes et ne somme pas au total.

| | n | Reseau seul % | Recherche % | p med. du bon coup | part de visites med. |
|---|---|---|---|---|---|
| sacrifice | 122 | 24.6 (17.8 a 32.9) | 42.6 (34.2 a 51.5) | 0.069 | 0.123 |
| fork | 108 | 46.3 (37.2 a 55.7) | 80.6 (72.1 a 86.9) | 0.276 | 0.959 |
| pin | 89 | 36.0 (26.8 a 46.3) | 69.7 (59.5 a 78.2) | 0.185 | 0.926 |
| mateIn2 | 70 | 51.4 (40.0 a 62.8) | 87.1 (77.3 a 93.1) | 0.332 | 0.979 |
| deflection | 54 | 46.3 (33.7 a 59.4) | 64.8 (51.5 a 76.2) | 0.290 | 0.902 |
| discoveredAttack | 49 | 38.8 (26.4 a 52.8) | 67.3 (53.4 a 78.8) | 0.201 | 0.921 |
| attraction | 47 | 25.5 (15.3 a 39.5) | 48.9 (35.3 a 62.8) | 0.059 | 0.269 |
| mateIn1 | 44 | 75.0 (60.6 a 85.4) | 93.2 (81.8 a 97.7) | 0.647 | 0.986 |
| hangingPiece | 27 | 70.4 (51.5 a 84.1) | 88.9 (71.9 a 96.1) | 0.644 | 0.981 |
| mateIn3 | 20 | 45.0 (25.8 a 65.8) | 65.0 (43.3 a 81.9) | 0.193 | 0.929 |
| skewer | 19 | 31.6 (15.4 a 54.0) | 63.2 (41.0 a 80.9) | 0.201 | 0.861 |
| trappedPiece | 14 | 64.3 (38.8 a 83.7) | 85.7 (60.1 a 96.0) | 0.388 | 0.959 |
| capturingDefender | 8 | 50.0 (21.5 a 78.5) | 87.5 (52.9 a 97.8) | 0.357 | 0.891 |
| doubleCheck | 6 | 16.7 (3.0 a 56.4) | 50.0 (18.8 a 81.2) | 0.064 | 0.302 |
| xRayAttack | 4 | 25.0 (4.6 a 69.9) | 75.0 (30.1 a 95.4) | 0.207 | 0.721 |
| interference | 3 | 66.7 (20.8 a 93.9) | 66.7 (20.8 a 93.9) | 0.393 | 0.966 |

## Reseau seul contre recherche, comparaison appariee

Les deux colonnes portent sur les memes puzzles, donc McNemar
s'applique. La case qui compte est la premiere : si elle est grosse, la
recherche detruit des tactiques que la policy voyait deja, ce qui est un
diagnostic tout autre qu'un reseau faible.

- Reseau bon, recherche mauvaise : **17**
- Reseau mauvais, recherche bonne : **153**
- McNemar : chi2 = 107.21, p = 4.01e-25

## Comparaison avec / sans historique (passe appariee)

Meme modele, memes puzzles, meme budget : la seule difference est la
presentation. Tant que le raccourci « pas d'historique = position
tactique » est present, la passe sans historique gonfle le prior du
coup solution et la value. Le chiffre a suivre d'une campagne a
l'autre est l'ecart median du prior : il doit converger vers zero
quand le raccourci est desappris. Les autres lignes evitent de
conclure sur le seul prior.

- Puzzles apparies : **500**
- Prior median du coup solution : avec 0.287, sans 0.470, **ecart median -0.0818**
- Part de visites mediane du coup solution : ecart median -0.0064
- Value du reseau : avec -0.166, sans +0.935, ecart median -1.036
- Recherche : avec 372/500, sans 408/500
- McNemar apparie : reussi avec seul 36, reussi sans seul 72, chi2 = 11.34, p = 7.57e-04
- Accord des coups de recherche entre les deux passes : 74.0 %

Les deux rapports complets suivent : bras avec historique, puis bras
sans historique.

# Banc de puzzles : resultats

Modele : `2026_04_30_09h53_iter436_unsupervised.onnx`, iteration None, global_step None
Banc : `data\puzzles_bench.txt`, bras sans historique
Recherche : 700 simulations, c_puct 1.4, batch 8, TT h0, 16 travailleurs, 1 workers de recherche, CPU
Duree : 5.1 min

Seul le PREMIER coup est score, celui ou il y a une tactique a trouver.
C'est aussi le seul coup que le self-play traite specialement, donc le
banc mesure ce que l'entrainement optimise.

La colonne reseau seul est une inference sans aucune recherche : c'est la
policy brute, la grandeur qui s'effondrait sur les puzzles prives
d'historique. La colonne recherche est le meme reseau avec le MCTS.

## Global

| | n | Reseau seul % | Recherche % | p med. du bon coup | part de visites med. |
|---|---|---|---|---|---|
| global | 500 | 54.0 (49.6 a 58.3) | 81.6 (78.0 a 84.8) | 0.470 | 0.969 |

## Par tranche de rating

| | n | Reseau seul % | Recherche % | p med. du bon coup | part de visites med. |
|---|---|---|---|---|---|
| 1000-1449 | 125 | 80.8 (73.0 a 86.7) | 96.0 (91.0 a 98.3) | 0.851 | 0.993 |
| 1450-1899 | 125 | 59.2 (50.4 a 67.4) | 91.2 (84.9 a 95.0) | 0.538 | 0.979 |
| 1900-2349 | 125 | 37.6 (29.6 a 46.3) | 73.6 (65.3 a 80.5) | 0.275 | 0.941 |
| 2350-2800 | 125 | 38.4 (30.3 a 47.2) | 65.6 (56.9 a 73.4) | 0.193 | 0.759 |

## Par theme tactique

Un puzzle portant plusieurs themes compte dans chacun : la ventilation
est multi-etiquettes et ne somme pas au total.

| | n | Reseau seul % | Recherche % | p med. du bon coup | part de visites med. |
|---|---|---|---|---|---|
| sacrifice | 122 | 45.9 (37.3 a 54.7) | 74.6 (66.2 a 81.5) | 0.353 | 0.925 |
| fork | 108 | 60.2 (50.8 a 68.9) | 86.1 (78.3 a 91.4) | 0.597 | 0.978 |
| pin | 89 | 41.6 (31.9 a 52.0) | 69.7 (59.5 a 78.2) | 0.249 | 0.904 |
| mateIn2 | 70 | 61.4 (49.7 a 72.0) | 95.7 (88.1 a 98.5) | 0.740 | 0.988 |
| deflection | 54 | 53.7 (40.6 a 66.3) | 70.4 (57.2 a 80.9) | 0.418 | 0.918 |
| discoveredAttack | 49 | 51.0 (37.5 a 64.4) | 81.6 (68.6 a 90.0) | 0.392 | 0.967 |
| attraction | 47 | 42.6 (29.5 a 56.7) | 72.3 (58.2 a 83.1) | 0.323 | 0.920 |
| mateIn1 | 44 | 75.0 (60.6 a 85.4) | 95.5 (84.9 a 98.7) | 0.848 | 0.991 |
| hangingPiece | 27 | 51.9 (34.0 a 69.3) | 77.8 (59.2 a 89.4) | 0.483 | 0.957 |
| mateIn3 | 20 | 40.0 (21.9 a 61.3) | 85.0 (64.0 a 94.8) | 0.275 | 0.959 |
| skewer | 19 | 47.4 (27.3 a 68.3) | 68.4 (46.0 a 84.6) | 0.168 | 0.939 |
| trappedPiece | 14 | 50.0 (26.8 a 73.2) | 78.6 (52.4 a 92.4) | 0.358 | 0.959 |
| capturingDefender | 8 | 50.0 (21.5 a 78.5) | 100.0 (67.6 a 100.0) | 0.334 | 0.917 |
| doubleCheck | 6 | 33.3 (9.7 a 70.0) | 66.7 (30.0 a 90.3) | 0.347 | 0.808 |
| xRayAttack | 4 | 0.0 (0.0 a 49.0) | 100.0 (51.0 a 100.0) | 0.138 | 0.905 |
| interference | 3 | 66.7 (20.8 a 93.9) | 100.0 (43.8 a 100.0) | 0.783 | 0.990 |

## Reseau seul contre recherche, comparaison appariee

Les deux colonnes portent sur les memes puzzles, donc McNemar
s'applique. La case qui compte est la premiere : si elle est grosse, la
recherche detruit des tactiques que la policy voyait deja, ce qui est un
diagnostic tout autre qu'un reseau faible.

- Reseau bon, recherche mauvaise : **9**
- Reseau mauvais, recherche bonne : **147**
- McNemar : chi2 = 120.31, p = 5.40e-28
