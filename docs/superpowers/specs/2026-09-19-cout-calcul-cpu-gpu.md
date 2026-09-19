# Repartition du cout de calcul : CPU, GPU, et pistes d'optimisation

Date : 2026-09-19
Statut : etude, a completer par les tests proposes en section 9
Portee : recherche UCI par vagues, modele iter316, GPU RTX PRO 2000 Blackwell 8 Gio

## 1. But

Faire l'etat des lieux des zones de calcul de la recherche (descente, tenseur, table,
evaluateur), chiffrer leur part respective, expliquer pourquoi un remplissage de 6 a 7 ne
multiplie pas le debit par 6, lister les leviers d'optimisation, et decrire les
experiences qui permettraient de trancher.

Cette etude ne produit aucune mesure nouvelle. Elle agrege des campagnes deja realisees
dans le cadre des taches 9 et 10 du plan multicœur, disponibles dans `out/multicore/`.

## 2. Sources et protocole

Campagnes utilisees :

- `out/multicore/phases.json`, `phases.md` : 700 simulations, GPU, lot 8, pool chaud,
  3 mesures par configuration, chronometrages actifs. Sert a la repartition par phase.
- `out/multicore/after.json`, `after.md` : 700 simulations, GPU, lot 8, workers 1/2/4/8,
  5 passages de 6 repetitions, pool chaud, ordre des workers alterne. Sert au choix du
  candidat et aux medianes.
- `out/multicore/queue.json` : 50 passages, 700 simulations, workers 1/2/8, pool chaud.
  Sert a la queue de latence (p95).
- `out/multicore/avant-r*.json`, `apres-r*.json` : comparaison interleaved ancien moteur
  contre nouveau workers1.
- `out/multicore/selfplay-after-fake.json`, `selfplay-after-gpu.json` : banc self-play
  partage, 8 plateaux, 100 vagues, 30 repetitions. Sert d'ancre pour le cout de l'appel.
- `docs/superpowers/specs/2026-09-14-search-bench-batching-gpu.md` : balayage GPU du lot,
  lots 0 a 64, 400 simulations, une seule session.
- `docs/superpowers/specs/2026-09-11-search-bench-resultats.md` : reference d'avant
  batching, CPU, 400 simulations.

Reglages communs : politique TT h0, taille de table 8192 (hors bot, qui utilise
4 000 000), c_puct 1.4, trois positions de reference (ouverture, milieu, finale).

Reserve importante : les valeurs absolues varient de 15 a 30 pour cent entre sessions
(etat d'horloge et thermique du GPU). Les rapports mesures dans une meme session sont
fiables, les comparaisons entre sessions le sont beaucoup moins. La campagne de queue a
50 passages et l'ordre alterne ont ete concus pour cette raison.

## 3. Debit observe

### 3.1 Avant batching

Reference CPU (2026-09-11), 400 simulations, une inference par simulation :

| position | chemin | sims/s (med) | inferences/s |
|---|---|---|---|
| finale | mcts_search | 292.2 | 292.9 |
| milieu | mcts_search | 292.6 | 293.3 |
| ouverture | mcts_search | 286.0 | 286.8 |

Le debit est plat a 2 pour cent pres quelle que soit la position. C'est la signature d'un
cout domine par la latence d'inference a batch 1, pas par la generation de coups.

### 3.2 Batching, meme session

Balayage GPU 2026-09-14, 400 simulations :

| position | lot 0 | lot 1 | lot 2 | lot 4 | lot 8 | lot 16 | lot 32 | lot 64 |
|---|---|---|---|---|---|---|---|---|
| ouverture | 316.8 | 325.9 | 569.2 | 708.5 | 785.5 | 775.3 | 783.1 | 813.7 |
| milieu | 328.3 | 319.6 | 445.9 | 446.7 | 467.9 | 467.2 | 471.8 | 477.3 |
| finale | 323.7 | 316.3 | 453.1 | 734.8 | 708.2 | 611.8 | 640.3 | 646.6 |

Remplissage reel (positions evaluees par appel) au lot 8 : 6.4 en ouverture, 5.0 en
milieu, 5.7 en finale. Au dela de 8, le debit est plat et le remplissage sature autour de
5 a 10 : l'arbre ne fournit pas plus de feuilles utiles.

### 3.3 Vagues

700 simulations, GPU, lot 8, pool chaud (`after.json`, medianes sur 30 observations) :

| position | chemin | w1 | w2 | w4 | w8 |
|---|---|---|---|---|---|
| ouverture | step_analysis | 733.5 | 812.3 | 835.0 | 846.8 |
| milieu | step_analysis | 328.7 | 340.7 | 356.1 | 348.8 |
| finale | step_analysis | 462.2 | 583.8 | 579.2 | 565.1 |

Le gain des vagues contre workers1 va de +21 pour cent (finale) a +7 pour cent (milieu,
ouverture), et il plafonne des 2 workers. La comparaison interleaved avec l'ancien moteur
ne montre pas de regression mono-worker reproductible ; l'ancien moteur fluctue de 700 a
1200 sims/s selon l'etat d'horloge, le nouveau reste stable.

## 4. Repartition par phase

Chemin mono-worker, 700 simulations, phases sequentielles propres (`phases.md`, lignes
w1). Toutes les valeurs sont en millisecondes.

| position | chemin | total | evaluateur | selection | tenseur et cle | expansion | backup | TT | assemblage |
|---|---|---|---|---|---|---|---|---|---|
| finale | step_analysis | 1447.5 | 1426.4 | 9.3 | 6.0 | 1.1 | 0.4 | 0.6 | 1.4 |
| milieu | step_analysis | 2063.5 | 2038.1 | 10.7 | 8.6 | 2.0 | 0.3 | 0.6 | 1.5 |
| ouverture | step_analysis | 923.8 | 903.2 | 8.0 | 7.4 | 1.6 | 0.3 | 0.5 | 1.0 |

L'evaluateur represente 96.6 a 98.5 pour cent du temps selon la position et le nombre de
workers. Tout le reste additionne fait 20 a 25 ms sur 700 simulations.

En multicœur (workers 8), la part non-evaluateur inclut l'attente de barriere :

- finale step : total 1300.4, evaluateur 1264.0 (97.2 %), attente workers 24.3.
- milieu step : total 2335.3, evaluateur 2280.8 (97.7 %), attente workers 40.7.
- ouverture step : total 853.3, evaluateur 824.2 (96.6 %), attente workers 19.3.

L'attente de barriere vaut 1.5 a 2.3 pour cent du total. Le reste du CPU worker est
cache par le parallelisme.

## 5. Cout CPU par simulation

Sur le chemin w1, le cout CPU total (tout sauf l'evaluateur) vaut :

- finale : 21.1 ms pour 700 simulations, soit 30 microsecondes par simulation.
- milieu : 25.4 ms, soit 36 microsecondes par simulation.
- ouverture : 20.6 ms, soit 29 microsecondes par simulation.

Contre 2.0 a 2.9 ms d'evaluateur par simulation, cela represente 1 a 1.5 pour cent du
temps. La simulation de coups n'est pas le goulet.

Decomposition par simulation, chemin w1 :

| phase | contenu | cout par simulation |
|---|---|---|
| selection | scoring UCB, application et annulation des coups de la descente | 11 a 15 us |
| tenseur et cle | getAlphaZeroTensor, hash semantique | 8.5 a 12 us |
| expansion | creation d'un enfant par coup legal | 1.5 a 3 us |
| assemblage | copie du tenseur dans le lot | 1 a 2 us |
| TT | probe et store | moins de 1 us |
| backup | remontee des valeurs | moins de 1 us |

Pour comparaison, le perft fait 1.8 million de noeuds par seconde, soit 0.56 us par
noeud. Un noeud de perft n'est pas comparable a un noeud d'arbre : `movePiece` revalide
la legalite du coup via `getLegalMovesForSquare`, et met a jour Zobrist, drapeaux de
roque, prise en passant, compteur des 50 coups et historiques. Un coup d'arbre coute
quelques microsecondes, pas quelques dixiemes, mais cela reste negligeable.

## 6. Pourquoi un remplissage de 6 ne multiplie pas le debit par 6

Le lot de N positions n'execute pas N positions dans le temps d'une seule. Il paie une
fois les frais fixes, puis N fois le cout par position :

```
T(N) = F + M * N
cout par simulation = F / N + M
```

Le gain maximal du batching est donc `(F + M) / M`, pas `N`. Au dela d'un certain
remplissage, `F / N` tend vers zero et il ne reste que `M`.

Ajustement approximatif sur le balayage 2026-09-14. Le lot 1 donne `F + M`, et le lot 8
donne `F + remplissage * M`, ou le remplissage est le nombre de positions reellement
evaluees par appel :

| position | F estime | M estime | plafond (F+M)/M | cout/sim a remplissage observe |
|---|---|---|---|---|
| ouverture | 2.2 ms | 0.92 ms | x3.4 | 1.27 ms (6.4) |
| finale | 2.0 ms | 1.05 ms | x2.9 | 1.41 ms (5.7) |
| milieu | 1.1 ms | 1.9 ms | x1.6 | 2.14 ms (5.0) |

L'ajustement a deux points est grossier et sensible a la derive d'horloge, mais il
explique les ordres de grandeur : a remplissage 6, les frais fixes sont deja presque
entierement amortis, et le plafond du batching seul est de x1.6 a x3.4 selon la position.
Les gains mesures (x1.4 a x2.5) sont coherents.

Point cle : `M` n'est pas du travail CPU de recherche. Il est entierement a l'interieur de
la phase evaluateur, donc dans l'appel ONNX Runtime : execution GPU du lot,
synchronisation, copies de sortie, et le softmax C++ sur 4672 logits par position
(`onnx_evaluator.cpp:64-80`). Pour un ResNet de 10 blocs et 128 filtres, le calcul brut
est de l'ordre de 0.2 GFLOP par position, soit quelques dizaines de microsecondes sur ce
GPU. Le reste du cout par position n'est donc pas du FLOP, mais de l'occupation GPU et
des frais par position.

Hypothese principale : les cartes de caracteristiques font 8x8, donc chaque noyau est
lance sur tres peu de threads et le GPU est loin d'etre sature. Si c'est le cas, un lot
nettement plus gros pourrait faire baisser le cout par position. La courbe de balayage ne
peut pas le montrer, parce que l'arbre plafonne le remplissage autour de 7.

### 6.1 Decrochage entre le banc self-play et la recherche

Il y a un indice plus fort que l'occupation, mesure le meme jour, avec le meme modele et le
meme GPU. Le banc self-play (`out/multicore/selfplay-after-gpu.json`) evalue 8 positions
par appel en 3.0 ms, soit 0.38 ms par position (305 ms pour 100 vagues de 8). La recherche
UCI paie 10 a 14 ms par appel a remplissage 5 a 7, soit 1.4 a 2 ms par position. Facteur 4
a 5, sans toucher au reseau.

La difference est structurelle, pas calculatoire :

- le self-play evalue toujours un lot de forme fixe, 8, en boucle serree, avec un buffer
  reutilise ;
- la recherche a des lots de taille variable (1 a 8 selon les collisions et la barriere),
  reconstruit un vecteur d'entree par vague, et laisse ONNX Runtime reallouer ses sorties
  a chaque appel.

Quand la forme du tenseur change, ONNX Runtime recalcule son plan memoire et peut refaire
des allocations GPU. `cudaMalloc` est couteux et synchronisant. Avec une forme fixe, tout
est amorti. C'est l'explication la plus probable de l'ecart, et elle est testable (T1b).

Consequence : le premier facteur 3 a 5 n'est pas dans le reseau, il est dans la plomberie
de l'appel. Le lot d'un seul arbre plafonnant vers 7, les frais fixes restent amortis sur
7 positions, pas sur 128 : le gain total realiste de ce chantier est de l'ordre de x3 a
x6, pas de x30.

## 7. Zones de calcul, synthese

| zone | part du temps | ce qui la borne | levier principal |
|---|---|---|---|
| Evaluateur (ONNX, GPU, softmax) | 96 a 98 % | forme du lot, frais par appel et cout par position | forme fixe et buffers reutilises, puis FP16/fusion |
| Selection (dont coups) | 0.5 a 1 % | coups joues par descente | marginal |
| Tenseur et cle | 0.4 a 0.6 % | 119x64 flottants par feuille | marginal |
| Expansion, backup, TT | moins de 0.3 % | structure de l'arbre | marginal |
| Attente de barriere (vagues) | 1.5 a 2.3 % | worker le plus lent de la vague | production continue |

## 8. Pistes d'optimisation

### 8.1 Assainir l'appel a l'evaluateur

Priorite 1, avant toute piste reseau : reproduire le regime du banc self-play (3.0 ms
pour un lot de 8).

- Toujours evaluer un lot de forme fixe, 8, quitte a dupliquer une position quand la vague
  en a moins, et reutiliser les buffers d'entree et de sortie sans allocation par appel.
  Voir le test T1b.
- Supprimer les `resize` de sortie par appel (`onnx_evaluator.cpp:61-62`) et preallouer.
- IO binding, memoire epinglee, capture CUDA graph pour reduire les frais par appel.
- Mesurer la vraie courbe du GPU hors arbre (T1) avant de conclure sur la taille de lot.
- Faire le softmax dans le graphe exporte, ou ne l'executer que sur les coups legaux (une
  trentaine) au lieu des 4672 (`onnx_evaluator.cpp:64-80`).

### 8.2 Augmenter le remplissage utile

- Production continue : une file decouple la production des feuilles de l'inference et
  permet de remplir des lots plus gros que la largeur d'une vague. C'est le seul levier
  qui peut reellement depasser le plafond des vagues.
- Divergence des descentes, du moins invasif au plus invasif :
  - inclure `n_in_flight` du parent dans `exploration_factor` (`mcts.cpp:277`,
    `mcts_wave.cpp:222`), indique comme premiere variante dans la spec de batching ;
  - amplitude du virtual loss, aujourd'hui 1 par descente (`mcts_reservation.cpp:36`) ;
  - c_puct, aujourd'hui 1.4 en dur ;
  - coefficient FPU, aujourd'hui 0.3 en dur (`mcts.cpp:271`, `mcts_wave.cpp:219`) ;
  - budget de tentatives de collision (`mcts_wave.cpp:284`) et fermeture du lot mono au
    premier `Pending` (`mcts_batch.cpp:130-133`).
  - Toute modification de ces valeurs change la recherche : validation qualite par le
    banc de puzzles obligatoire.
- Bruit de Dirichlet ou temperature de policy : a eviter pour le bot, reserve a
  l'entrainement.
- Collecte orientee diversite : repartir la vague sur des enfants distincts de la racine
  plutot que premier arrive. Structurel, mais attaque directement le remplissage.

### 8.3 Reduire le CPU de recherche

Gains marginaux par construction (1 a 1.5 % du temps), a traiter seulement si on touche a
la fonctionnalite. Le seul poste notable est le tenseur et la cle (8 a 12 us par feuille),
qui remplit 119x64 flottants et parcourt l'historique.

### 8.4 Le cas chaud

Toutes les mesures ci-dessus sont a arbre froid. En partie reelle, `update_root` conserve
le sous-arbre et toutes ses positions dans la table. Une fraction des simulations evite
alors le reseau, et par la loi d'Amdahl le gain du batching et de la production continue
diminue. Ce cas n'est pas mesure.

### 8.5 Le reseau et la quantification

Le reseau fait 10 blocs de 128 filtres avec blocs SE, environ 14.5 Mo de poids en FP32
(dont 11.8 Mo pour les blocs et 2.1 Mo pour la tete value), et environ 0.4 GFLOP par
position. A 1 ms par position, cela correspond a environ 0.4 TFLOP/s effectifs, soit
quelques pour cent seulement de la puissance crete du GPU. Le reseau n'est donc pas limite
par le calcul, mais par l'occupation (cartes 8x8) et par les frais par appel.

Pistes, du plus simple au plus lourd :

- FP16 : conversion de l'ONNX exporte (`convert_float_to_float16`) ou export en half.
  Divise par deux le trafic des poids et active les tensor cores. A valider sur le banc de
  puzzles, la value et les logits bougeant legerement.
- Softmax dans le graphe, ou sur les seuls coups legaux : environ 4672 exponentielles par
  position aujourd'hui.
- TensorRT EP : fusionne les couches et choisit de meilleurs algorithmes pour du 8x8.
  Piste la plus prometteuse si l'on est limite par l'occupation et le nombre de noyaux, au
  prix d'une dependance et d'une validation.
- INT8 : support limite sur le provider CUDA, et la quantification dynamique sert mal les
  convolutions. Peu prometteur ici.
- Reseau plus leger ou distille : exige un reentrainement, hors de ce chantier.

Plancher theorique : a efficacite realiste sur des convolutions 8x8, le calcul seul
represente environ 0.05 a 0.15 ms par position. C'est le plancher absolu, inatteignable en
jeu reel a cause du remplissage plafonne vers 7.

## 9. Tests a faire

### T1 : courbe du GPU hors arbre

Le test qui separe le plafond de l'arbre de celui du GPU. Un petit banc C++ (sur le modele
de `selfplay_phase_bench`) appelle `evaluate_batch` sur des tenseurs aleatoires de taille
constante, lots 1, 2, 4, 8, 16, 32, 64, 128, 256, avec echauffement et repetitions.
Rapporter millisecondes par appel et par position, GPU et CPU. Reponse attendue : si le
cout par position chute nettement entre 8 et 128, alors la production continue a du
mouvement et la cible de lot est connue. S'il stagne, le GPU est deja a son regime et le
seul levier restant est la reduction de M.

### T1b : forme de lot fixe contre lots partiels

Le test le plus rentable, et le plus simple. Dans le banc de recherche, forcer chaque
appel a evaluer exactement 8 positions (dupliquer une position quand la vague en a moins)
avec des buffers reutilises, puis comparer le temps evaluateur par appel et par position
au regime actuel de taille variable. Si on retombe vers les 3.0 ms du banc self-play,
c'est un facteur 3 a 5 pour un changement local. Cela repond aussi a la question de
savoir si l'essentiel de l'ecart vient du changement de forme ou des allocations.

### T2 : profiler l'appel ONNX

Sous-decouper la phase evaluateur en `session.Run`, copies de sortie et softmax, et
repeter la mesure a forme fixe puis a forme variable. Cela dit quelle part de M est CPU,
quelle part est GPU, et quelle part de l'ecart vient du changement de forme.

### T3 : balayage de divergence

Balayer amplitude de virtual loss {1, 2, 3} x c_puct {1.0, 1.4, 2.0, 3.0} x coefficient
FPU {0.2, 0.3, 0.5} a 700 simulations, en mesurant remplissage, sims/s et temps
evaluateur. Chaque configuration gagnante en debit passe ensuite le banc de puzzles, car
ces parametres changent la recherche. Il faut d'abord rendre ces constantes
parametrables.

### T4 : cas chaud

Mesurer une sequence de coups avec reutilisation d'arbre (`step_analysis` puis
`update_root`), et reporter la fraction de simulations qui evite le reseau. C'est ce
chiffre qui determine l'ampleur reelle du gain en partie.

### T5 : production continue, seulement apres T1

Si T1 montre que des lots de 32 a 128 sont beaucoup plus efficaces par position, alors la
file d'inference continue devient prioritaire. La mesurer sur un microbenchmark de file
avant de toucher a la concurrence de l'arbre.

### T6 : quantification et fusion

Preparer un export FP16 du meme modele et le comparer au FP32 sur T1 et T2, puis sur le
banc de puzzles pour la non-inferiorite. Mesurer separement l'apport de TensorRT EP si la
dependance est acceptable. A ne lancer qu'apres T1b, pour ne pas optimiser le reseau alors
que l'appel lui-meme est le gisement principal.

## 10. Questions en suspens

1. Pourquoi le banc self-play evalue 8 positions en 3.0 ms alors que la recherche en paie
   10 a 14 a remplissage comparable. Forme de lot, allocations, ou autre. T1b et T2
   doivent trancher.
2. Pourquoi M varie d'un facteur deux entre positions (0.9 a 1.9 ms) alors que le reseau
   est le meme. Bruit de session, ou comportement dependant des donnees du couple
   ONNX Runtime et GPU.
3. Le plafond de x1.6 a x3.4 tient-il apres assainissement des appels et reduction de M.
4. Quelle part du gain se transmet en partie reelle avec un arbre chaud et un fort taux de
   hits de table.
5. Le gain de debit justifie-t-il le cout de complexite de la production continue, sachant
   qu'il ne depasse pas un facteur faible.
6. Faut-il explorer la piste reseau (FP16, TensorRT) avant la piste ordonnancement, ou
   l'inverse. La reponse depend de T1b et T2.

## 11. Plafond realiste

Etat actuel : 10 a 14 ms par appel, 1.4 a 2 ms par position, evaluateur a 96 a 98 % du
temps de recherche.

- Meme reseau, meme GPU, autre harnais : 3.0 ms par appel de 8, soit 0.38 ms par position.
  Le premier facteur 3 a 5 est donc dans l'appel, pas dans le reseau.
- Apres assainissement des appels, un cout de 0.1 a 0.3 ms par position est envisageable
  avec FP16 et fusion des couches.
- Ce qui borne : le lot d'un seul arbre plafonne vers 7, donc les frais fixes restent
  amortis sur 7 positions, pas sur 128.
- Gain total realiste de ce chantier : x3 a x6, pas x30.

Reponse courte : non, 10 ms par inference n'est pas proche du plafond. Le gros gain est
immediat et peu risque (forme de lot fixe, buffers reutilises), le gain reseau vient
apres.

## 12. Fichiers de reference

- Donnees : `out/multicore/{phases,after,queue,avant-r*,apres-r*}.json`.
- Code : `src/mcts.cpp` (UCB, FPU, selection), `src/mcts_wave.cpp` (collecteur,
  collisions, budget de tentatives), `src/mcts_batch.cpp` (boucle mono, fermeture sur
  collision), `src/mcts_reservation.cpp` (virtual loss), `src/onnx_evaluator.cpp`
  (appel ONNX et softmax), `src/evaluation_cache.cpp` (table).
- Specs : `2026-09-14-search-bench-batching-gpu.md`,
  `2026-09-11-search-bench-resultats.md`, `2026-09-11-mcts-batching-design.md`,
  `2026-09-14-mcts-multicore-design.md`.
