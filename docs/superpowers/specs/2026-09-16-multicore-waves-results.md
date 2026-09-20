# Recherche MCTS multicoeur par vagues : mesures et decision

Date : 2026-09-19

Statut : tache 10 du plan `2026-09-16-mcts-multicore-waves.md`, decision d'activation prise.

Modele : `2026_04_23_23h25_iter316_unsupervised.onnx`, sha256 `20a19f82e170`.

Banc de puzzles sha256 `a60b584abc25`.

Materiel : RTX PRO 2000 Blackwell Laptop 8 Gio, driver 595.95, 16 coeurs logiques
Intel64 Family 6 Model 197. ONNX Runtime 1.24.3, provider CUDA.

Reglages : politique TT h0, table 8192, lot 8, c_puct 1.4, 700 simulations,
trois positions de reference (ouverture, milieu, finale), un seul processus GPU.
Les valeurs absolues varient de 15 a 30 pour cent entre sessions (horloge et
thermique) ; seuls les rapports mesures dans une meme session sont utilises ici.

Analyses de phases et pistes d'optimisation : voir
`2026-09-19-cout-calcul-cpu-gpu.md`, qui agrege les memes campagnes.

## 1. Debit, pool chaud

Campagne `out/multicore/after.json` : 5 passages de 6 repetitions, ordre des
workers alterne a chaque passage, pool chaud, 30 observations par configuration.
Medianes de simulations par seconde, chemin `step_analysis` :

| position | w1 | w2 | w4 | w8 | gain w8 |
|---|---|---|---|---|---|
| ouverture | 733.5 | 812.3 | 835.0 | 846.8 | +15.4 % |
| milieu | 328.7 | 340.7 | 356.1 | 348.8 | +6.1 % |
| finale | 462.2 | 583.8 | 579.2 | 565.1 | +22.3 % |

Le gain plafonne des 2 workers. `mcts_search` donne les memes ordres.

Campagne de queue dediee `out/multicore/queue.json` : 50 passages interleaved,
workers 1/2/8, 50 observations par configuration. Le critere d'acceptation porte
sur le p95 de latence, qui correspond au p5 de debit :

| position | chemin | w8 mediane | w8 p95 latence |
|---|---|---|---|
| finale | step_analysis | x1.212 | x0.882 |
| milieu | step_analysis | x1.069 | x1.045 |
| ouverture | step_analysis | x1.071 | x0.926 |
| finale | mcts_search | x1.255 | x0.832 |
| milieu | mcts_search | x1.081 | x1.018 |
| ouverture | mcts_search | x1.083 | x0.971 |

Aucune hausse de p95 de latence au dela de +5 %, et la queue s'ameliore meme en
finale. w2 passe partout sauf milieu step (+6.4 %, marginal) ; w8 est retenu.

## 2. Regime reel de la boucle UCI

`uci.py` decoupe une recherche en appels de `BATCH_SIZE = 64` simulations, pas
en un appel unique de 700. Diagnostic `step_analysis` repete, 704 simulations,
arbre et pool persistants, 3 repetitions :

| tranche | position | w1 sims/s | w8 sims/s | collisions w8 |
|---|---|---|---|---|
| 64 | ouverture | 799.1 | 912.5 | 729 a 820 |
| 64 | milieu | 373.1 | 462.8 | 3007 a 3307 |
| 64 | finale | 582.5 | 580.0 | 1320 a 1592 |
| 20 | ouverture | 503.9 | 572.4 | 607 a 649 |
| 20 | milieu | 244.5 | 256.1 | 3225 a 3309 |
| 20 | finale | 319.1 | 405.5 | 1125 a 1443 |

Aux deux tranches du bot, w8 gagne de +5 a +27 % sauf finale a 64 (un tirage a
417 sims/s contre 633 et 691, bruit). A tranche de 8 simulations, en revanche,
w8 perd 19 % : les arbres naissants font 970 collisions pour 800 simulations
(693 contre 857 sims/s). Le protocole a temps egal du banc puzzle utilise des
tranches de 8 et penalise donc w8 ; ce n'est pas le regime du bot.

## 3. Repartition par phase

Chronometrages `out/multicore/phases.json` : l'evaluateur represente 96.6 a
98.7 % du temps mur dans toutes les configurations. L'attente de barriere vaut
1.5 a 2.3 % en w8 ; selection, tenseur, expansion, backup et TT additionnent
20 a 25 ms sur 700 simulations. Le remplissage passe de 6.1 a 6.7 (finale) et de
6.9 a 7.2 (ouverture), ce qui reduit le nombre d'appels GPU physiques.

Le gain des vagues vient donc surtout du remplissage et de la preparation CPU
parallelisee ; le GPU reste le goulet. Voir `2026-09-19-cout-calcul-cpu-gpu.md`
pour le plafond realiste (x3 a x6 apres assainissement de l'appel a
l'evaluateur).

## 4. Absence de regression mono-worker

Comparaison interleaved ancien moteur (`da29e7b`, pyd 475 648 octets) contre
nouveau workers1 : 3 rounds alternes, 3 passages de 700 simulations chacun.
L'ancien fluctue de 700 a 1200 sims/s selon l'etat d'horloge (round 3 en boost),
le nouveau reste entre 700 et 820. Aucune regression reproductible. Les
compteurs et les distributions de visites sont identiques.

## 5. Self-play partage

`selfplay_phase_bench`, 8 plateaux, 100 vagues, 30 repetitions, compare aux
references de la tache 2 :

| variante | collection | traitement | compteurs |
|---|---|---|---|
| fake | 0.738 ms a 1.321 ms (+0.58 ms pour 100 vagues) | 4.390 a 4.508 ms | identiques |
| GPU | 1.928 a 2.003 ms | 340.97 a 305.04 ms (-10.5 %) | identiques |

Le surcout CPU des atomiques, du cache et des reservations est de 0.6 ms par
100 vagues sur le faux reseau, negligeable face aux 305 ms de reseau en GPU.
Les visites de reference sont identiques au bit pres.

## 6. Qualite

- Prefiltre 500 puzzles : w1 369, w8 371, delta +0.4 point,
  IC95 [-0.6 ; +1.6].
- Campagne complete 2500 puzzles : w1 1927 (77.08 %), w8 1920 (76.80 %),
  delta -0.28 point, IC95 [-0.8 ; +0.24], 24 paires perdues contre 17 gagnees,
  McNemar p = 0.35. Verdict : non-inferiorite, marge de -1 point respectee.
  La borne basse est proche du seuil ; la repetition du candidat prescrite par
  le plan a ete ecartee pour duree (44 minutes) par le proprietaire.
- Mesure a temps egal, 500 puzzles, budget 1.5 s (mediane observee des 700
  simulations mono-worker) : w1 366, w8 364, delta -0.4 point,
  IC95 [-1.6 ; +0.8], verdict indetermine. Depassement median de 8.8 ms et p95
  de 51 ms cote candidat. w8 termine 472 simulations medianes contre 728, effet
  des tranches de 8 decrit en section 2 ; le taux de resolution reste a egalite.

## 7. Decision

`MCTS_WORKER_COUNT = 8` est active dans `uci.py`, a cote de
`MCTS_BATCH_SIZE = 8`, et passe en cinquieme argument de `step_analysis`. Tous
les criteres du plan sont franchis : debit en hausse reproductible, p95 de
latence sous +5 %, invariants verifies, non-inferiorite qualite a 2500 puzzles.

Reserves consignees :

- le taux de hits de table en partie reelle a ete mesure le 2026-09-19 (cas chaud,
  `out/multicore/hot-tree.json`) : le taux monte de 8 a 81 % mais les appels reseau
  restent a 700 pour 700 simulations, car un hit de table materialise les enfants et la
  descente continue vers une feuille reseau. La reserve Amdahl est donc sans objet ; voir
  `2026-09-19-cout-calcul-cpu-gpu.md`, section 8.4 ;
- chaque recherche paie une inference d'expansion de racine, incluse dans les
  mesures ;
- le bot utilise une table de 4 000 000 d'entrees (environ 4.16 Gio), les
  mesures sont a 8192 ;
- la campagne 2500 n'a pas ete repetee ; l'incertitude residuelle de qualite va
  jusqu'a -0.8 point ;
- le test a temps egal avec tranches de 8 penalise w8 ; le regime reel du bot
  (tranches de 64) est plus favorable.

## 8. Tests executes

- 14/14 CTest, 250/250 pytest a la fin de la tache 9.
- Stress multicœur de la tache 8 : 1800 recherches, 115200 simulations.
- Transmission du cinquieme argument UCI verifiee par test unitaire.
- Smoke UCI en sous-processus : `uci`, `isready`, `position startpos`,
  `go nodes`, `stop`, `quit`, bestmove legal.
