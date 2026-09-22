# Backlog technique

Constats issus de la revue de code du 2026-08-07 et de la campagne perft, classés par
rapport gain sur effort. Rien ici n'est en cours.

## Correctifs MCTS et rebalayage de virtual loss

La revue statique du 2026-09-21 a produit dix correctifs et leurs tests
(`superpowers/plans/2026-09-21-audit-correctifs-mcts.md`). Deux constats commandent la
suite :

- R1, réservation pondérée : dès que l'amplitude du virtual loss dépasse 1, chaque
  descente laissait un résidu permanent de +1 sur les noeuds du chemin. Le balayage de
  divergence du 2026-09-19 est donc non interprétable dans les deux sens, et R1
  conditionne tout réglage futur de la recherche. C'est aussi la cause probable du rejet
  de `virtual_loss = 2`, qui mérite une campagne propre.
- R9, fin de lot self-play : `generate_games` démarre C + N - 1 parties et abandonne les
  C - 1 dernières à la destruction du gestionnaire, avec un biais plausible vers les
  parties courtes dans les données collectées. À quantifier avant de conclure.

**État au 2026-09-22.** Les dix lots de l'audit sont faits et validés sur la branche
`codex/audit-mcts` ; les tests restants sont des campagnes GPU ou des points documentés
comme sans objet.

**Rebalayage : fait, `virtual_loss = 2` activé.** Quatre invocations A, B, B, A en
alternance : +9 à +20 % de débit sur les trois positions, p95 de latence en baisse de 13 à
20 %, collisions en milieu de 3284 à 2054. Barrière qualité franchie sur 500 puis 2500
puzzles : 1926 contre 1928, IC95 [-0,4 ; +0,56], McNemar p = 0,87, réseau seul identique au
bit près. Le rejet de septembre venait bien de la fuite R1. Le réglage est propagé au bot,
au tournoi, à la GUI et à l'ancrage Stockfish. Le self-play n'est pas concerné :
chaque arbre y est descendu une seule fois par vague, donc l'amplitude y est
inerte. Rapport complet :
`superpowers/specs/2026-09-22-rebalayage-virtual-loss-resultats.md`.

**Pools self-play, mesure préliminaire.** À 100/20 simulations, la matrice (128, 256),
(256, 512) et (512, 512) donne 192,9, 231,2 et 254,4 plies nouveaux par seconde, avec
départs = fins = total partout. Le plus grand pool est le plus rapide à budget réduit, ce
qui va contre l'intuition du genou à 128. Une confirmation à 700/100 est nécessaire avant
de changer la taille de production, qui reste 256.

- [x] Traiter R2 puis R1 du plan de correctifs, prérequis du rebalayage.
- [x] Exécuter le rebalayage de virtual loss après R1 : comparaison A/B interleaved,
      préfiltre 500 puis campagne 2500, décision et propagation :
      `superpowers/plans/2026-09-21-rebalayage-virtual-loss.md`.
- [x] Mesurer et décider R9-B (fin de génération self-play) après le diagnostic R9-A :
      compteurs, génération finie et rapport honnête faits ; la matrice de pools est
      mesurée à budget réduit, la confirmation à 700/100 reste à faire :
      `superpowers/plans/2026-09-21-fin-de-lot-self-play.md`.

## Banc de puzzles batché sur GPU (inspiration self-play)

Le banc actuel tourne sur 16 processus CPU, chaque puzzle cherche seul avec un lot de 8,
et le GPU reste inutilisé : 2500 puzzles prennent environ 12 minutes, proche du plafond
CPU de 16 cœurs. Le self-play montre la voie : une feuille par partie, un seul appel
réseau pour 256 positions. Le coût par position tombe de 0,41 à 0,65 ms au lot de 8 à
0,10 à 0,16 ms aux lots de 128 à 256, soit un facteur 2 à 3 attendu sur la campagne.

- [ ] Prototype Python : 256 instances MCTS indépendantes, une par puzzle (table propre,
      pas de contamination entre puzzles), un évaluateur GPU partagé, collecte d'une
      feuille par instance via `advance_to_leaf`, un `evaluate_batch` commun, puis
      `expand_and_backup`.
- [ ] Petite surcharge C++ nécessaire : lire les statistiques d'une racine quelconque
      (`get_analysis_results(root)`), la méthode actuelle ne lisant que la racine
      d'analyse.
- [ ] Mesurer sur 500 puzzles contre le mode 16 processus, puis décider d'un pilote C++
      propre si le facteur 2 à 3 se confirme.
- [ ] Contrainte : un mode batché produit une nouvelle référence, à ne pas mélanger avec
      les chiffres historiques, comme la règle 700 contre 800.

## Prochaine tâche décidée

**Banc de test sur puzzles.** Mesurer la qualité recherche + évaluation à nombre de
simulations fixé, sur des puzzles à solution connue. Le perft ne valide que la
génération de coups ; sans cet instrument, aucun changement de MCTS ou de réseau n'est
mesurable autrement que par un tournoi de 16 parties, dont les barres d'erreur
avalent tout ce qui fait moins de 100 Elo.

Contrainte de conception : **les puzzles doivent être présentés avec un historique.**
Un banc sur FEN nues mesurerait la performance gonflée par le raccourci décrit plus bas,
pas la force réelle.

Les données sont prêtes : `data/puzzles_bench.txt`, 5000 puzzles avec leur historique réel,
committé et disjoint de l'entraînement par construction. Voir
`superpowers/specs/2026-08-07-puzzle-pipeline-resultats.md`.

**Notes de conception détaillées : `superpowers/specs/2026-08-08-puzzle-bench-notes.md`.**
Onze subtilités identifiées, dont la contamination des mesures par la table de
transposition (indexée sur le seul Zobrist, donc un hit d'un puzzle précédent peut renvoyer
une value calculée sous un autre historique : il faut un `MCTS` neuf par puzzle), et le fait
que le coût de la mesure est dominé par le batch de 1, donc que le banc deviendra 10 à 40
fois plus rapide après l'entrée §1.

## 1. Batcher la recherche côté UCI

Le plus gros écart du projet. Il existe deux points d'entrée vers le GPU :

- `selfplay_manager.cpp:88` : `evaluate_batch(..., 512)`, batché
- `mcts.cpp:176` : `evaluate(...)` puis `evaluate_batch(..., 1)`, **batch de 1**

Le second est celui qu'emprunte `step_analysis`, donc le bot en partie fait une
inférence GPU par simulation. Le `BATCH_SIZE = 20` de `uci.py` n'est que la granularité
de la boucle Python, pas un batch GPU.

La machinerie existe déjà (`advance_to_leaf`, `expand_and_backup`), écrite pour le
self-play. Il manque le virtual loss pour que plusieurs descentes simultanées dans un
même arbre ne convergent pas vers la même feuille.

Gain attendu : un ordre de grandeur sur le nombre de simulations à temps constant.

**Notes de conception détaillées : `superpowers/specs/2026-08-08-uci-batching-notes.md`.**
Elles contiennent la dérivation du signe du virtual loss sur ce code précis, la structure de
la boucle, et six pièges vérifiés dans le code, dont le plus sérieux : `select_leaf` fait de
l'expansion paresseuse depuis la TT et continue à descendre, donc une feuille collectée peut
recevoir des enfants pendant la même collecte et `expand_and_backup` en créerait un second
jeu. Il faut un drapeau `is_pending` sur `MCTSNode`.

**Fait.** Le batching par virtual loss est entré le 2026-09-15 (lot 8 validé), puis
la recherche multicœur par vagues le 2026-09-19. Mesures sur GPU, 700 simulations,
h0 : médiane de +8 à +26 % selon la position, p95 de latence sous +5 %, qualité
non-inférieure sur 2500 puzzles, self-play sans régression. `uci.py` utilise
`MCTS_WORKER_COUNT = 8` depuis cette date. Rapport complet :
`superpowers/specs/2026-09-16-multicore-waves-results.md`. La production continue
n'est pas décidée et attend le test T1b de
`superpowers/specs/2026-09-19-cout-calcul-cpu-gpu.md`.

## 2. Corriger la clé de la table de transposition

L'entrée du réseau contient 8 plies d'historique, les plans de répétition, le compteur
des 50 coups et le flag amnésie. Le hash Zobrist ne contient **rien de tout ça**. Un hit
peut donc renvoyer une policy et une value calculées sous un contexte différent, ce qui
est du bruit permanent sur toutes les évaluations.

**Design détaillé : `superpowers/specs/2026-09-15-tt-evaluation-key-design.md`.** Le
compteur des 50 coups sera exact. La profondeur historique ne sera pas fixée
intuitivement : 0 et 1 sont les candidates principales, 3 et 7 servent de témoins. Le
benchmark mesurera séparément la perte de hits, les appels réseau, le débit, l'accord
des coups et la conversion de la finale tour contre roi.

**Mesuré le 2026-09-15 :** les quatre campagnes de 2 500 puzzles et les 48
parties de finales sont terminées. `h0` est retenu comme défaut selon les
critères prévus. Les résultats et leurs limites sont dans
`superpowers/specs/2026-09-15-tt-key-benchmark-results.md`.

Au passage, `TTEntry` réserve la place pour 128 coups (`std::array<std::pair<int,float>,
128>`), soit 1040 octets par entrée, alors qu'une position en a environ 35 d'utiles.
Avec `tt_size=4_000_000` en self-play, cela fait 4,16 Go alloués et initialisés à zéro.
En passant à `{uint16 move, uint16 prob}` et une capacité de 64, on tombe à environ
270 octets : 4 fois moins de RAM et un probe qui tient dans 4 lignes de cache au lieu
de 16.

## 3. Race dans l'UCI (une ligne)

`parse_position` (`uci.py:97`) modifie `self.board` et appelle `mcts.update_root()` sans
arrêter le thread de recherche. Pendant un ponder, `step_analysis` descend et remonte
l'arbre sur ce même plateau. Les deux opérations ne sont pas atomiques ensemble : entre
le `update_root` et le `move_piece`, une simulation peut passer et descendre dans un
arbre désynchronisé du plateau.

Candidat idéal pour un `Problème lors de l'application du coup dans select_leaf`
aléatoire ou un bestmove illégal en partie. Correctif : `self.stop_search()` en tête de
`parse_position`.

Plus mineur : `MCTS::get_root_q()` (`mcts.cpp:366`) lit l'arbre sans prendre le mutex.

## 4. Confondant historique / tactique — DÉCIDÉ : corriger les données, pas l'architecture

**Décision du 2026-08-07, après révision : on garde les 119 plans.** Le confondant se
corrige à la source, en fournissant aux positions de puzzles leur historique réel.

### Le problème

Les positions de puzzles n'ont pas d'historique, les positions de partie en ont. La
présence d'historique est donc **corrélée au label** « position tranchante », et le réseau
peut lire les 96 plans d'historique comme un drapeau plutôt que d'apprendre la tactique.

Effet mesuré par Mathieu dans Nibbler : sur une position de puzzle présentée sans
historique, le prior de recherche du coup tactique est nettement plus haut, donc le coup
est trouvé. Avec historique, il ne l'est pas. Le raccourci est confirmé empiriquement.

Le mode amnésie à 5 % cassait la corrélation parfaite mais pas la corrélation, et surtout
les leçons tactiques restaient stockées dans une région de l'espace d'entrée que la partie
réelle ne visite jamais. C'est ça qui bloquait le transfert, et c'est pourquoi entraîner
plus longtemps n'aurait pas suffi.

### La correction retenue

Le problème n'était pas que les puzzles soient des positions inhabituelles, c'était qu'ils
soient **structurellement identifiables**. Une position de puzzle avec 8 plies réels et
une position de partie avec 8 plies réels ne présentent aucune différence de structure,
donc aucun drapeau à lire.

Les puzzles Lichess viennent de vraies parties et le CSV porte le champ `GameUrl`. L'API
d'export permet donc de récupérer la séquence de coups menant à chaque position.

Faisabilité vérifiée le 2026-08-07 : `POST /api/games/export/_ids` accepte **300 IDs par
requête**, corps en texte brut séparé par des virgules, réponse en flux PGN ou ndjson.
100 000 puzzles représentent donc 334 requêtes, soit quelques minutes en respectant la
consigne d'une requête à la fois.

Robustesse : **ne pas se fier au numéro de ply du `GameUrl`.** Rejouer les coups de la
partie jusqu'à ce que la position corresponde à la FEN du puzzle. Le procédé s'auto-valide
et tout puzzle dont aucun ply ne correspond est écarté.

### Travail induit

- `extract_lichess_puzzle.py` : conserver le `GameUrl`, récupérer les parties par lots de
  300, et écrire pour chaque puzzle la séquence de coups plutôt qu'une FEN nue.
- Côté C++ : charger la position initiale puis rejouer les coups, pour que
  `m_boardHistory` contienne les 8 entrées attendues. Le chargement actuel par `loadFEN`
  n'en produit qu'une.
- Le mode amnésie est **conservé**, taux ramené de 5 % à 1 %. Son but déclaré disparaît,
  mais il entraîne le réseau à fonctionner sans historique, cas qui subsiste pour les FEN
  collées à la main dans une GUI. Une fois les puzzles porteurs d'un historique réel, il ne
  corrèle plus avec la tacticité et devient de l'augmentation pure. Sa raison d'être doit
  être redocumentée dans le code, le commentaire actuel devenant faux.

Effet secondaire bienvenu : cela referme aussi le trou de prise en passant décrit
ci-dessous, puisque les positions de puzzles auront une position précédente réelle.

### Résidu honnête

Même avec un historique réel, il subsiste une différence de **style** entre un historique
issu d'une partie humaine et un historique de self-play. Le réseau pourrait en principe
l'exploiter. C'est un signal statistique faible et non un drapeau structurel, et il
s'appliquerait aussi à la position elle-même : supprimer l'historique ne le corrigerait
pas. Ne différencie donc pas les deux solutions.

### Trou de prise en passant, refermé au passage

`getAlphaZeroTensor` n'a aucun plan de prise en passant : l'information n'existe que par
comparaison de deux positions consécutives. Pour une position chargée par `loadFEN`,
`m_boardHistory` ne contient qu'une entrée, donc **sur toutes les positions de puzzles une
prise en passant disponible est aujourd'hui invisible pour le réseau**, alors que le moteur
la connaît et la génère dans les coups légaux.

À noter, le papier AlphaZero ne comporte pas non plus de plan de prise en passant : les
119 plans actuels correspondent exactement à sa spécification. Le trou est hérité, pas
introduit.

Rejouer les coups referme le cas des puzzles. Restent les FEN nues collées à la main dans
une GUI, cas mineur et hors périmètre.

### Pourquoi c'est un confondant et non un décalage de distribution

La distinction a guidé toute l'analyse. Un décalage de distribution se corrige en
élargissant la distribution, ce que faisait le mode amnésie. Mais ici la présence
d'historique est corrélée au label, donc le réseau n'a pas besoin de comprendre la
tactique : il lui suffit de lire les 96 plans comme un drapeau.

Deux façons de tuer un confondant : égaliser la covariable entre les groupes, ou supprimer
la covariable. La correction retenue égalise (les deux groupes ont désormais un historique
réel). Les variantes à 1 ou 2 positions supprimaient.

### Options écartées

1. **Réduction à 2 positions (35 plans) ou 1 position (22 plans).** Abandonnées **pour ce
   motif** : le confondant est réglé sans elles. Elles restaient justifiées par des gains
   annexes réels et sans rapport, conservés en réserve plus bas.
2. **Jeter les 8 premiers plies des parties issues de puzzles.** Trois lignes, mais on
   perdrait la recherche à 4000 simulations sur la position de puzzle elle-même, qui est
   tout l'intérêt de l'injection.
3. **Fabriquer les positions tactiques depuis le corpus PGN via Stockfish.** Donnerait un
   historique réel par construction et sans réseau, mais **perdrait le rating Lichess**,
   calibré sur des millions de tentatives humaines. Rédhibitoire pour un banc stratifié par
   difficulté. Reste une piste valable pour enrichir les données d'entraînement.
4. **Historique synthétique par analyse retrograde**, ou variante répétant la position
   courante dans les huit créneaux. Créent une troisième distribution, « historique
   fabriqué », et remplacent un confondant par un autre, moins visible.

### En réserve : réduction du nombre de plans comme optimisation

Sans lien avec le confondant, donc à décider séparément. Gains : tenseurs 3 à 5 fois plus
petits (replay buffer, RAM, copies), construction du tenseur d'autant moins chère,
première convolution réduite. Coût : un fine-tuning avec pic de loss et perte d'Elo
temporaire. Mesurable avec le banc de puzzles.

Éléments d'analyse à conserver si ce chantier revient :

- `119` apparaît dans **19 occurrences fonctionnelles sur 8 fichiers** (`bindings.cpp`,
  `chessboard.cpp`, `mcts.cpp`, `onnx_evaluator.cpp`, `selfplay_manager.cpp/hpp`,
  `lib.py`, `model.py`, `puzzle_bench.py`), sans définition partagée. Une migration
  partielle ne casserait pas la compilation mais ferait lire de la mémoire arbitraire à
  ONNX Runtime via `expected_elements` (`onnx_evaluator.cpp:46`). **Introduire une
  constante unique exposée par les bindings avant toute migration.**
- Seul `conv_input.weight` change de forme, donc environ 96 % des poids se copieraient à
  l'identique. `transfer_weights.py` devrait trancher le tenseur au lieu de l'ignorer sur
  non-correspondance de forme.
- Le replay buffer existant serait réutilisable par simple sélection de canaux, sans
  régénérer une seule partie.
- Les plans `total_moves` et `no_progress` sont ceux que le bug `astype(np.uint8)` mettait
  à zéro dans le dataset supervisé (voir §9). Leur valeur réelle n'a jamais été éprouvée,
  et ils seraient les premiers candidats à la suppression.

## 5. Alléger la représentation du plateau

`Square` stocke ses propres `file` et `rank`, redondants avec son index, et `Piece`
porte un `m_value` (matériel 1/3/5/9) inutilisé dans un moteur zero-knowledge. Un
`Square` pèse donc 20 octets pour 4 bits d'information utile, et un plateau 1280 octets.

`updateHistory` (`chessboard.cpp:864`) pousse une copie complète du plateau à chaque
coup, et `undoMove` la recopie en sens inverse. À 700 simulations et ~10 plies de
profondeur, cela fait environ 18 Mo de memcpy par coup joué, pour un historique dont
seules les 8 dernières entrées servent au tenseur.

Un `Move` contient deux `Square` complets plus un `Piece`, soit environ 64 octets pour
ce qui tient dans 16 bits. Et `movePiece` alloue deux `std::vector` sur le tas à chaque
appel, `getLegalMoveIndices` trois de plus.

**Attention à la priorisation :** ces coûts sont réels mais le CPU n'est pas le facteur
limitant. À 1,77 M nœuds/s, la génération de coups coûte environ 0,6 µs par nœud contre
plusieurs millisecondes pour une inférence. Ce chantier se justifie pour la qualité du
code et comme préalable aux bitboards, pas pour l'Elo.

## 6. Séparer validation et exécution dans movePiece

`movePiece` valide le coup en regénérant les coups légaux de la case de départ, puis
l'applique. Le MCTS l'appelle sur des coups **qu'il vient lui-même de générer**, donc
légaux par construction, et paie cette validation à chaque nœud.

Séparer un `makeMove` rapide et confiant d'un `isLegal` explicite appelé seulement par
l'UCI et la GUI supprimerait ce coût. Cela rendrait aussi le nommage évident : aucun nom
ne convient actuellement parce que la fonction fait deux choses.

## 7. Bruit de Dirichlet appliqué de façon incohérente

Trois chemins ajoutent du bruit à la racine (`selfplay_manager.cpp:81`, `:100`, `:205`)
et un chemin n'en ajoute aucun : quand `play_best_move` re-crée une racine et que
`expand_node_single` tombe sur un hit TT, la fonction retourne sans créer d'enfants,
ceux-ci sont créés paresseusement plus tard dans `select_leaf`, et le bruit n'est jamais
appliqué.

Une fraction non contrôlée des coups de self-play est donc cherchée sans exploration à
la racine, ce qui rend la distribution des données d'entraînement inhomogène.

## 8. Copies inutiles au retour du self-play

`return m_finished_games;` (`selfplay_manager.cpp:401`) copie le membre au lieu de le
déplacer. Puis pybind convertit `std::vector<GameResult>` en liste Python en recopiant
chaque `GameResult`. Puis `convert_game_results` (`lib.py:514`) construit une liste
Python d'arrays numpy par position, et `np.array([...])` recopie encore.

Avec 512 parties d'environ 700 Ko, cela fait plusieurs centaines de Mo copiés trois
fois, sur une machine qui a déjà 4 Go de table de transposition.

## 9. Divers, à faible coût

- `checkEnPassant()` (`chessboard.cpp:546`) positionne le drapeau sur toute poussée
  double, même quand aucune capture n'est possible. Deux positions identiques hashent
  donc différemment, ce qui fait rater des répétitions triples et pollue la TT.
- Le roque ne vérifie pas la présence de la tour (`chessboard.cpp:1802`), seulement le
  drapeau de droit et les cases vides. Cohérent en jeu normal, mais un `loadFEN` avec
  des droits incohérents fabriquerait une tour fantôme.
- `astype(np.uint8)` dans `convert_pgn_to_binary.py:49` tronque à zéro les plans 113
  (compteur de coups) et 118 (règle des 50 coups), qui sont des flottants dans [0,1].
  Le préentraînement supervisé a donc vu ces deux plans constants à zéro alors que le
  self-play leur envoie des valeurs réelles. À corriger si une passe supervisée est
  refaite.
- `random.shuffle(files)` dans `sharded_dataset.py:30` a lieu **avant** le découpage par
  worker, et chaque worker a une seed différente : le découpage n'est pas une partition.
  Certains shards sont lus par plusieurs workers, d'autres jamais.
- `ONNXEvaluator` (`onnx_evaluator.cpp:20`) élargit le chemin avec
  `std::wstring(s.begin(), s.end())`, ce qui casse sur tout chemin non-ASCII.
- **Softmax légal, optimisation indépendante du multicœur.** `ONNXEvaluator`
  applique actuellement le softmax aux 4672 logits, puis le MCTS renormalise les
  ~35 coups légaux. Un softmax limité aux coups légaux est mathématiquement équivalent,
  hors petites différences d'arrondi, et évite l'essentiel de ce calcul. Ce chantier
  devra expliciter l'interface logits/probabilités pour ne pas casser les consommateurs
  du self-play, puis être mesuré isolément. Il n'est ni un prérequis ni un critère
  d'acceptation de la recherche multicœur.
- `inline int Piece::getZobristIndex()` (`piece.hpp:40`) utilise un qualificateur
  `Piece::` sur une définition interne à la classe : extension MSVC, refusée par GCC et
  Clang.
- Pas de limite mémoire sur l'arbre d'analyse. Chaque nœud étendu coûte environ 3 Ko
  (35 enfants, chacun un `make_unique` séparé), donc un `go infinite` long grossit sans
  borne. Une arène de nœuds serait un double gain, mémoire et localité.
- Garde manquante dans `play_best_move` (`selfplay_manager.cpp:150`) : si
  `children` est vide, `children.front()` est un comportement indéfini et `best_move`
  reste à -1, ce qui produit ensuite un accès `m_board[-1]`. Inatteignable aujourd'hui
  puisque les FEN de puzzles sont des mats en 1, jamais terminales.
- `TT_MAX_MOVES = 128` tronque silencieusement les positions ayant plus de coups
  légaux : les coups au-delà deviennent inatteignables dans la recherche.
- Environ 60 Mo de binaires suivis par git (`chess_engine.pdb` à 45 Mo, `onnxruntime.dll`
  à 14 Mo). La règle `*.dll` du `.gitignore` ne s'applique pas aux fichiers déjà indexés.
- `id name Lc0 Custom` dans le handshake UCI (`uci.py:64`).

## 10. Mesurer le désapprentissage du confondant historique / tactique

Contexte : jusqu'à l'itération 300 environ, les puzzles injectés en self-play
n'avaient pas d'historique, et le réseau a appris que « pas d'historique =
position tactique ». Le pipeline d'août corrige les données (puzzles rejoués
depuis la position initiale, donc historique réel), mais le modèle actuel
(iter436, avril) porte encore le raccourci, et aucune campagne n'a encore été
faite sur les données corrigées. Voir la section 4 pour l'analyse du
confondant lui-même.

Mesure du raccourci sur une position réelle (partie `O7KXpd5v`, après
11...Nf5, modèle iter436, 25 000 simulations) :

| passe | prior de Qxa8, seul coup gagnant | Q racine |
|---|---|---|
| FEN nu, sans historique | 39,0 % | +0,11 |
| historique réel | 0,05 % | -0,87 |

Facteur 800 sur le prior du coup solution, et une value fausse d'environ
1,3 pion dans le second cas. En partie, l'historique est toujours présent :
le modèle est donc bien plus faible tactiquement que ne le suggéraient les
tests en FEN nu.

Travail proposé : ajouter au banc de puzzles (`python_src/puzzle_bench.py`)
une seconde passe qui rejoue les mêmes positions avec un historique vidé, et
rapporter l'écart des métriques entre les deux passes (prior du coup solution,
coup choisi, accord avec la recherche). Un chiffre unique à suivre d'une
campagne à l'autre, qui doit converger vers zéro quand le raccourci est
désappris. Attention à ne pas conclure sur le seul prior : une position facile
peut le faire monter sans rapport avec le confondant, d'où la comparaison de
plusieurs métriques.

**Outil fait le 2026-09-22.** `puzzle_bench.py --comparer-historique` mesure
chaque puzzle deux fois, avec puis sans historique, dans le même travailleur.
Il écrit le CSV avec historique, un CSV `_sans`, leurs sidecars, et un rapport
qui ouvre sur la section de comparaison appariée. Le chiffre à suivre est
l'écart médian du prior du coup solution (`avec - sans`), accompagné de la part
de visites, de la value, des taux de résolution avec McNemar apparié et de
l'accord des coups de recherche. La fonction pure `comparer_historique` vit
dans `bench_metrics.py` et se teste sans modèle.

**Campagne de référence faite le 2026-09-22, modèle iter436.** Sur 2500 puzzles :
prior médian du coup solution 0,288 avec historique contre 0,516 sans, écart
médian **-0,089** ; value -0,216 contre +0,930, écart **-1,064** ; résolution
1911 contre 2077, soit **+6,6 points sans historique**, McNemar 173 contre 339,
p = 3e-13. Le raccourci est donc bien présent et coûte cher en mesure. Le chiffre
à suivre doit converger vers zéro après réentraînement sur les données corrigées.
Campagne complète : `out/multicore/hist-2500.md`.

Le banc doit rester disjoint de l'entraînement, c'est le cas aujourd'hui
(0 recouvrement entre les 5 000 puzzles du banc et les 100 000 de
`training_data/puzzles_train.txt`).

## Phase 2 du perft, presque faite

Voir la section « Phasage » de `superpowers/specs/2026-08-07-perft-design.md` :

- fait : profondeurs calibrees figees par position et commande `calibre`
  (43 428 093 noeuds mesures, 27 s avec `--strict` sur la machine de developpement)
- fait : integration CTest, `chess_perft_bench` (instantane),
  `chess_perft_roundtrip` (instantane) et `chess_perft_calibre` (timeout 300 s),
  18 tests au total
- fait : decodage des index deplace dans `Chessboard::decodeMoveIndex`, expose au
  Python en `Chessboard.decode_move_index`, table Python supprimee ; la commande
  `roundtrip` controle `encodeMove(decodeMoveIndex(i)) == i` sur chaque coup legal
  de chaque noeud, mutation verifiee
- restant : nettoyer les chemins en dur `C:/Users/M47h1/...` de
  `python_src/train_supervised.py` (le backlog visait `src/main.cpp`, qui n'en a
  plus)
