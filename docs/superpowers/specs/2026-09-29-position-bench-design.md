# Banc externe de positions Stockfish : conception

Date : 2026-09-29
Révision : 2026-09-30, cinq minutes comme cible souple, sans arrêt automatique.
Statut : spec validée, prête pour le plan d'implémentation

Remplacer l'évaluation automatique par 16 parties contre un Stockfish bridé par
un banc fixe de positions externes, annotées une fois par Stockfish à pleine
puissance. Le banc doit produire un signal apparié, stable et interprétable en
environ cinq minutes. Il ne cherche pas à produire un Elo absolu.

## 1. Motivation et contraintes

L'ancre actuelle joue 16 parties avec 700 simulations par coup contre Stockfish
2600 limité à 200 000 noeuds. Elle cumule trois sources de bruit : le faible
nombre de parties, l'échantillonnage des premiers coups de LapZero et le moteur
adverse bridé. Elle mobilise en plus huit travailleurs Python et plusieurs
processus Stockfish.

Sur les 58 évaluations archivées dans `sim_results`, la durée médiane reconstruite
depuis les historiques W&B et les PGN est de 13,39 minutes. Les extrêmes observés
sont 11,17 et 17,60 minutes. La précision des horodatages PGN ajoute environ une
minute d'incertitude, sans changer la conclusion : le coût actuel est très
supérieur à la cible.

Le remplacement respecte les contraintes suivantes :

| Contrainte | Décision |
|---|---|
| Durée récurrente | cible d'environ 300 secondes, chargement et agrégation compris, sans limite dure |
| Cadence | toutes les 4 itérations par défaut, réglable par `--eval-every` |
| Comparabilité | mêmes positions, mêmes budgets et mêmes paramètres à chaque passage |
| Provenance principale | parties externes à LapZero et jamais utilisées pour l'entraînement |
| Historique | historique réel rejoué depuis la position initiale |
| Référence | centipions, mat et WDL Stockfish pré-calculés pour chaque coup légal |
| Score principal | regret d'espérance de résultat dérivé de la WDL, plus faible est meilleur |
| Diagnostics | value scalaire, MCTS et précision en centipions, sans score composite |
| Recherche | sous-banc fixe de 256 positions à 384 simulations |
| Elo | hors périmètre |

Les parties anciennes de LapZero ne participent pas au score principal. Elles
pourront former plus tard un banc `on_policy` séparé, sans jamais être agrégées au
score externe.

## 2. Provenance du banc

### 2.1 Source retenue

Le banc `v1` provient des archives mensuelles de broadcasts Lichess :

`https://database.lichess.org/#broadcasts`

Un broadcast est une retransmission de parties réellement jouées dans des
tournois, sur échiquier ou en ligne. Ce n'est pas nécessairement une partie jouée
sur le serveur Lichess. Les archives contiennent des opens, championnats, ligues,
compétitions juniors et tournois de haut niveau.

L'extraction commence avec août 2026, puis ajoute juillet et juin, dans cet ordre,
jusqu'à disposer d'au moins 20 000 parties valides. Chaque archive et son SHA-256
sont enregistrés dans le manifeste. Août 2026 est postérieur aux données du
préentraînement supervisé de février 2026.

La licence des broadcasts est CC BY-SA 4.0. Le manifeste conserve l'URL, le mois,
le nom de l'archive et l'attribution Lichess. Les PGN bruts ne sont pas committés.

### 2.2 Filtres de partie

Une partie candidate doit satisfaire tous les critères suivants :

- variante `Standard` ;
- départ depuis la position initiale standard, sans en-tête `SetUp` ou `FEN` ;
- Elo présent et supérieur ou égal à 2000 pour chacun des deux joueurs ;
- ligne principale PGN entièrement légale ;
- au moins 16 demi-coups joués ;
- identifiant de partie stable, ou identifiant synthétique SHA-256 du PGN si
  l'archive n'en fournit pas.

Les joueurs marqués `BOT` sont volontairement conservés. Une position provenant
d'un bot est traitée comme toute autre position. Le type humain ou bot est stocké
comme métadonnée et rapporté comme tranche diagnostique.

### 2.3 Absence de fuite d'entraînement

Sont explicitement interdits comme sources du score principal :

- le dump Lichess de février 2026 utilisé pour le préentraînement ;
- les 105 000 PGN Lichess déjà mis en cache dans le dépôt ;
- les PGN de grands maîtres utilisés pour le fine-tuning supervisé ;
- `data/puzzles_bench.txt`, dont les parties sources recoupent les données déjà
  consommées par le pipeline supervisé ;
- le replay buffer de self-play ;
- les anciennes évaluations contre Stockfish et les parties du bot LapZero.

Le manifeste `v1` porte la mention `training_forbidden: true`. Les scripts
d'entraînement ne découvrent jamais automatiquement ce répertoire et aucun chemin
du banc n'est ajouté aux chargeurs de données.

## 3. Extraction déterministe des positions

### 3.1 Préserver l'entrée réelle du réseau

Le réseau consomme 119 plans, dont huit positions d'historique. Une FEN isolée ne
suffit donc pas. Chaque enregistrement conserve :

- la position initiale ;
- tous les coups UCI depuis cette position jusqu'à la position cible ;
- la FEN cible, utilisée comme contrôle et non comme source de vérité ;
- le numéro de demi-coup ;
- les métadonnées de la partie et de sa source.

Au chargement, le banc rejoue toujours l'historique complet dans `ChessBoard`.
L'évaluation échoue si la FEN reconstruite diffère de la FEN stockée.

### 3.2 Phases

La phase est calculée avant annotation à partir du demi-coup et du matériel. Le
poids de phase vaut 1 pour chaque cavalier et fou, 2 pour chaque tour et 4 pour
chaque dame, avec un maximum initial de 24.

- ouverture : demi-coups 8 à 24 et poids de phase supérieur ou égal à 18 ;
- finale : poids de phase inférieur ou égal à 8 ;
- milieu : toutes les autres positions non terminales.

Pour chaque partie, l'extracteur peut proposer au plus une position par phase.
La position est choisie par le plus petit SHA-256 de
`game_id | ply | lapzero-position-bench-v1`, ce qui rend le choix indépendant de
l'ordre de lecture des archives.

### 3.3 Déduplication

Deux occurrences sont considérées identiques si le réseau et le générateur de
coups ne peuvent pas les distinguer. La clé est le SHA-256 de :

1. `board.get_alphazero_tensor()` sérialisé dans un format canonique ;
2. la liste triée des indices de coups légaux ;
3. le camp au trait et le compteur de demi-coups nécessaire à la règle des
   cinquante coups ;
4. le nombre de répétitions de la position courante, plafonné à trois et calculé
   sur l'historique complet.

Cette clé distingue les mêmes pièces obtenues avec des historiques, répétitions,
droits de roque ou prises en passant différents. Une simple FEN ou un Zobrist de la
position courante ne remplace pas cette clé.

## 4. Annotation Stockfish hors ligne

La fabrication du banc peut durer bien plus de cinq minutes. La cible de cinq
minutes ne concerne que l'évaluation récurrente d'un checkpoint, une fois les
annotations figées.

### 4.1 Configuration reproductible

Le constructeur reçoit explicitement le chemin de Stockfish. Le manifeste stocke
la sortie UCI `id name`, le SHA-256 du binaire et celui du réseau NNUE chargé.

Chaque recherche utilise :

- `Threads = 1` ;
- `Hash = 128` Mio ;
- `UCI_LimitStrength = false` ;
- `UCI_ShowWDL = true` ;
- aucune table Syzygy ;
- `Clear Hash` avant chaque recherche ;
- une limite en noeuds, jamais une limite en secondes.

Plusieurs processus Stockfish peuvent annoter des positions en parallèle. Une
recherche donnée reste monofilaire et indépendante, ce qui évite que l'ordre des
tâches ou l'ordonnancement des fils modifie les étiquettes.

### 4.2 Deux passes

La première passe analyse une position candidate à 50 000 noeuds, en simple PV.
Elle fournit une WDL de criblage. Le constructeur conserve une réserve de 12 500
positions couvrant les quotas de la section 5.

La seconde passe analyse séparément chaque coup légal de la réserve avec
`root_moves=[move]` et 200 000 noeuds. Le score est toujours converti du point de
vue du joueur au trait dans la position cible. Pour chaque coup, le fichier stocke :

- le coup UCI et son index de policy ;
- les nombres WDL Stockfish ;
- l'espérance `S = (win + 0.5 * draw) / (win + draw + loss)` ;
- le score en centipions ou le mat annoncé ;
- la profondeur atteinte et le nombre de noeuds effectivement visités.

La WDL n'est ni une sortie supplémentaire demandée à LapZero, ni une évaluation
indépendante de Stockfish. Elle est dérivée de l'évaluation Stockfish selon son
[modèle WDL officiel](https://github.com/official-stockfish/WDL_model), qui tient
notamment compte du matériel et normalise la valeur des centipions. À partir de
l'espérance `S`, la cible compatible avec la tête value scalaire de LapZero est :

`V_sf = 2 * S - 1 = (win - loss) / (win + draw + loss)`

`V_sf` appartient donc à `[-1, 1]`, comme la sortie `tanh` du réseau. Pour une
position et un ensemble de `root_moves` donnés, le classement des coups reste
principalement celui de l'évaluation en centipions. La conversion WDL change
surtout l'amplitude attribuée aux erreurs. Elle la borne et évite qu'une poignée
de scores très élevés ou de mats domine la moyenne.

Le meilleur score de la position est `S* = max S(coup)`. Le regret d'un coup est
`S* - S(coup)`. Tous les coups légaux sont annotés, car un modèle futur peut choisir
un coup absent d'un MultiPV court.

### 4.3 Audit de stabilité des étiquettes

Un échantillon déterministe de 5 % des positions finales est réanalysé à 400 000
noeuds par coup légal. Les étiquettes à 200 000 noeuds sont acceptées si :

- dans au moins 95 % des positions auditées, le meilleur coup à 200 000 noeuds a
  un regret inférieur ou égal à 0,02 selon l'analyse à 400 000 noeuds ;
- la moyenne des variations absolues de `S*` est inférieure ou égale à 0,01.

Si une condition échoue, toute la seconde passe est reconstruite à 400 000 noeuds
par coup, la sélection est recalculée et un nouvel audit est effectué à 800 000
noeuds avec les mêmes critères. Si ce deuxième audit échoue, la publication est
arrêtée pour examiner les annotations. Le budget retenu et les résultats des
audits figurent dans le manifeste.

## 5. Composition du banc figé

Le fichier final contient exactement 10 000 positions. Une seule position par
partie est autorisée. La sélection se fait après la seconde passe Stockfish.

La WDL définit trois catégories :

- disputée : `0,35 <= S* <= 0,65` ;
- avantage : `0,15 <= S* < 0,35` ou `0,65 < S* <= 0,85` ;
- décisive : `S* < 0,15` ou `S* > 0,85`.

Les quotas exacts sont :

| Phase | Disputée | Avantage | Décisive | Total |
|---|---:|---:|---:|---:|
| Ouverture | 1 000 | 600 | 400 | 2 000 |
| Milieu | 3 000 | 1 800 | 1 200 | 6 000 |
| Finale | 1 000 | 600 | 400 | 2 000 |
| Total | 5 000 | 3 000 | 2 000 | 10 000 |

Dans chaque case, les positions sont ordonnées par le SHA-256 de
`position_id | lapzero-position-bench-v1-final` et retenues dans cet ordre, sous
les contraintes supplémentaires suivantes :

- au moins 45 % de positions avec les blancs au trait et 45 % avec les noirs ;
- au plus 100 positions d'un même événement ;
- au plus 20 occurrences d'un même joueur, toutes couleurs confondues.

Si un quota ne peut pas être rempli, le constructeur ajoute le mois précédent et
recommence la sélection. Il ne relâche silencieusement aucun filtre.

Le sous-banc MCTS contient 256 positions prises dans ces 10 000. Il reprend au
plus près les proportions de phase et de WDL, avec une sélection par hachage
indépendante. Il est stocké comme une liste d'identifiants dans le manifeste et ne
change pas entre les checkpoints.

## 6. Format et versionnement

Le banc vit dans `data/position_bench/v1/` :

- `positions.jsonl.zst` : un objet par position avec historique, métadonnées et
  annotations de tous les coups légaux ;
- `manifest.json` : schéma, hashes, sources, licence, paramètres Stockfish,
  quotas, identifiants du sous-banc MCTS et résultat de l'audit ;
- `README.md` : provenance, attribution et interdiction d'entraînement.

Le manifeste contient aussi le SHA-256 du fichier compressé. Le chargeur vérifie
le schéma, les comptes, le hash et les paramètres du sous-banc avant toute
inférence. Le dataset `v1` est immuable. Toute modification de source, de filtre,
de budget Stockfish ou de quota crée `v2` et une nouvelle série W&B.

Les archives PGN et les fichiers de travail d'annotation restent hors de Git. Le
banc dérivé et son manifeste sont committés s'ils restent sous la limite Git de
100 Mio par fichier. Dans le cas contraire, le manifeste est committé et le banc
est distribué comme artefact W&B identifié par son SHA-256. Le chargeur refuse un
artefact dont le hash ne correspond pas au manifeste.

## 7. Évaluation récurrente

### 7.1 Policy et value sur 10 000 positions

Le chemin principal charge les tenseurs par lots de 1 024 dans une session ONNX
GPU unique. Pour chaque position :

1. rejouer l'historique complet côté C++ ;
2. obtenir les logits et la value ;
3. appliquer un softmax uniquement sur les coups légaux ;
4. joindre chaque probabilité à la WDL Stockfish pré-calculée ;
5. écrire les mesures brutes dans un résultat compressé.

Le même passage batché calcule les métriques de policy et de value. Tester la
value à chaque évaluation n'ajoute donc pas un second parcours du modèle.

Ce passage n'exécute jamais Stockfish. Il doit être déterministe au bit près sur
les coups choisis. Sur la même machine, le même ONNX et la même version d'ONNX
Runtime, les probabilités individuelles doivent être reproductibles à `1e-7` et
les métriques agrégées à `1e-6`.

### 7.2 Recherche sur 256 positions

Le sous-banc utilise :

- 384 simulations par position ;
- `c_puct = 1,4` ;
- aucun bruit de Dirichlet ;
- aucun échantillonnage par température ;
- taille de batch MCTS égale à 8 et batch fixe activé ;
- huit travailleurs de recherche dans un seul processus ;
- virtual loss égal à 2, FPU égal à 0,30 et facteur de tentatives de collision égal à 4 ;
- une recherche neuve et une table de transposition froide par position.

Les 256 positions sont évaluées séquentiellement, dans l'ordre figé du sous-banc.
Le batching et les huit travailleurs servent à collecter plusieurs feuilles d'un
même arbre, pas à rechercher plusieurs positions en parallèle. Un unique objet
`MCTS` et un unique évaluateur ONNX GPU sont réutilisés pendant tout le passage.
`mcts_search()` crée et détruit sa racine locale à chaque appel. Entre deux
positions, l'orchestrateur remet la table de transposition et les compteurs à zéro
au repos, mais conserve le `SearchExecutor`, ses travailleurs et la session ONNX
chauds.

Cette configuration réutilise le chemin multicore batché existant. Elle ne lance
aucun processus Stockfish, aucun pool de 16 moteurs et aucune recherche concurrente
sur le même évaluateur. Le parallélisme interne peut introduire un très faible
bruit d'ordonnancement, raison pour laquelle la policy brute reste le score
principal.

### 7.3 Cible de durée

La cible est d'environ 300 secondes pour un passage complet, chargement,
agrégation et écriture des résultats compris. C'est un ordre de grandeur, pas une
échéance : un dépassement ponctuel n'interrompt pas l'évaluation et n'invalide pas
son score. Le temps réel et sa décomposition sont journalisés à chaque passage.

Une durée qui tend régulièrement vers huit ou neuf minutes déclenche une analyse
du coût avant de retenir la configuration. Le nombre de positions et le budget
de simulations restent fixes pendant les évaluations ; aucune réduction
automatique n'est utilisée pour atteindre artificiellement la cible.

Une évaluation interrompue par une erreur ou par l'utilisateur ne publie aucun
score agrégé partiel. Elle conserve seulement son statut, sa durée et le nombre
de positions terminées. Aucun watchdog ni arrêt forcé à cinq minutes n'est requis.

## 8. Métriques

### 8.1 Policy, value, centipions et MCTS

#### Policy

Pour une policy légale normalisée `pi`, le regret attendu est :

`R_policy = somme pi(coup) * (S* - S(coup))`

La métrique principale est la moyenne de `R_policy` sur les 10 000 positions. Elle
est continue, utilise toute la distribution de policy et s'exprime en perte
d'espérance de score de partie. Plus elle est faible, meilleur est le modèle.

Les métriques complémentaires sont :

- regret du coup argmax de la policy ;
- masse de policy sur les coups dont le regret est inférieur ou égal à 0,02 ;
- masse catastrophique sur les coups dont le regret est supérieur ou égal à 0,20.

#### Value scalaire

La cible de value est `V* = 2 * S* - 1`. À chaque évaluation, le rapport calcule :

- `MAE = moyenne(abs(V_modèle - V*))` ;
- `RMSE = racine(moyenne((V_modèle - V*)^2))` ;
- le biais signé `moyenne(V_modèle - V*)` ;
- la corrélation de Pearson entre `V_modèle` et `V*`.

Ces courbes restent diagnostiques. La cible d'entraînement de LapZero est le
résultat final de ses parties de self-play, alors que `V*` est une espérance de
résultat estimée par Stockfish. La corrélation est journalisée comme absente si
l'une des deux séries a une variance nulle.

#### Centipions

Les centipions restent visibles pour répondre à une question plus tactique : de
combien l'évaluation brute chute-t-elle lorsque la policy choisit son coup argmax ?
Le rapport publie :

- le regret en centipions médian et son 90e percentile ;
- la proportion de coups argmax à 20, 50 et 100 centipions ou moins du meilleur ;
- la couverture de ces statistiques.

Une position est exclue de ces seuls diagnostics si le meilleur coup ou le coup
choisi est évalué comme un mat. Aucun plafond arbitraire ne convertit un mat en
centipions. La couverture indique explicitement la fraction restante, tandis que
les métriques WDL continuent d'inclure toutes les positions.

#### MCTS

Pour le sous-banc MCTS, les mêmes regrets sont calculés à partir de la distribution
de visites et de son argmax. Ils sont publiés séparément de ceux de la policy
directe.

Chaque mesure est publiée globalement et par phase, catégorie WDL et type de
joueur humain ou bot. Aucun score composite ne mélange policy, value, centipions
et MCTS.

### 8.2 Décider objectivement si le modèle progresse

Deux checkpoints sont comparés position par position. La différence de regret est
donc appariée, ce qui élimine la variance de composition du banc.

Le rapport calcule un intervalle bootstrap apparié à 95 %, avec 10 000
rééchantillonnages et une graine fixe. Le nouveau checkpoint est déclaré en progrès
sur le score principal si la borne supérieure de l'intervalle de
`R_nouveau - R_précédent` est strictement négative.

Le rapport donne également la fraction de positions améliorées, inchangées et
dégradées. Une amélioration globale accompagnée d'une forte dégradation des finales
reste ainsi visible. Les variations de value et de MCTS sont présentées à côté,
mais ne changent pas la décision portée par le score principal de policy.

### 8.3 Noms W&B

La boucle d'entraînement journalise au minimum :

- `eval/position/policy_expected_regret` ;
- `eval/position/policy_argmax_regret` ;
- `eval/position/near_best_mass` ;
- `eval/position/catastrophic_mass` ;
- `eval/position/value_mae` ;
- `eval/position/value_rmse` ;
- `eval/position/value_bias` ;
- `eval/position/value_correlation` ;
- `eval/position/policy_argmax_cp_regret_median` ;
- `eval/position/policy_argmax_cp_regret_p90` ;
- `eval/position/policy_within_20cp` ;
- `eval/position/policy_within_50cp` ;
- `eval/position/policy_within_100cp` ;
- `eval/position/policy_cp_coverage` ;
- `eval/position/search_expected_regret` ;
- `eval/position/search_argmax_regret` ;
- `eval/position/delta_policy_regret` et les deux bornes à 95 % ;
- `eval/position/duration_s` ;
- `eval/position/completed_positions` ;
- `eval/position/status` ;
- `eval/position/dataset_version`.

Les anciennes séries `eval/elo_estim`, `eval/wins`, `eval/draws` et `eval/losses`
restent dans l'historique W&B, mais ne sont plus alimentées par défaut.

## 9. Architecture Python

### `python_src/build_position_bench.py`

CLI hors ligne pour télécharger ou lire les archives, vérifier leur hash, parser
les PGN, extraire les candidats, piloter Stockfish, appliquer les quotas et écrire
le dataset figé. Les étapes sont reprenables : criblage, annotation et sélection
écrivent des fichiers de travail dotés d'un manifeste. Une reprise refuse des
paramètres ou hashes différents.

### `python_src/position_bench_metrics.py`

Logique pure et testable sans ONNX ni Stockfish : schéma, espérance WDL, regret,
agrégation par tranche, bootstrap apparié et validation du manifeste.

### `python_src/position_bench.py`

Chargement du modèle et du dataset, inférence ONNX batchée, sous-banc MCTS,
écriture des résultats bruts et résumé JSON. Il réutilise le chargement de modèle
et le codage des coups déjà validés par `puzzle_bench.py`, sans modifier le banc de
puzzles existant.

### `python_src/train_self_play.py` et `python_src/run_selftrain.py`

Le bloc `evaluate_against_anchor` est remplacé par l'appel au banc de positions.
`--eval-every` vaut 4 par défaut dans les deux points d'entrée. Le chemin du banc,
le nombre de travailleurs de recherche et la cible de durée sont configurables,
mais leur valeur effective est écrite dans W&B.

Le chemin Stockfish n'est plus une dépendance obligatoire de la boucle de
self-play. `stockfish_player.py` et le match d'ancrage restent disponibles comme
outils manuels, sans suppression de fonctionnalité.

Les résultats par position sont enregistrés hors Git sous
`position_bench_results/<checkpoint>.npz`, avec un sidecar contenant les hashes du
modèle et du dataset. Ils permettent une nouvelle agrégation sans relancer le
réseau.

## 10. Gestion des erreurs

- PGN illégal, incomplet ou non standard : rejet explicite avec compteur par
  raison.
- Elo absent ou inférieur à 2000 pour un joueur : rejet avant extraction.
- Historique rejoué différent de la FEN de contrôle : erreur de données, position
  exclue.
- Coup Stockfish absent des coups légaux ou score sans WDL : annotation rejetée et
  retentée une fois dans un nouveau processus ; second échec fatal pour le build.
- Quota impossible à remplir : ajout du mois précédent, jamais de relâchement
  silencieux.
- Hash ou schéma du dataset invalide : évaluation refusée avant chargement du
  modèle.
- Coup légal sans annotation : évaluation refusée, aucun regret approximé.
- Mat dans un diagnostic en centipions : position exclue de ce diagnostic et
  couverture réduite, sans affecter les métriques WDL.
- Cible de 300 secondes dépassée : évaluation poursuivie jusqu'au bout et durée
  réelle publiée ; un passage complet reste valide.
- Erreur ou interruption : statut explicite, aucun score agrégé partiel publié.
- Résultat précédent issu d'un autre dataset ou budget : comparaison appariée
  refusée.

## 11. Tests

### Construction du dataset

- parser un PGN de broadcast avec commentaires et variantes en ne gardant que la
  ligne principale ;
- conserver une partie `BOT` à Elo suffisant ;
- rejeter variante, Elo insuffisant, `SetUp`, historique illégal et partie trop
  courte ;
- reproduire les mêmes candidats quand l'ordre des PGN change ;
- distinguer deux entrées réseau partageant la même FEN mais pas le même historique ;
- appliquer exactement les quotas, plafonds par événement et par joueur ;
- vérifier la reprise avec mêmes paramètres et son refus avec paramètres différents ;
- convertir correctement la WDL et le score Stockfish du point de vue du joueur au
  trait, y compris après un coup des noirs et sur un mat.

### Métriques

- regret nul si toute la masse porte sur un meilleur coup ;
- regret attendu exact sur une distribution synthétique à trois coups ;
- seuils de masse proche et catastrophique aux valeurs frontières ;
- conversion exacte de `S` vers la cible scalaire `V_sf` ;
- MAE, RMSE, biais et corrélation sur des valeurs synthétiques ;
- médiane, 90e percentile et seuils en centipions, avec exclusion des mats et
  calcul de couverture ;
- agrégation par phase, WDL et type humain ou bot ;
- bootstrap apparié déterministe avec la graine fixée ;
- refus d'une comparaison entre deux versions de dataset.

### Runtime

- accord du softmax masqué avec les priors C++ déjà vérifiés par le banc de puzzles ;
- aller-retour entre coup UCI, index 4672 et annotation ;
- mêmes coups policy et écarts numériques inférieurs aux tolérances de la
  section 7.1 sur deux exécutions du même ONNX ;
- MCTS sans bruit de Dirichlet, avec exactement 384 simulations terminées ;
- positions MCTS traitées sans chevauchement sur un objet réutilisé, avec racine
  neuve, table froide et pool de travailleurs persistant ;
- passage complet publié même au-delà de la cible de durée ; interruption ou
  erreur sans publication de score partiel ;
- exécution de bout en bout sur un fixture de quelques positions avec un faux
  évaluateur, puis avec le vrai module C++ quand il est disponible.

## 12. Critères d'acceptation

L'implémentation est acceptée lorsque :

1. le manifeste prouve que les sources sont postérieures au préentraînement et ne
   viennent pas de LapZero ;
2. le dataset contient exactement 10 000 positions et le sous-banc MCTS exactement
   256 identifiants ;
3. chaque coup légal de chaque position possède une WDL Stockfish et un score en
   centipions ou en mat ;
4. l'audit de stabilité de la section 4.3 passe ;
5. deux passages policy sur le même modèle produisent les mêmes métriques ;
6. trois évaluations complètes consécutives sont chronométrées sur la machine
   d'entraînement de référence ; leur durée vise environ cinq minutes, avec les
   dépassements ponctuels acceptés et une investigation si elle tend vers huit
   ou neuf minutes ;
7. W&B reçoit séparément les métriques de policy, value, centipions et MCTS, ainsi
   que le statut, la durée et la version du dataset ;
8. la boucle de self-play ne lance plus le match Stockfish par défaut ;
9. le match Stockfish manuel reste utilisable ;
10. la suite Python et la suite C++ existantes restent vertes.

## 13. Ce que le banc ne dira pas

- Il ne produit pas un Elo absolu.
- Il ne remplace pas les parties réelles du bot Lichess.
- La WDL Stockfish est une calibration issue de son évaluation, pas une probabilité
  littérale que LapZero obtienne ce résultat contre un adversaire donné.
- Il ne mesure pas directement la gestion du temps, l'adaptation à un adversaire ou
  la conversion complète d'une partie.
- Il peut finir par subir un surapprentissage humain si les décisions de recherche
  sont répétitivement prises à partir de ses positions détaillées. Pour limiter ce
  risque, les résultats agrégés sont utilisés pendant le développement et les
  positions individuelles ne sont jamais injectées dans l'entraînement.
- Le score MCTS peut garder un faible bruit dû au parallélisme. Le score principal
  reste donc celui de la policy directe, déterministe et calculé sur les 10 000
  positions.
