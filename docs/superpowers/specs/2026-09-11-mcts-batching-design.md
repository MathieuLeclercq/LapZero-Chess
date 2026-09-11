# Batching de la recherche MCTS avec virtual loss : conception

Date : 2026-09-11
Statut : spec validée, prête pour le plan d'implémentation

Faire évaluer plusieurs positions par inférence au lieu d'une seule, dans un même arbre de
recherche, au moyen d'un virtual loss.

Prérequis déjà levés : la race `parse_position` est corrigée, et le pipeline de mesure
(`2026-09-11-search-bench-design.md`) fournit la référence de débit et le filet
d'invariants.

## 1. Le problème, mesuré

Le moteur fait **une inférence GPU par simulation** (`mcts.cpp:180`).

La référence `2026-09-11-search-bench-resultats.md` donne un résultat plus parlant que le
débit brut : **286 à 299 simulations par seconde, quelle que soit la position**. Une finale
à 14 coups légaux et un milieu de jeu à 48 tournent à 2 pour cent près à la même vitesse,
et des arbres de profondeur 15 et 19 aussi.

Cette insensibilité totale à la forme de l'arbre est la preuve que le coût n'est ni dans la
génération de coups, ni dans la descente, ni dans le calcul : c'est une latence fixe
d'environ 3,4 ms par simulation, payée pour lancer un noyau sur une seule position. Le GPU
n'apporte d'ailleurs rien, 376 sims/s avec contre 373 sans.

## 2. Ce que la recherche documentaire a apporté

Lecture du code de Leela Chess Zero, `src/search/classic/node.h` et `search.cc`.

```cpp
uint32_t n_in_flight_ = 0;   // (AKA virtual loss.) How many threads currently process this node
int GetNStarted() const { return n_ + n_in_flight_; }
```

```cpp
current_score[idx] = current_pol[idx] * puct_mult / (1 + nstarted) + util;
```

**Le virtual loss de LC0 est un compteur séparé qui n'entre que dans le dénominateur du
terme U.** Leur Q n'est pas touché : `wl_`, `d_` et `m_` ne bougent qu'au
`FinalizeScoreUpdate`. Ce n'est donc pas le virtual loss classique de Chaslot, qui ajoute
une défaite à la value, mais une inflation de visites.

Deux enseignements, qui ont chacun changé la conception.

**Un piège propre à ce code.** LC0 sépare `n_`, qui sert à Q, de `n_started`, qui sert à U.
Or `MCTSNode::q_value()` vaut `total_value / visit_count` et partage donc son compteur avec
le terme U. Incrémenter naïvement `visit_count` en croyant suivre LC0 ne reproduirait ni
LC0 ni Chaslot : Q se diluerait vers zéro, ce qui pénalise un nœud gagnant mais
**avantage un nœud perdant**.

**LC0 ne force pas la divergence.** `TryStartScoreUpdate` renvoie un booléen et un échec est
enregistré comme une collision. Ils acceptent que des descentes retombent au même endroit
plutôt que d'insister.

Sources : [node.h](https://github.com/LeelaChessZero/lc0/blob/master/src/search/classic/node.h),
[search.cc](https://github.com/LeelaChessZero/lc0/blob/master/src/search/classic/search.cc),
[searchparams.h de KataGo](https://github.com/lightvector/KataGo/blob/master/cpp/search/searchparams.h),
qui expose `numVirtualLossesPerThread` en `double` avec 1.0 par défaut.

## 3. Périmètre retenu

| Décision | Choix |
|---|---|
| Méthode | approche LC0, compteur séparé, Q intact |
| Noyau | **unique**, servant `step_analysis` et `mcts_search` |
| Taille de batch | paramètre **par appel**, défaut 32 |
| Boucle séquentielle | **conservée**, sous `batch_size = 0` |
| Détection de collision | `n_in_flight > 0` sur un nœud sans enfants |

**Un noyau unique servant les deux chemins**, parce que le banc de puzzles, qui est la
barrière qualité, appelle `mcts_search`. Ne batcher que `step_analysis` laisserait la
barrière mesurer le chemin non batché, donc ne rien voir du travail.

**La taille de batch par appel** permet au harnais de balayer 1, 2, 4, 8, 16, 32, 64 en une
exécution, et rend `batch_size = 1` directement testable.

## 4. Architecture

Un noyau privé `MCTS::run_search(MCTSNode* root, Chessboard& board, int simulations,
float c_puct, int batch_size)`, dans un nouveau fichier `src/mcts_batch.cpp`.
`step_analysis` et `mcts_search` deviennent des enveloppes qui l'appellent, chacune gardant
sa gestion de racine : création ou réutilisation de `m_analysis_root` pour l'une, racine
locale pour l'autre.

Le fichier séparé suit le précédent de `mcts_observe.cpp` : `mcts.cpp` fait déjà plus de
520 lignes.

Sémantique de `batch_size` :

| Valeur | Comportement |
|---|---|
| `0` | boucle séquentielle actuelle, conservée telle quelle |
| `1` | boucle batchée, un seul élément |
| `>= 2` | boucle batchée, défaut 32 |

La boucle séquentielle est gardée pour deux raisons : elle sert de référence au test
d'équivalence, et elle reste une porte de sortie si le batching décevait.

## 5. Le virtual loss, dérivé sur ce code

Ajout d'un champ à `MCTSNode` :

```cpp
uint32_t n_in_flight = 0;   // virtual loss : descentes en cours passant par ce noeud
```

`ucb_score` l'utilise dans le **seul** dénominateur de U :

```cpp
float u = exploration_factor * prior / (1.0f + visit_count + n_in_flight);
float exploitation = (visit_count == 0) ? (parent_q - fpu_reduction) : -q_value();
return exploitation + u;
```

`q_value()` continue d'utiliser `visit_count` seul, donc Q reste honnête.

Application et annulation sont des incréments entiers sur le chemin, du nœud vers la racine
par les pointeurs `parent` :

```cpp
static void poser_virtual_loss(MCTSNode* noeud) {
    for (MCTSNode* n = noeud; n != nullptr; n = n->parent) n->n_in_flight += 1;
}

static void annuler_virtual_loss(MCTSNode* noeud) {
    for (MCTSNode* n = noeud; n != nullptr; n = n->parent) n->n_in_flight -= 1;
}
```

**On ne peut pas réutiliser `backup()`** pour cela : il alterne le signe en remontant
(`mcts.cpp:71`), ce qui n'a aucun sens pour un compteur.

### Pourquoi l'approche LC0 plutôt que Chaslot, dans ce code précis

Chaslot dissuaderait plus fort, et les deux tournent dans des moteurs solides. Le choix ne
se fait donc pas sur la force théorique du mécanisme mais sur les effets de bord.

Chaslot incrémenterait `visit_count`, ce qui dérèglerait au passage deux choses sans
rapport : le `parent_q` qui alimente le FPU des frères non visités (`mcts.cpp:120`), et
l'`exploration_factor` en `c_puct * sqrt(node->visit_count)` du parent (`mcts.cpp:121`). Un
compteur séparé ne perturbe que ce qu'il doit perturber.

Note pour plus tard : `exploration_factor` utilise `sqrt(visit_count)` du parent, sans le
`n_in_flight`. C'est délibéré, pour que le virtual loss n'agisse que sur le terme U des
enfants. Si la divergence des descentes se révélait insuffisante, l'inclure est la première
variante à essayer, avant d'augmenter l'amplitude.

## 6. Les collisions

Le piège principal du chantier ne provoque aucun plantage. `select_leaf` fait de
l'expansion paresseuse et **continue de descendre** après un succès de table
(`mcts.cpp:82-105`). Une feuille déjà collectée, qui n'a pas encore d'enfants, peut donc en
recevoir pendant la même collecte, après quoi l'expansion en créerait un second jeu.
L'arbre serait corrompu et les priors dupliqués, en silence.

**`n_in_flight > 0` sur un nœud sans enfants décrit exactement cet état**, donc aucun
drapeau supplémentaire n'est nécessaire. `select_leaf` s'arrête sur un tel nœud et signale
une collision, comme `TryStartScoreUpdate` renvoie `false` chez LC0.

À la collision, on annule le virtual loss posé sur ce chemin et **on arrête la collecte**
avec un batch partiellement rempli. On n'insiste pas : en début de recherche l'arbre n'a
pas toujours N branches distinctes à offrir, et boucler serait fragile.

## 7. La collecte, et la capture du plateau

```
tant qu'il reste des simulations :
    collecter jusqu'a batch_size feuilles :
        descendre en posant n_in_flight sur chaque noeud du chemin
        si terminal ou succes de table  -> deja traite, annuler, simulation comptee
        si collision                    -> annuler ce chemin, arreter la collecte
        sinon, le plateau est SUR la feuille : capturer
               le tenseur, les coups legaux, le Zobrist, l'echec
        annuler les coups joues pour revenir a la racine
    une seule inference sur le lot
    pour chaque feuille :
        annuler n_in_flight en remontant par les parents
        developper depuis la capture, puis remonter la valeur
```

Le point qui simplifie tout : **après la collecte, le plateau n'est plus jamais nécessaire.**
L'annulation du virtual loss remonte par les pointeurs `parent`, le backup aussi, et
l'expansion utilise la capture.

C'est ce qui évite de rejouer le chemin de chaque feuille avant son backup, ce qui aurait
coûté deux traversées supplémentaires par feuille. La capture ne coûte rien puisque le
plateau est déjà sur la feuille au moment où on y prend le tenseur.

Cela demande une variante `expand_and_backup_prepared(feuille, capture, policy, value)` qui
ne touche pas au plateau. L'originale reste : le self-play s'en sert et son plateau, lui,
est bien positionné, chaque partie ayant le sien.

Par feuille, la capture retient : le tenseur (119 x 64 flottants), les indices légaux, le
hachage Zobrist et l'état d'échec. Pour un batch de 32 cela représente une vingtaine de
kio, sans allocation par simulation si les tampons sont réutilisés entre les lots.

## 8. Barrières et critères d'acceptation

| Contrôle | Ce qu'il attrape |
|---|---|
| `batch_size` 1 contre 0, mêmes sorties | toute erreur dans la machinerie de collecte |
| `inspect_tree`, zéro violation | enfants dupliqués, visites incohérentes |
| `root.visit_count == simulations` | virtual loss mal annulé |
| Balayage du débit sur 1 à 64 | la courbe de gain réelle |
| Banc de puzzles à simulations égales | dégradation de la qualité par simulation |

L'équivalence à batch 1 devrait être exacte, une seule descente n'étant perturbée par
personne et l'annulation étant un décrément entier. Elle est prise comme **conséquence
attendue et non comme contrainte de conception** : si elle échouait à un epsilon près sans
autre symptôme, c'est le critère qu'on relâcherait, pas l'algorithme.

Le dernier contrôle est le plus important et le plus facile à oublier. **Plus de
simulations ne garantit pas plus de force** : le virtual loss modifie le comportement de la
recherche, donc il faut vérifier séparément le gain en volume et l'absence de perte en
qualité par simulation. Le banc de puzzles tranche ce point, à 700 simulations, en
comparaison appariée avec la référence déjà commitée.

Seuil de signification : l'étendue mesurée entre passages va jusqu'à 12 pour cent, donc un
gain inférieur à cela ne serait pas distinguable du bruit sur un seul passage.

## 9. Ce qui reste incertain

**L'ampleur du gain.** La référence mesure un arbre neuf. En partie réelle `update_root`
conserve le sous-arbre du coup joué, donc le taux de succès de la table est plus élevé et,
par la loi d'Amdahl, le batching n'accélère que la fraction des simulations qui appellent
réellement le réseau. La direction du gain ne fait pas de doute, son ampleur si. Voir la
section « Réserve sur le cas froid » de `2026-09-11-search-bench-resultats.md`.

**Le réglage de `c_puct`.** Retarder l'information de N descentes change l'exploration
effective. Le `1.4` en dur (`uci.py:298`, `selfplay_manager.cpp:387`) devra être revérifié,
mais après que le batching fonctionne, pas pendant.

**La taille de la table de transposition.** 8192 entrées suffisent à 400 simulations mais
deviendront sous-dimensionnées à 10 ou 40 fois plus, les chemins convergeant bien plus
souvent. À rebalayer au moment de mesurer le moteur batché.

**La table indexée sur le seul Zobrist**, qui ignore les 8 plans d'historique du tenseur.
C'est une inexactitude préexistante, mais le batching augmentera le trafic de table donc
l'exposition. Premier suspect si le banc de puzzles bouge bizarrement.

## 10. Hors périmètre

- Le self-play, qui batche déjà par 512 entre parties et n'a pas besoin du virtual loss :
  ses descentes viennent de 512 plateaux différents, donc elles divergent naturellement.
- Le réglage de `VIRTUAL_LOSS` au delà de 1 par descente, et la variante consistant à
  inclure `n_in_flight` dans l'`exploration_factor` du parent.
- Toute comparaison Chaslot contre LC0, que le banc de puzzles pourrait trancher plus tard.
