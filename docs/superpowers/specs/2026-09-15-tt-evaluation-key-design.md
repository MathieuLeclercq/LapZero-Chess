# Clé de cache des évaluations réseau

Date : 15 septembre 2026

## 1. Problème

La table de transpositions de LapZero stocke la policy et la value produites par le
réseau. Elle est actuellement indexée par le seul Zobrist de la position courante.
Ce Zobrist décrit les pièces, le trait, les droits de roque et la prise en passant,
mais pas le compteur des 50 coups ni l'historique envoyé au réseau.

Une entrée peut donc être retrouvée pour un tensor différent de celui qui avait
produit sa policy et sa value. L'audit des finales a montré que ce cas n'est pas
seulement théorique : en tour contre roi, conserver la TT provoque une nulle par la
règle des 50 coups, tandis qu'une TT neuve permet au même modèle de mater.

## 2. État échiquéen et entrée du réseau

Deux notions doivent rester séparées.

L'état nécessaire aux règles des échecs comprend la position courante, le trait, les
droits de roque, la prise en passant, le compteur des 50 coups et les informations
nécessaires aux répétitions. Les positions précédentes n'influencent pas autrement la
valeur théorique d'une position.

Le réseau de LapZero ne reçoit cependant pas seulement cet état minimal. Son tensor
contient huit blocs temporels de quatorze plans : la position courante et les sept
positions précédentes, avec deux plans de répétition par bloc. Il reçoit aussi le
nombre total de coups, le compteur des 50 coups et le mode d'amnésie par la présence
ou l'absence des blocs passés. Le réseau peut avoir appris à utiliser ces signaux,
même lorsqu'ils ne sont pas indispensables à un joueur d'échecs parfait.

Le problème de clé est donc un compromis entre deux objectifs :

- ne pas réutiliser une évaluation lorsque son contexte pertinent diffère ;
- conserver les hits dont la différence d'historique a peu d'effet pratique.

## 3. Définition de la profondeur historique

La profondeur historique de la clé ne modifie jamais le tensor envoyé au réseau. Elle
configure seulement combien de positions antérieures participent au hash du cache.

| Profondeur | Positions couvertes par la clé |
|---:|---|
| 0 | position courante seulement |
| 1 | position courante et position précédente |
| 3 | position courante et trois positions précédentes |
| 7 | les huit positions temporelles vues par le réseau |

La comparaison ne parcourt pas directement les plans du tensor. Une empreinte des
informations effectivement encodées dans chaque bloc retenu est combinée dans une clé
de 64 bits.

La profondeur 7 est le témoin d'exactitude, pas le choix favori. Elle interdit tout
partage entre deux évaluations dont les huit blocs temporels diffèrent. La profondeur
0 correspond au modèle échiquéen minimal. La profondeur 1 ajoute le dernier coup,
signal que le réseau de LapZero semble réellement exploiter d'après les expériences
sur les puzzles avec et sans historique.

## 4. Compteur des 50 coups

Le compteur exact des 50 coups participe toujours à la clé. Il ne sera pas quantifié
dans la première implémentation.

Cette exigence devrait faire perdre moins de hits que ne le suggère une comparaison
comme 2 contre 3. Le Zobrist contient le joueur au trait : après un nombre impair de
demi-coups, ce joueur change déjà et les deux positions ne peuvent pas partager la
même entrée. Pour retrouver la même position avec le même joueur, la différence de
compteur est normalement paire et provient d'un cycle supplémentaire. Ce cycle
rapproche réellement la nulle et modifie le plan `no_progress` du réseau.

Les transpositions obtenues par permutation de coups atteignent généralement la même
position à la même profondeur et gardent le même compteur. La réutilisation d'une
branche explorée avant le coup adverse garde également le compteur correspondant à
cette branche. Ces hits utiles ne sont donc pas supprimés.

Un bucket de largeur 2 ne récupérerait presque aucun hit naturel, puisque les deux
valeurs regroupées correspondent à des joueurs au trait différents. Un bucket plus
large commencerait à fusionner les cycles qui posent précisément problème. La
quantification ne sera envisagée que si les mesures montrent une perte importante.

## 5. Références externes

La recherche DAG de Lc0 calcule sa clé avec la position courante, le compteur exact
des 50 coups et un nombre configurable de positions antérieures. Son option
`CacheHistoryLength` varie de 0 à 7 et vaut 0 par défaut. La documentation de l'option
reconnaît qu'une profondeur inférieure à celle du réseau peut réutiliser une
évaluation issue d'un autre historique.

Lc0 possède aussi un cache neuronal générique plus permissif, fondé sur le hash de la
position courante. Le code contient un `TODO` indiquant que la prise en compte de
l'historique doit être améliorée. Il n'existe donc pas une politique unique dans Lc0,
mais sa TT DAG fournit un précédent direct pour « compteur exact, historique
configurable ».

Stockfish conserve une clé large et désactive certains cutoffs de TT lorsque le
compteur atteint 96, en raison de l'interaction entre graphe et historique. Cette
solution est adaptée à une TT alpha-bêta, mais moins à LapZero : notre entrée stocke
directement une policy et une value calculées par un réseau sensible au compteur.

Sources :

- `PositionHistory::HashLast` de Lc0 :
  <https://github.com/LeelaChessZero/lc0/blob/master/src/chess/position.cc#L126-L134>
- utilisation dans la recherche DAG :
  <https://github.com/LeelaChessZero/lc0/blob/master/src/search/dag_classic/search.cc#L1966-L1969>
- définition de `CacheHistoryLength` :
  <https://github.com/LeelaChessZero/lc0/blob/master/src/search/classic/params.cc#L315-L320>
- cache neuronal actuel de Lc0 :
  <https://github.com/LeelaChessZero/lc0/blob/master/src/neural/memcache.cc#L33-L38>
- protection de Stockfish à compteur élevé :
  <https://github.com/official-stockfish/Stockfish/blob/master/src/search.cpp#L812-L831>

## 6. Politiques à comparer

Le benchmark comparera cinq politiques sans présumer du résultat :

1. `legacy` : Zobrist courant seulement, comportement actuel ;
2. `h0` : contexte courant exact et aucune position antérieure ;
3. `h1` : contexte courant exact et une position antérieure ;
4. `h3` : contexte courant exact et trois positions antérieures ;
5. `h7` : contexte courant exact et sept positions antérieures.

Dans les quatre nouvelles politiques, le contexte courant comprend au minimum :

- le Zobrist courant ;
- le compteur exact des 50 coups ;
- la catégorie de répétition du bloc courant ;
- la valeur effectivement encodée dans le plan du nombre total de coups ;
- les informations courantes de roque et de prise en passant déjà couvertes par le
  Zobrist.

Pour chaque position antérieure couverte par la profondeur, la clé combine uniquement
le placement des pièces et sa catégorie de répétition. Elle ne doit pas inclure les
anciens droits de roque, l'ancienne prise en passant ou le joueur qui avait le trait,
car ces informations ne figurent pas dans les quatorze plans historiques. Employer le
Zobrist historique brut serait inutilement strict.

Le mode d'amnésie est représenté par l'absence effective des blocs passés, pas par un
bit global. Ainsi, `h0` peut partager une évaluation entre les modes normal et amnésique
puisqu'il ignore volontairement tous les blocs passés. À partir de `h1`, un bloc absent
et un bloc rempli ont des empreintes différentes. `h7` doit ainsi distinguer exactement
les éléments temporels qui peuvent changer le tensor, sans distinguer des métadonnées
que le réseau ne voit pas.

Si une décision devait être prise sans mesure, `h1` serait le choix provisoire : il
reste proche de l'état échiquéen minimal tout en protégeant le dernier coup, dont la
sensibilité a déjà été observée. Le choix définitif doit cependant venir des résultats.

## 7. Instrumentation et benchmarks

L'implémentation doit permettre de classer les consultations de cache :

- correspondance sur le Zobrist courant ;
- rejet causé par le compteur des 50 coups ;
- rejet causé par le contexte courant restant ;
- rejet causé par l'historique supplémentaire ;
- hit accepté par la politique active.

Les cinq politiques seront comparées sur les mêmes positions, sans bruit de
Dirichlet, avec le même modèle et les mêmes budgets de simulations. Les mesures sont :

- taux de hits et motifs de rejet ;
- nombre d'appels réseau et de batches ;
- évaluations par seconde et durée totale ;
- remplissage moyen des batches ;
- accord du meilleur coup avec la politique `h7` ;
- score du banc de puzzles ;
- résultat, longueur et compteur maximal de la finale tour contre roi.

Le banc de recherche sur positions isolées mesure le débit brut. Une séquence fixe de
positions avec arbre et TT réutilisés mesure le comportement réel entre plusieurs
coups. Le diagnostic tour contre roi vérifie la disparition de la régression qui a
motivé le chantier.

La perte de hits doit être mesurée avant toute tentative de bucket. Si `h0` et `h1`
ont des performances proches, le résultat qualitatif et la stabilité du meilleur coup
les départagent. `h3` et `h7` servent à détecter un éventuel effet des historiques plus
anciens, pas à imposer leur utilisation.

## 8. Contraintes de validation

La correction devra :

1. faire passer le test `xfail` du compteur des 50 coups ;
2. distinguer deux états de répétition différents ;
3. distinguer les modes normal et amnésique dès que la profondeur couvre au moins un
   bloc passé effectivement disponible, mais pas en `h0` ;
4. respecter exactement la profondeur sélectionnée dans des tests synthétiques ;
5. utiliser la même fonction de clé dans tous les chemins séquentiels et batchés ;
6. conserver les invariants d'arbre et l'équilibre du virtual loss ;
7. passer la suite complète, le perft strict et le fuzz différentiel existants ;
8. faire mater la finale tour contre roi avec réutilisation normale de la TT.

La politique retenue ne doit pas être choisie sur le seul taux de hits. Un hit qui
retourne une évaluation provenant d'un contexte nuisible est une erreur évitée, pas une
perte de performance.

## 9. Décision de conception

Le compteur des 50 coups sera exact. Les profondeurs 0 et 1 sont les candidates
principales. Les profondeurs 3 et 7 restent des témoins expérimentaux. Aucun bucket du
compteur ne sera ajouté sans preuve mesurée qu'il récupère un volume significatif de
hits et ne dégrade pas les tests qualitatifs.
