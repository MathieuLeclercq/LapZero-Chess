# Comparaison des politiques TT

Date : 15 septembre 2026. Commit du benchmark : `a85217a`.
Modèle : `2026_04_23_23h25_iter316_unsupervised.onnx` (itération 316).
GPU : NVIDIA RTX PRO 2000 Blackwell Generation Laptop GPU. Recherche GPU,
batch 8, 700 simulations, sans bruit de Dirichlet sur les
puzzles. Le banc de recherche utilise une TT de 8 192 entrées, le diagnostic
de finale une TT de 131 071 entrées. Les 2 500 puzzles de chaque politique
sont les mêmes et sont chargés avec leur historique réel.

## Décision

`h0` est retenu comme profondeur historique par défaut de la TT. Son contexte
courant, notamment le compteur exact des 50 coups, reste vérifié à chaque hit ;
seules les positions antérieures sont exclues de la clé. Le réseau reçoit
toujours son tensor de 119 plans avec l'historique complet. Le mode `legacy`
reste disponible pour comparer l'ancien comportement, et `h1` et `h7` restent
sélectionnables explicitement.

La décision suit les cinq critères prévus dans le
[plan d'implémentation](../plans/2026-09-15-tt-evaluation-key.md) :

1. `h0` passe les régressions et mate la finale tour contre roi causale en 31
   demi-coups, contre une nulle à 100 demi-coups en `legacy`.
2. Son taux puzzle de 77,04 % est inférieur de 0,16 point à `h1`, donc dans la
   marge maximale prévue de 0,5 point.
3. La comparaison appariée `h0`/`h1` compte 7 puzzles réussis seulement par
   `h0` et 11 seulement par `h1` ; McNemar `p = 0,48` ne détecte pas une
   dégradation significative. Ce résultat ne prouve pas l'équivalence.
4. `h0` dépasse `legacy` de 0,24 point sur les mêmes puzzles.
5. Dans le banc de recherche isolée, `h0` a un débit supérieur à `h1` dans
   chacune des trois positions sur le chemin `mcts_search` (voir ci-dessous).

Cette décision est serrée. `h1` réussit quatre puzzles de plus sur 2 500 et
mate `dame_pion_fou_2` en 31 demi-coups, contre 97 pour `h0`. Un tournoi de
parties réelles reste nécessaire pour établir un gain de niveau ou confirmer
que `h0` est le meilleur choix en finale. Les résultats ici n'estiment pas un Elo.

## Débit de recherche

Médianes de sept passages à 700 simulations, batch 8, en simulations par
seconde sur le chemin `mcts_search` :

| Position | `legacy` | `h0` | `h1` | `h7` |
|---|---:|---:|---:|---:|
| Ouverture | 975,3 | 1 051,4 | 1 004,3 | 981,7 |
| Milieu de jeu | 454,7 | 500,4 | 402,3 | 494,2 |
| Finale | 799,7 | 715,3 | 706,6 | 703,4 |

La finale isolée perd du débit face à `legacy`, car les hits dont le compteur
ne correspond pas sont rejetés. `h0` y accepte 9,2 % des consultations, contre
5,7 % pour `h1` et 14,2 % pour `legacy`. Les données détaillées, y compris
`step_analysis`, remplissage des batches et motifs de rejet, figurent dans
[le rapport de recherche](2026-09-15-tt-key-search-results.md).

Les quatre passages puzzles complets ont duré respectivement 15,4 min
(`legacy`), 20,9 min (`h0`), 20,6 min (`h1`) et 15,7 min (`h7`). Cette mesure
murale ne reproduit pas l'ordre du débit isolé : les campagnes ont été lancées
successivement avec 16 processus et une charge GPU variable. Elle ne prouve
donc pas que `h0` accélère le banc de puzzles en conditions réelles.

## Conversion des finales

Sur 12 positions et quatre politiques, 48 parties ont été jouées avec arbre et
TT persistants. Toutes les parties ont été rejouées sans coup illégal ni erreur
de synchronisation avec l'oracle de légalité.

| Politique | Mats | Nulles par les 50 coups | Tour contre roi causal | Deux fous contre roi |
|---|---:|---:|---|---|
| `legacy` | 10/12 | 2 | nulle, 100 demi-coups | nulle, 100 demi-coups |
| `h0` | 11/12 | 1 | mat, 31 demi-coups | nulle, 100 demi-coups |
| `h1` | 11/12 | 1 | mat, 73 demi-coups | nulle, 100 demi-coups |
| `h7` | 12/12 | 0 | mat, 55 demi-coups | mat, 67 demi-coups |

La longueur de mat pour chacun des 12 cas et les compteurs maximaux sont dans
[le rapport des finales](2026-09-15-tt-key-endgame-results.md). Le cas deux
fous contre roi dépend aussi de la connaissance de mat du réseau : son échec
avec `h0` et `h1` ne démontre pas à lui seul une erreur de clé TT.

## Puzzles appariés

Puzzles communs valides : **2500**. Chaque CSV contient 2 500 lignes et zéro
erreur de données. La table ci-dessous conserve les résultats bruts de la
comparaison automatique.

## Resultats par politique

| Politique | n | Resolus | Reussite | Somme des durees individuelles |
|---|---:|---:|---:|---:|
| legacy | 2500 | 1920 | 76.80 % | 14171.0 s |
| h0 | 2500 | 1926 | 77.04 % | 19494.8 s |
| h1 | 2500 | 1930 | 77.20 % | 19242.8 s |
| h7 | 2500 | 1928 | 77.12 % | 14415.8 s |

## Comparaisons appariees

Le delta est le taux de droite moins le taux de gauche.

| Gauche | Droite | Gauche seule | Droite seule | Accord coups | Delta reussite | McNemar p |
|---|---|---:|---:|---:|---:|---:|
| legacy | h0 | 9 | 15 | 97.88 % | +0.24 points | 0.307 |
| legacy | h1 | 11 | 21 | 97.56 % | +0.40 points | 0.112 |
| legacy | h7 | 13 | 21 | 97.32 % | +0.32 points | 0.23 |
| h0 | h1 | 7 | 11 | 98.12 % | +0.16 points | 0.48 |
| h0 | h7 | 10 | 12 | 97.96 % | +0.08 points | 0.831 |
| h1 | h7 | 11 | 9 | 98.36 % | -0.08 points | 0.823 |

La dernière colonne de durée additionne les temps des 2 500 puzzles traités
en parallèle ; ce n'est pas le temps mural du passage. Le protocole et les
résultats ne démontrent pas une amélioration du niveau du bot sur Lichess.

## Validation

Avant ce benchmark, les invariants MCTS, les tests du compteur exact, le
perft strict et le fuzz différentiel avaient passé la tâche de vérification.
Le défaut `h0` a été contrôlé par un test qui distingue effectivement `h0`
de `h1` sur deux historiques différents : il échouait avec l'ancien défaut
`h1`, puis a passé après recompilation avec `h0`.

Après sélection de `h0` :

- build Release de `chess_engine`, `chess_perft` et `chess_tests` : réussi ;
- tests Python ciblés : 69 réussis ; suite complète : 207 réussis ;
- CTest : 2 tests C++ réussis sur 2 ;
- perft strict rapide avec contrôle FEN : 277 949 noeuds, zéro violation ;
- fuzz différentiel : 400 000 positions, 5 148 parties, zéro divergence,
  1 051 roques et 2 162 promotions joués ;
- `git diff --check` : aucune erreur de formatage.

Les avertissements Python concernent l'export ONNX TorchScript historique.
La campagne de 200 000 positions avait aussi trouvé zéro divergence mais
n'atteignait pas le seuil de 1 000 roques du script ; elle a été prolongée
à 400 000 positions pour satisfaire ce critère sans changer le test.
