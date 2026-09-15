# Diagnostic de conversion des finales

Date : 15 septembre 2026

## Configuration

- Modèle : `2026_04_23_23h25_iter316_unsupervised.onnx`
- 12 positions, 700 simulations par coup
- batch MCTS : 8
- arbre et table de transpositions réutilisés entre les coups
- maximum : 110 demi-coups par partie
- politiques : `legacy`, `h0`, `h1`, `h7`
- 48 parties au total

## Résultats globaux

| Politique | Mats | Nulles par les 50 coups | Parties | Durée murale |
|---|---:|---:|---:|---:|
| `legacy` | 10 | 2 | 12 | 15,4 min |
| `h0` | 11 | 1 | 12 | 20,9 min |
| `h1` | 11 | 1 | 12 | 20,6 min |
| `h7` | 12 | 0 | 12 | 15,7 min |

La durée murale correspond au temps de chaque passage complet. Les durées
individuelles sont influencées par la longueur de la partie et par le nombre de
réutilisations de la TT.

## Résultats par famille

| Famille | `legacy` | `h0` | `h1` | `h7` |
|---|---|---|---|---|
| Tour contre roi, 4 positions | 3 mats, 1 nulle | 4 mats | 4 mats | 4 mats |
| Dame contre roi, 3 positions | 3 mats | 3 mats | 3 mats | 3 mats |
| Deux tours contre roi, 2 positions | 2 mats | 2 mats | 2 mats | 2 mats |
| Dame+pion contre fou, 2 positions | 2 mats | 2 mats | 2 mats | 2 mats |
| Deux fous contre roi, 1 position | 1 nulle | 1 nulle | 1 nulle | 1 mat |

## Position causale tour contre roi

Sur la position qui reproduisait la régression observée :

| Politique | Résultat | Demi-coups | Compteur maximal | Durée |
|---|---|---:|---:|---:|
| `legacy` | nulle par les 50 coups | 100 | 100 | 57,2 s |
| `h0` | mat | 31 | 31 | 10,3 s |
| `h1` | mat | 73 | 73 | 25,7 s |
| `h7` | mat | 55 | 55 | 17,1 s |

Les trois politiques corrigées suppriment donc la régression précise qui avait
motivé le chantier de la clé TT. `h0` donne la trajectoire la plus courte sur
cette position, mais ce résultat n'est pas monotone avec la profondeur historique.

## Interprétation

La profondeur historique modifie fortement les trajectoires du MCTS. Elle ne
produit pas une hiérarchie simple : `h0` mate la position causale en 31
demi-coups, `h1` en 73 et `h7` en 55.

La finale deux fous contre roi n'est pas un test isolé de la clé TT. Le réseau
échoue avec `legacy`, `h0` et `h1`, et réussit avec `h7`. Elle doit donc être
interprétée comme un test combiné du réseau, de la recherche et de la TT, pas
comme une preuve que `h7` est globalement préférable.

Les résultats puzzles sont nécessaires pour départager `h0` et `h1` sur le jeu
réel. Ce rapport ne modifie pas encore le défaut du moteur.
