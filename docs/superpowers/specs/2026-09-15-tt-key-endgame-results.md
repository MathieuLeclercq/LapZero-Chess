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

| Politique | Mats | Nulles par les 50 coups | Parties | Somme des durées des parties |
|---|---:|---:|---:|---:|
| `legacy` | 10 | 2 | 12 | 158,9 s |
| `h0` | 11 | 1 | 12 | 161,8 s |
| `h1` | 11 | 1 | 12 | 142,8 s |
| `h7` | 12 | 0 | 12 | 123,7 s |

Les durées sont la somme des douze temps affichés par le banc pour chaque
politique. Elles ne comprennent pas l'initialisation du modèle et ne doivent pas
être confondues avec les durées des passages puzzles. La durée d'une partie
dépend aussi de sa longueur et du nombre de réutilisations de la TT.

## Résultats par famille

| Famille | `legacy` | `h0` | `h1` | `h7` |
|---|---|---|---|---|
| Tour contre roi, 4 positions | 3 mats, 1 nulle | 4 mats | 4 mats | 4 mats |
| Dame contre roi, 3 positions | 3 mats | 3 mats | 3 mats | 3 mats |
| Deux tours contre roi, 2 positions | 2 mats | 2 mats | 2 mats | 2 mats |
| Dame+pion contre fou, 2 positions | 2 mats | 2 mats | 2 mats | 2 mats |
| Deux fous contre roi, 1 position | 1 nulle | 1 nulle | 1 nulle | 1 mat |

## Demi-coups pour chaque position

Chaque cellule donne l'issue et le nombre de demi-coups joués avant le mat ou
la nulle. Une nulle par les 50 coups survient ici au bout de 100 demi-coups.
Ce nombre comprend les coups des deux camps ; 31 demi-coups correspondent à
environ 15 coups complets et un dernier coup.

| Position | `legacy` | `h0` | `h1` | `h7` |
|---|---:|---:|---:|---:|
| Tour contre roi, roi noir au centre (`tour_centre`) | nulle, 100 | mat, 31 | mat, 73 | mat, 55 |
| Tour contre roi, roi noir au bord (`tour_bord`) | mat, 3 | mat, 3 | mat, 3 | mat, 3 |
| Tour contre roi, roi noir au coin (`tour_coin`) | mat, 3 | mat, 3 | mat, 3 | mat, 3 |
| Tour contre roi, trait aux noirs (`tour_trait_noirs`) | mat, 20 | mat, 52 | mat, 14 | mat, 54 |
| Dame contre roi, roi noir au centre (`dame_centre`) | mat, 27 | mat, 11 | mat, 11 | mat, 11 |
| Dame contre roi, roi noir au bord (`dame_bord`) | mat, 3 | mat, 3 | mat, 3 | mat, 3 |
| Dame contre roi, trait aux noirs (`dame_trait_noirs`) | mat, 18 | mat, 38 | mat, 26 | mat, 36 |
| Deux tours contre roi, roi noir au bord (`deux_tours_bord`) | mat, 3 | mat, 3 | mat, 3 | mat, 3 |
| Deux tours contre roi, roi noir au coin (`deux_tours_coin`) | mat, 3 | mat, 3 | mat, 3 | mat, 3 |
| Dame+pion contre fou, position 1 (`dame_pion_fou_1`) | mat, 15 | mat, 23 | mat, 31 | mat, 39 |
| Dame+pion contre fou, position 2 (`dame_pion_fou_2`) | mat, 17 | mat, 97 | mat, 31 | mat, 39 |
| Deux fous contre roi (`deux_fous`) | nulle, 100 | nulle, 100 | nulle, 100 | mat, 67 |

Dans `dame_pion_fou_2`, `h0` finit par mater, mais seulement après 97
demi-coups. Le compteur maximal y atteint 82, contre 16 pour `h1` et 24 pour
`h7`. C'est l'autre trajectoire longue à considérer dans le choix final.

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
