# Clé d'évaluation TT : résultats provisoires

## Statut

Ce rapport a la structure du rapport de décision final, mais la campagne puzzle
complète n'a pas encore été exécutée. Les résultats tactiques ci-dessous
proviennent du préfiltre apparié de 200 puzzles. La recommandation est donc
provisoire et le défaut du moteur reste `h1` jusqu'à la campagne complète.

## Configuration

- Commit testé : `2753e90`
- Modèle : `2026_04_23_23h25_iter316_unsupervised.pt`, itération 316,
  `global_step` 19415
- Modèle d'inférence : export ONNX dynamique du même checkpoint
- GPU : NVIDIA RTX PRO 2000 Blackwell Generation Laptop GPU, 8 151 MiB
- Pilote NVIDIA : 595.95
- Recherche : 700 simulations, `c_puct=1.4`, batch 8
- Table de transpositions du banc de débit : 8 192 entrées
- Politiques : `legacy`, `h0`, `h1`, `h3`, `h7`

`legacy` valide uniquement le Zobrist courant. Les modes `hN` valident aussi
le compteur exact des 50 coups et le contexte courant du réseau. `N` indique
le nombre de positions antérieures qui doivent également correspondre.

## Résumé

La collision sémantique de la TT explique bien la nulle observée en finale.
Sur tour et roi contre roi, `legacy` atteint la règle des 50 coups alors que
les quatre politiques corrigées matent. `h0` mate le plus vite dans cette
trajectoire, en 31 demi-coups.

Le préfiltre tactique ne montre aucune dégradation de `h0` ou `h1`. Les deux
résolvent 74 % des 200 puzzles, contre 73 % pour `legacy`, et choisissent le
même coup dans 99 % des positions.

Le banc de débit favorise généralement `h0` face à `h1`, avec un avantage
particulièrement marqué sur la position de milieu de jeu. Cette mesure doit
être interprétée prudemment car les étendues montrent du bruit et la position
de milieu de jeu donne un remplissage de batch inférieur avec `h1`.

## Débit de recherche

Chaque valeur est la médiane de sept passages indépendants. Le tableau complet,
avec les étendues, les positions réseau par seconde, le remplissage et les
catégories de rejet, se trouve dans
`2026-09-15-tt-key-search-results.md`.

| Position | Chemin | legacy | h0 | h1 | h3 | h7 |
|---|---|---:|---:|---:|---:|---:|
| Ouverture | `mcts_search` | 975,3 | 1 051,4 | 1 004,3 | 1 018,2 | 981,7 |
| Ouverture | `step_analysis` | 990,0 | 1 020,5 | 1 011,0 | 1 030,5 | 968,6 |
| Milieu | `mcts_search` | 454,7 | 500,4 | 402,3 | 479,6 | 494,2 |
| Milieu | `step_analysis` | 470,1 | 497,4 | 421,6 | 490,8 | 500,4 |
| Finale | `mcts_search` | 799,7 | 715,3 | 706,6 | 707,3 | 703,4 |
| Finale | `step_analysis` | 783,7 | 692,2 | 676,5 | 712,0 | 699,0 |

Unité : simulations par seconde.

Par rapport à `h1`, `h0` est plus rapide sur les six couples mesurés. L'écart
va d'environ 0,9 % à 24,4 %. Le résultat du milieu de jeu est le moins stable
et ne doit pas être extrapolé seul au jeu réel.

`legacy` est plus rapide dans la finale, mais cette vitesse provient en partie
de hits qui ne sont pas sémantiquement valides. Elle ne constitue donc pas un
avantage acceptable.

## Hits et rejets

Sur la finale isolée, le taux de hit passe de 14,2 % en `legacy` à 9,2 % en
`h0`, 5,7 % en `h1`, 1,4 % en `h3` et 0 % en `h7`. La profondeur historique
réduit donc rapidement la réutilisation du cache.

Les politiques corrigées rejettent environ 3,4 à 3,7 % des consultations de
cette position à cause du compteur des 50 coups. `h1` ajoute environ 3,4 % de
rejets historiques, alors que `h0` n'en ajoute aucun par définition.

Ces résultats soutiennent l'intuition initiale : demander l'égalité de sept
positions antérieures détruit presque tous les hits, alors que le compteur
exact suffit à éliminer la collision qui causait la nulle.

## Conversion de finale

Position : `8/8/8/8/8/2k5/8/R3K3 w - - 0 1`, arbre et TT réutilisés entre les
coups, 700 simulations, batch 8.

| Politique | Issue | Demi-coups | Compteur maximal | Hits TT | Rejets 50 coups | Durée |
|---|---|---:|---:|---:|---:|---:|
| legacy | Nulle des 50 coups | 100 | 100 | 214 822 | 0 | 67,6 s |
| h0 | Mat | 31 | 31 | 12 669 | 7 148 | 17,1 s |
| h1 | Mat | 73 | 73 | 10 025 | 14 209 | 25,6 s |
| h3 | Mat | 45 | 45 | 1 764 | 6 695 | 14,5 s |
| h7 | Mat | 55 | 55 | 5 | 7 359 | 17,2 s |

Le témoin `legacy` reproduit le bug. Tous les modes corrigés le suppriment.
Le grand nombre de rejets liés au compteur montre que le cas n'est pas une
collision hypothétique rare dans cette finale.

## Préfiltre puzzle

Le préfiltre utilise les mêmes 200 lignes du banc pour chaque politique, avec
700 simulations, batch 8 et 16 processus travailleurs. Il mesure uniquement le
premier coup tactique. Aucune erreur de données ou de processus n'a été relevée.

| Politique | Puzzles | Résolus | Réussite | Durée murale approximative |
|---|---:|---:|---:|---:|
| legacy | 200 | 146 | 73,0 % | 0,8 min |
| h0 | 200 | 148 | 74,0 % | 0,8 min |
| h1 | 200 | 148 | 74,0 % | 0,8 min |
| h7 | 200 | 148 | 74,0 % | 0,8 min |

| Comparaison | Gauche seule | Droite seule | Accord des coups | Delta droite | McNemar p |
|---|---:|---:|---:|---:|---:|
| legacy / h0 | 0 | 2 | 98 % | +1,0 point | 0,48 |
| legacy / h1 | 0 | 2 | 99 % | +1,0 point | 0,48 |
| h0 / h1 | 0 | 0 | 99 % | 0,0 point | 1,00 |
| h0 / h7 | 1 | 1 | 98 % | 0,0 point | 1,00 |

L'échantillon est trop petit pour détecter sûrement une différence inférieure
à un point. Il suffit néanmoins au rôle du préfiltre : aucune politique
corrigée ne perd les deux points qui auraient imposé son rejet immédiat.

## Application provisoire de la règle de décision

1. `h0` passe les tests et mate la finale : satisfait.
2. `h0` n'est pas inférieur de plus de 0,5 point à `h1` : satisfait sur le
   préfiltre, égalité parfaite.
3. McNemar ne détecte pas de dégradation de `h0` face à `h1` : satisfait sur le
   préfiltre, `p=1,00`.
4. `h0` ne perd pas plus de 0,5 point face à `legacy` : satisfait sur le
   préfiltre, avantage de 1,0 point.
5. `h0` a un débit ou un taux de hits supérieur à `h1` : satisfait sur le banc
   de débit.

La recommandation provisoire est donc `h0`. Elle est cohérente avec la nature
markovienne des échecs : une fois le plateau, le trait, les droits de roque, la
prise en passant, la répétition courante et le compteur des 50 coups connus,
les positions antérieures ne changent pas la légalité future. Elles changent
cependant l'entrée du réseau, raison pour laquelle la campagne tactique complète
reste nécessaire avant de choisir définitivement `h0`.

## Décision provisoire

Conserver `DEFAULT_CACHE_HISTORY_DEPTH = 1` tant que les 2 500 puzzles appariés
n'ont pas été exécutés. Si la campagne complète confirme les cinq critères,
passer le défaut à `h0`. Sinon conserver `h1`.

Au rythme du préfiltre, les quatre passages complets devraient prendre environ
40 minutes sur cette machine.

## Validations déjà acquises

- 204 tests Python réussis.
- Perft strict réussi sur les six positions de référence du palier rapide.
- Zéro divergence de génération de coups sur 50 000 positions différentielles.
- Zéro violation d'arbre sur 30 combinaisons de politique, batch et position.
- 250 puzzles de smoke, zéro erreur.

## Limites

- La campagne puzzle complète n'a pas encore été exécutée.
- Le banc de débit utilise des positions isolées et une TT de 8 192 entrées. En
  partie réelle, la TT est plus grande et réutilisée après le coup adverse, ce
  qui peut produire davantage de hits, particulièrement en finale.
- Une seule trajectoire tour contre roi a été jouée par politique.
- Aucun tournoi Elo apparié n'a encore comparé `h0` à `h1`.
- La dérive thermique et le bruit GPU ne sont pas complètement éliminés par
  sept passages.
