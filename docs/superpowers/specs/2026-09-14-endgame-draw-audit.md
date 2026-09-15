# Audit des nulles évitables en finale

Date : 14 septembre 2026

## Symptôme

Le moteur peut atteindre la règle des 50 coups dans des finales matériellement
gagnantes, notamment roi et tour contre roi. Le comportement est observé avec le
modèle `2026_04_23_23h25_iter316_unsupervised`.

## Conclusion

Le compteur des 50 coups, son annulation et les détections terminales du MCTS sont
corrects dans les cas testés. Le batching n'est pas nécessaire pour reproduire la
nulle.

Le défaut moteur identifié est la clé de la table de transpositions. Elle utilise le
Zobrist de la position courante, qui contient les pièces, le trait, les droits de
roque et la prise en passant. Elle ne contient pas :

- le compteur des 50 coups ;
- les huit positions d'historique du tensor ;
- les plans de répétition ;
- le nombre total de coups ;
- le mode d'amnésie.

Ces données modifient pourtant les 119 plans vus par le réseau. La TT peut donc rendre
la value et la policy calculées pour un tensor différent. C'est une collision
sémantique, même lorsque le hash Zobrist lui-même ne collisionne pas.

La limitation était déjà documentée dans la spec du batching et existait avant son
implémentation. Le chemin séquentiel historique reproduit la nulle.

## Preuves automatisées

`python_src/tests/test_endgame_rules.py` vérifie :

- nulle exactement au centième demi-coup calme ;
- remise à zéro après un coup de pion et après une capture ;
- restauration correcte après `undoMove()` ;
- priorité du mat sur la règle des 50 coups ;
- détection de la troisième occurrence ;
- même détection terminale dans le MCTS séquentiel et batché ;
- présence effective du compteur no-progress dans le tensor ;
- test de régression actuellement marqué `xfail` pour la clé de TT incomplète.

Résultat ciblé avant correction : `9 passed, 1 xfailed`.

## Mesures avec le modèle réel

Toutes les recherches utilisent 700 simulations par coup, sauf la mesure de contrôle
à 100 simulations. La TT du diagnostic contient 131 071 entrées.

| Position et mode | Résultat | Longueur | Durée |
|---|---:|---:|---:|
| Tour contre roi, batch 8, arbre et TT réutilisés | nulle 50 coups | 100 plies | 197,4 s |
| Tour contre roi, batch 0, arbre et TT réutilisés | nulle 50 coups | 100 plies | 208,1 s |
| Tour contre roi, batch 8, arbre supprimé mais TT gardée | nulle 50 coups | 100 plies | 93,9 s |
| Tour contre roi, batch 8, arbre et TT recréés à chaque coup | mat | 43 plies | 28,9 s |
| Dame et pion contre fou, batch 8, arbre et TT réutilisés | mat | 15 plies | 8,6 s |
| Tour contre roi, batch 8, 100 simulations, réutilisation | mat | 47 plies | 6,0 s |
| Tour contre roi, batch 0, 100 simulations, réutilisation | mat | 57 plies | 19,5 s |

Le mode sans arbre mais avec TT compte 306 178 hits sur 354 967 consultations. Le
mode séquentiel avec réutilisation compte 211 983 hits sur 266 524 consultations.
Une simulation peut consulter plusieurs entrées au cours de sa descente.

Sur la même position tour contre roi :

- compteur 0 : value réseau `+0,642951` ;
- compteur 99 : value réseau `-0,044066` ;
- écart : `-0,687017`.

Les tensors diffèrent uniquement sur les 64 valeurs du plan no-progress. Le réseau
réagit donc fortement au compteur, mais cette information peut être contournée par la
TT.

## Interprétation

Le modèle sait convertir les deux finales testées et comprend l'urgence du compteur.
Il peut néanmoins encore être imparfait en finale : ses cibles de value distinguent
gain, nulle et perte, sans récompenser directement un mat plus court. Cette faiblesse
potentielle ne suffit pas à expliquer les résultats ci-dessus, puisque retirer la TT
persistante rétablit le mat dans la reproduction principale.

La TT de l'UCI réel contient 4 millions d'entrées. Une table plus grande réduit les
collisions d'index entre hashes différents, mais peut conserver plus longtemps les
entrées sémantiquement périmées ayant exactement le même Zobrist.

## Correction recommandée

La conception détaillée et les variantes de profondeur historique sont consignées dans
`2026-09-15-tt-evaluation-key-design.md`. Elle affine la recommandation ci-dessous : le
compteur des 50 coups doit être exact, tandis que l'étendue historique de la clé sera
choisie par benchmark entre 0, 1, 3 et 7 positions antérieures.

Conserver le Zobrist actuel pour les règles et ajouter une clé distincte pour le cache
réseau. Cette clé doit représenter exactement les informations qui déterminent le
tensor dans son mode de référence. Des modes moins stricts sont volontairement testés,
car deux historiques différents peuvent produire le même plateau courant et des tensors
différents sans que cette différence soit nécessairement utile au niveau de jeu.

La correction devra être validée ainsi :

1. Faire passer le test `xfail` sans modifier son assertion.
2. Ajouter des cas avec historique différent, répétition différente et amnésie.
3. Relancer les invariants MCTS, les tests batch 0 contre batch 8 et le banc puzzle.
4. Rejouer la finale tour contre roi à 700 simulations avec réutilisation normale.
5. Mesurer le coût de la nouvelle clé et le taux de hits avant et après.

Le script `python_src/dev_tools/endgame_conversion.py` reproduit les parties avec les
modes `reuse`, `reset` et `fresh`. `python-chess` y sert seulement d'oracle de légalité
et de synchronisation.

Les 48 conversions de finales et la comparaison appariée des 2 500 puzzles
sont consignées dans `2026-09-15-tt-key-endgame-results.md` et
`2026-09-15-tt-key-benchmark-results.md`. Les trois politiques corrigées
matent la position tour contre roi causale, tandis que `legacy` atteint la
nulle des 50 coups.
