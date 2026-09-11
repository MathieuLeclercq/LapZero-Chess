# Banc de recherche : resultats

Modele : `2026_04_23_23h25_iter316_unsupervised.onnx`, iteration 316, global_step 19415
Protocole : 5 passages, 400 simulations, c_puct 1.4, CPU, un seul processus

Trois grandeurs distinctes. Les **simulations par seconde** mesurent le
debit de la recherche. Les **inferences par seconde** comptent les appels
reels au reseau : une simulation qui s'arrete sur un noeud terminal ou
sur un succes de table n'en coute aucune. Le **taux de table** est la
part des consultations reussies.

## Debit

| Position | Chemin | passages | sims/s (med) | sims/s (min a max) | inferences/s (med) | taux table |
|---|---|---|---|---|---|---|
| finale | mcts_search | 5 | 292.2 | 278.8 a 302.0 | 292.9 | 10.7 % |
| finale | step_analysis | 5 | 299.1 | 272.9 a 304.6 | 299.9 | 10.7 % |
| milieu | mcts_search | 5 | 292.6 | 290.1 a 298.5 | 293.3 | 3.6 % |
| milieu | step_analysis | 5 | 289.1 | 267.9 a 302.2 | 289.8 | 3.6 % |
| ouverture | mcts_search | 5 | 286.0 | 276.1 a 300.5 | 286.8 | 2.7 % |
| ouverture | step_analysis | 5 | 292.8 | 278.2 a 298.9 | 293.6 | 2.7 % |

## Invariants d'arbre

| Position | noeuds | profondeur max | violations |
|---|---|---|---|
| ouverture | 13049 | 17 | 0 |
| milieu | 16682 | 19 | 0 |
| finale | 7855 | 15 | 0 |

Aucune violation.

Les deux jambes sont lancees separement : inspect_tree parcourt tout l'arbre,
donc son cout croit avec le nombre de simulations et il fausserait la mesure de
debit qu'il est cense proteger.

## Controles de coherence

- Rapport inferences sur simulations : 1,0024 a 1,0028 partout, pour un attendu
  de 401/400 = 1,0025. Le + 1 est l'expansion de la racine, que step_analysis et
  mcts_search font hors de la boucle de simulations. Chaque defaut de table
  declenche donc exactement une inference, et aucune n'est perdue ni comptee
  deux fois.
- Taux de table strictement compris entre 0 et 100 pour cent sur les six lignes,
  donc les compteurs sont bien branches.
- Etendue relative a la mediane : de 2,9 a 11,9 pour cent. La machine etait
  calme, la mesure sert de reference. Un gain inferieur a 12 pour cent apres le
  batching ne serait pas distinguable du bruit sur un seul passage.
- Zero violation d'invariant sur les trois positions.

## Lecture

Le debit est remarquablement homogene, de 286 a 299 simulations par seconde,
quels que soient la position et le chemin. C'est la signature d'un cout domine
par la latence d'inference a batch 1 : ni le nombre de coups legaux (14 en
finale, 48 au milieu) ni la profondeur de l'arbre n'y changent grand chose.

Le taux de succes de la table est faible, de 2,7 a 10,7 pour cent. Presque
chaque simulation paie une inference complete, donc il n'existe pas de reservoir
de simulations gratuites derriere lequel un gain de batching pourrait se diluer.
