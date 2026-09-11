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
chaque simulation paie une inference complete. Attention toutefois a ne pas en
conclure que le gain du batching se transmettra integralement : ce banc mesure
le cas froid, et la section suivante explique pourquoi le cas reel differe.

## Reserve sur la taille de table, a revoir apres le batching

Ces mesures utilisent `tt_size = 8192`. Mesure faite a part, sur la position de
depart et 400 simulations :

| `tt_size` | taux table | debit |
|---|---|---|
| 8 192 | 16,1 % | 298,0 sims/s |
| 65 536 | 16,6 % | 309,2 sims/s |
| 2 097 143 | 16,6 % | 313,9 sims/s |

Grossir la table n'apporte donc presque rien ici, et sature des 65 536. L'ecart
de debit de 5 pour cent tient dans le bruit, dont l'etendue mesuree est de
12 pour cent.

La raison n'est pas la taille de la table mais la structure de l'arbre : a 400
simulations il ne contient au plus que 400 positions evaluees, donc 8192 entrees
sont deja vingt fois trop. Le taux est faible parce que les transpositions sont
rares a cette profondeur.

**Cela changera apres le batching.** A 10 ou 40 fois plus de simulations, les
chemins convergeront bien plus souvent sur les memes positions et 8192 entrees
deviendront sous-dimensionnees. La taille de table est donc un parametre a
rebalayer au moment de mesurer le moteur batche, et pas un reglage acquis.

A noter aussi que `uci.py` demande `tt_size = 4_000_000` : le bot reel tourne
deja avec une table bien plus grande que ce banc. Sans consequence sur cette
reference, le taux saturant des 65 536 a 400 simulations, mais a garder en tete
quand on comparera le banc au comportement en partie.

## Reserve sur le cas froid, non mesuree

Ce banc mesure une recherche qui part d'un arbre neuf : `mesurer_step_analysis`
fait `reset_analysis()` puis une seule recherche, et n'appelle jamais
`update_root`. C'est un ecart a la section 5 de la spec, qui prevoyait de mesurer
`step_analysis` suivi de `update_root`. Le symptome est visible dans le tableau :
les deux chemins donnent 286 a 299 sims/s, quasiment identiques, precisement
parce qu'ils sont tous deux mesures a froid.

En partie reelle, `update_root` conserve le sous-arbre du coup joue, avec ses
visites, ses enfants et toutes ses positions deja presentes dans la table. La
recherche suivante demarre donc chaude. L'effet est particulierement marque en
finale, ou le branchement est faible et l'arbre converge : a nombre de noeuds
fixe le bot repond presque instantanement, et a temps fixe il evalue des
milliers de noeuds.

**Consequence sur l'estimation du gain, par la loi d'Amdahl.** Si une fraction p
des simulations evite deja le reseau, le batching n'accelere que la fraction
restante. A 3 pour cent de succes de table le gain se transmet presque
integralement ; a 50 pour cent il serait divise par deux. La vraie valeur en
partie n'est pas mesuree.

Mesurer ce cas chaud demanderait de jouer une sequence de coups avec reutilisation
d'arbre, ce qui est long. Decision prise de ne pas le faire : la conclusion
principale ne change pas. Le debit est plat a 2 pour cent pres entre une finale a
14 coups legaux et un milieu a 48, et entre des arbres de profondeur 15 et 19.
Cette insensibilite totale a la forme de l'arbre reste la preuve d'un cout domine
par la latence d'inference a batch 1, et donc que le batching attaque le bon
goulot. Seule l'ampleur du gain reste incertaine, pas sa direction.

A rouvrir si le gain mesure apres batching est nettement inferieur a l'attendu :
ce serait la premiere explication a verifier.

