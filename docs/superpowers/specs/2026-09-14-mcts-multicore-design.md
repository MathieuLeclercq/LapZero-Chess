# Recherche MCTS multicœur : conception

Date initiale : 2026-09-14

Révision : 2026-09-16

Statut : spec révisée, prête pour revue

## 1. But et décision d'architecture

La recherche UCI regroupe déjà jusqu'à 8 feuilles dans une seule inférence GPU, mais un
seul cœur CPU effectue toutes les descentes. Le GPU attend donc que ce cœur ait préparé le
lot, puis le cœur attend le GPU. Le but est de faire préparer les descentes par plusieurs
cœurs sur un arbre partagé, sans corruption silencieuse et sans baisse mesurable de la
qualité de recherche à nombre de simulations égal.

Le chantier sera livré en deux étapes compatibles :

1. **Recherche par vagues**, retenue pour la première implémentation. Des workers CPU
   persistants collectent les feuilles en parallèle. Ils s'arrêtent à une barrière, puis
   le coordinateur effectue l'inférence, les expansions et les backups. Cette étape
   valide les plateaux privés, la sélection concurrente, le virtual loss, la table de
   transposition et les invariants, sans faire se chevaucher lecture et écriture de
   l'arbre.
2. **Production continue**, envisagée seulement après validation et mesure des vagues.
   Les workers continueraient à descendre pendant les inférences et les backups. Elle
   réutilisera les abstractions de la première étape, mais demandera une synchronisation
   supplémentaire des statistiques et de la structure de l'arbre. Elle aura son propre
   plan d'implémentation et son propre critère d'acceptation.

Le premier réglage étudié est 8 workers CPU avec un lot GPU maximal de 8. Les deux valeurs
restent paramétrables. Le chemin actuel à un worker reste disponible comme référence et
comme solution de repli.

## 2. État actuel vérifié

Le batching UCI fournit déjà les fondations suivantes :

- `n_in_flight` est un virtual loss de type LC0, séparé de `visit_count` et de Q ;
- une feuille collectée capture son tenseur, ses coups légaux et sa clé de cache avant que
  le plateau soit ramené à la racine ;
- `step_analysis()` protège l'appel complet contre `update_root()` ;
- `inspect_tree()` vérifie notamment les enfants dupliqués, les parents, les priors, les
  visites et l'annulation du virtual loss ;
- les compteurs distinguent les appels réseau, les positions réseau, la table de
  transposition et les terminaux ;
- `search_bench.py` mesure `mcts_search()` et `step_analysis()` sur le modèle ONNX réel ;
- le banc de puzzles permet une comparaison appariée à nombre de simulations égal.

La table de transposition utilise désormais une clé sémantique. Le réglage de référence
est `cache_history_depth = 0`, nommé h0. Il protège notamment le compteur exact des 50
coups, la catégorie de répétition courante et le contexte courant. Toutes les mesures du
multicœur utiliseront h0, sauf expérience explicitement séparée.

Le code actuel n'est pas thread-safe sur un arbre partagé : `visit_count`, `total_value`,
`n_in_flight`, `is_terminal`, `children` et les entrées de table sont lus ou modifiés sans
synchronisation adaptée à ce cas.

## 3. Contraintes imposées par le code

### 3.1 Un plateau complet et privé par worker

`Chessboard` ne peut pas être partagé entre workers, même pour une opération présentée
comme une lecture. `getLegalMovesForSquare()` passe par `isMoveSafe()`, qui déplace
temporairement les pièces et le roi avant de restaurer l'état.

En outre, `movePiece()` et `undoMove()` manipulent le trait, le roque, la prise en passant,
les compteurs, le hash et les trois historiques. `getAlphaZeroTensor()` lit ces historiques
pour les 8 snapshots et les plans de répétition. Une reconstruction depuis la seule FEN
courante serait donc incorrecte.

Chaque worker possède un `Chessboard` complet, copié depuis le plateau racine au début de
l'appel. Il descend et annule les coups uniquement sur cette copie. Le plateau fourni à
`step_analysis()` ne doit jamais être modifié par un worker.

### 3.2 Un arbre unique

Les workers partagent le même arbre afin de conserver la mutualisation des statistiques,
la table de transposition et la réutilisation après `update_root()`. Les variantes avec un
arbre par cœur sont écartées : elles dupliquent les évaluations, diluent la profondeur et
compliquent fortement la réutilisation de racine.

### 3.3 Le self-play n'est pas un modèle direct

`SelfPlayManager` utilise déjà plusieurs threads OpenMP, mais sur des parties et des
racines distinctes. Sa phase parallèle lit la table, puis les écritures et expansions ont
lieu dans une phase séquentielle. Cela confirme l'intérêt des plateaux privés et du GPU
groupé, mais pas la sûreté des fonctions MCTS actuelles sur une racine partagée.

Le nouvel ordonnanceur concerne la recherche UCI. L'abstraction thread-safe de la table
sera toutefois commune au self-play, qui doit conserver sa correction et ses performances.

## 4. Composants communs aux deux étapes

### 4.1 Exécuteur de workers persistant

Le `MCTS` possède paresseusement un petit exécuteur CPU :

- les threads sont créés au premier appel multicœur ;
- ils dorment sur une variable de condition entre les recherches ;
- chaque recherche active seulement le nombre demandé de workers ;
- ils sont joints dans le destructeur de `MCTS`, pas à chaque coup ;
- chaque worker conserve un contexte réutilisable avec son `Chessboard` et ses buffers.

Cette durée de vie évite que la création de 8 threads pollue chaque mesure et chaque coup.
`worker_count == 1` contourne entièrement l'exécuteur et conserve le noyau batché actuel.

### 4.2 Résultat de collecte immuable

Une descente produit un objet déplaçable, sans pointeur vers le plateau local, contenant au
minimum :

- le nœud réservé et le chemin exact qui porte le virtual loss ;
- la nature du résultat : terminal, vraie évaluation GPU ou collision ;
- la clé h0 et les coups légaux quand ils sont nécessaires ;
- le tenseur uniquement pour une vraie évaluation GPU ;
- la valeur terminale lorsqu'elle est déjà connue.

Le résultat transfère au coordinateur la responsabilité de terminer ou d'annuler la
réservation exactement une fois.

### 4.3 Évaluateur interchangeable

Le MCTS dépendra d'une petite interface d'évaluation par lot, implémentée par
`ONNXEvaluator`. Une implémentation déterministe réservée aux tests pourra :

- renvoyer une policy et une value fixes et reproductibles ;
- enregistrer le nombre d'appels et la taille de chaque lot ;
- attendre sur une barrière ou introduire un délai contrôlé ;
- lever une exception à un appel choisi.

Le coût d'un appel virtuel par lot est négligeable devant une inférence. Cette interface
permet de provoquer les interleavings difficiles sans dépendre du GPU ni du timing de la
machine.

### 4.4 Garde de réservation RAII

Une garde déplaçable possède le chemin réservé. Tant qu'elle est active, chaque nœud du
chemin porte une unité de `n_in_flight`. Sa destruction annule automatiquement ces unités.
Le coordinateur la désarme seulement après avoir effectué le backup correspondant.

Cette garde couvre les collisions, les terminaux, les hits TT, les exceptions, les arrêts
et les lots partiels. Aucun chemin d'erreur ne doit dépendre d'une boucle manuelle oubliée.

### 4.5 Surface publique et activation UCI

`mcts_search()` et `step_analysis()` reçoivent `worker_count` après `batch_size`, avec une
valeur par défaut de 1 afin de préserver tous les appelants Python actuels. Une valeur
inférieure à 1 est refusée. Le mode multicœur demande un `batch_size` strictement positif.

Les bindings exposent le paramètre sans changer les valeurs par défaut existantes. Le
réglage UCI reste à 1 worker pendant le développement. Après validation seulement,
`MCTS_WORKER_COUNT` est placé à côté de `MCTS_BATCH_SIZE` dans `uci.py`, avec 8 comme
premier candidat mesuré et non comme valeur imposée d'avance.

## 5. Étape 1 : recherche multicœur par vagues

### 5.1 Déroulement d'une vague

Pour chaque vague :

1. le coordinateur crée au plus `min(batch_size, simulations_restantes)` permis globaux
   de collecte ;
2. les workers actifs descendent en parallèle avec leurs plateaux privés ;
3. un worker peut collecter plusieurs résultats successifs, afin que 2 ou 4 workers
   puissent tout de même remplir un lot de 8 ;
4. les workers déposent leurs résultats dans des vecteurs locaux, puis atteignent la
   barrière ;
5. quand tous les collecteurs sont arrêtés, le coordinateur fusionne les résultats dans
   un ordre stable, traite les terminaux, lance au plus une inférence groupée,
   développe les feuilles et effectue tous les backups ;
6. la vague suivante ne commence qu'après publication complète de ces modifications.

Le nombre de tentatives par vague est borné. Une collision ne compte pas comme simulation
terminée, rend son permis et libère sa réservation. Un hit TT conserve le même permis
pendant la poursuite de la descente. Un terminal ou une vraie feuille GPU consomme le
permis. Il ne peut donc jamais y avoir plus de feuilles GPU que la capacité du lot, ni plus
de résultats finaux que le budget restant. Si les collisions empêchent de remplir le lot,
le coordinateur exécute un lot partiel afin de garantir le progrès.

### 5.2 Synchronisation minimale en phase 1

Pendant la collecte d'une vague, aucun worker ne modifie `visit_count`, `total_value` ou
un vecteur `children` déjà publié. Ces données sont donc stables jusqu'à la barrière. Seuls
les éléments suivants sont concurrents :

- `n_in_flight`, qui devient atomique ;
- l'état de feuille `unexpanded`, `pending`, `expanded` ou `terminal`, publié avec une
  transition atomique ;
- les compteurs d'observabilité, déjà atomiques ;
- la table de transposition, protégée séparément.

Un worker qui rencontre une feuille tente `unexpanded -> pending` avant le probe TT. Un
seul gagne la propriété. Les autres déclarent une collision, libèrent leur chemin et
recommencent tant que la vague autorise une tentative.

Sur un hit TT, le propriétaire copie le snapshot protégé, construit les enfants dans le
nœud encore invisible, publie `expanded` avec une écriture release, puis continue la même
descente. Le hit ne consomme pas une simulation et ne provoque pas de backup à lui seul,
ce qui préserve la sémantique de `select_leaf()` actuelle. Les autres workers ne lisent
`children` qu'après avoir observé `expanded` avec une lecture acquire.

Sur un miss TT nécessitant le GPU, le nœud reste `pending` jusqu'au traitement du lot par
le coordinateur. Celui-ci est le seul à créer les enfants issus du réseau et publie ensuite
`expanded`.

Cette séparation évite un mutex par nœud et évite aussi les lectures incohérentes de
`visit_count` et `total_value` pendant la première implémentation.

### 5.3 Virtual loss pendant la descente

Le virtual loss doit devenir visible pendant la sélection, pas après l'arrivée à la
feuille comme dans la boucle mono-thread actuelle :

1. la racine est ajoutée à la garde ;
2. le worker calcule les scores avec les `n_in_flight` déjà publiés ;
3. l'enfant choisi est immédiatement ajouté à la garde avant de poursuivre ;
4. le processus se répète jusqu'au résultat de collecte.

Deux workers peuvent encore sélectionner presque simultanément le même enfant. Le virtual
loss réduit cette probabilité, mais il ne constitue pas la garantie de correction. La
transition exclusive vers `pending` garantit qu'une même feuille n'est pas développée
deux fois.

### 5.4 Budget exact de simulations

Un ticket de simulation n'est consommé qu'au moment où un résultat terminal ou GPU est
effectivement backupé. Un hit TT qui permet de poursuivre la descente, une collision ou
une exception ne consomme pas de ticket. Le coordinateur arrête la dernière vague au
budget demandé, même si moins de positions que la taille maximale du lot restent à
évaluer.

À la fin d'un appel réussi, le nombre de backups nouveaux est exactement égal au nombre
de simulations demandé.

## 6. Étape 2 : production continue

Cette étape n'est pas incluse dans le premier plan d'implémentation. Elle devient utile si
les chronométrages montrent que la barrière laisse encore le CPU ou le GPU inactif de
façon significative.

Le modèle cible est une file centrale d'inférence : les workers produisent en continu,
le service GPU regroupe les requêtes disponibles, puis les résultats sont développés et
backupés pendant que d'autres descentes avancent. Le nombre de workers peut alors dépasser
la taille de lot pour masquer la latence.

Les composants de l'étape 1 sont conservés : contextes privés, résultats immuables, garde
RAII, propriété `pending`, table thread-safe et évaluateur contrôlable. La différence
importante est que les statistiques et `children` peuvent désormais être lus pendant un
backup ou une expansion.

Le mécanisme de synchronisation des statistiques ne sera pas choisi avant les mesures de
l'étape 1. Trois solutions seront comparées sur un microbenchmark ciblé :

- atomiques pour N et W, avec protocole garantissant un snapshot cohérent de Q ;
- petit verrou par nœud ;
- verrous rayés indexés par l'adresse stable du nœud.

Le critère est la correction sans data race, puis le débit sous forte contention à la
racine et la mémoire par nœud. Un verrou utilisé seulement par les writers est interdit :
les readers doivent participer au même protocole. Le choix retenu sera consigné avant le
plan de l'étape 2.

## 7. Table de transposition partagée

La table directe reste partagée et conserve la politique h0. Les accès concurrents passent
par une abstraction de cache qui ne retourne jamais un `TTEntry*` brut.

Pour une clé donnée :

```text
tt_index = key.position_hash % tt_size
stripe   = tt_index % stripe_count
```

Le verrou de la bande est donc dérivé de l'index physique réellement accédé, pas seulement
d'un second modulo indépendant sur le hash.

Un probe prend le verrou, valide tous les champs de contexte, copie la value et seulement
les entrées de policy légale utiles dans un snapshot local, puis libère le verrou. Un store
prépare ses données, prend le même verrou et remplace l'entrée. Aucun enfant n'est alloué
et aucun backup n'est effectué sous ce verrou.

Le même cache sert au self-play. Les tests doivent donc couvrir ses accès OpenMP existants
et vérifier qu'une position identique avec deux compteurs des 50 coups différents ne peut
ni produire un faux hit, ni observer une policy partiellement écrite.

Deux nœuds logiques différents peuvent légitimement demander en même temps la même
évaluation. Dédupliquer ces requêtes n'est pas un objectif de ce chantier. En revanche, un
même nœud ne peut être complété ou développé deux fois pour une réservation donnée.

## 8. Instrumentation des performances

Le chronométrage est ajouté avant le parallélisme. Il est optionnel, désactivé par défaut
et accumulé localement par thread, puis agrégé après la recherche. Il ne doit pas ajouter
un atomic ou un verrou à chaque nœud lorsque la mesure est désactivée.

Les phases mesurées sont :

- sélection et application des coups ;
- préparation du tenseur et de la clé ;
- probe/store TT et attente de verrou TT ;
- attente des workers et barrière ;
- assemblage du lot ;
- inférence ONNX ;
- expansion ;
- backup ;
- collisions et travail abandonné.

Le rapport distingue le temps mur, la somme des temps CPU, le temps d'inférence et les
temps d'attente. Ces grandeurs ne doivent pas être additionnées comme si elles étaient
séquentielles, surtout lors de la future production continue.

Le surcoût de l'instrumentation désactivée est mesuré. Il doit rester dans le bruit du banc
de référence, avec une limite d'acceptation de 2 % sur la médiane.

## 9. Invariants, erreurs et cycle de vie

Après toute recherche réussie :

- tous les `n_in_flight` sont nuls ;
- aucun nœud ne reste `pending` ;
- chaque réservation est terminée ou annulée exactement une fois ;
- chaque simulation demandée produit exactement un backup ;
- aucun nœud n'est développé deux fois ;
- les enfants sont uniques, leurs parents sont corrects et leurs priors somment à 1 ;
- le plateau d'entrée conserve son FEN, son hash, ses historiques et son tenseur ;
- le coup principal est légal sur le plateau racine.

Si l'évaluateur, un worker ou le coordinateur lève une exception, la session :

1. publie l'arrêt ;
2. réveille tous les workers ;
3. annule toutes les gardes encore actives ;
4. attend que les workers reviennent au repos ;
5. restaure les états `pending` non publiés ;
6. relaie l'exception.

Le verrou UCI existant reste détenu pendant `step_analysis()`. Aucun `update_root()` ni
destruction de racine ne peut avoir lieu tant que les workers possèdent des pointeurs vers
l'arbre.

## 10. Stratégie de test

### 10.1 Socle déterministe

- Vérifier qu'une copie complète de `Chessboard` conserve FEN, hash, coups légaux,
  historiques et tensor, puis que jouer et annuler sur la copie ne modifie pas l'original.
- Verrouiller le comportement `worker_count == 1` sur le noyau batché actuel.
- Tester séparément la garde RAII, y compris move, annulation, désarmement et exception.
- Tester les transitions exclusives de l'état de feuille.
- Tester le cache thread-safe avec deux contextes de règle des 50 coups, lecture pendant
  écriture forcée et forte collision sur une même bande.

### 10.2 Évaluateur contrôlé et interleavings

Avec l'évaluateur déterministe :

- imposer les tailles de lots attendues pour 1, 2, 4 et 8 workers ;
- vérifier qu'un hit TT matérialise les enfants, poursuit la descente et ne consomme pas
  une simulation à lui seul ;
- bloquer une évaluation sur une barrière pendant que plusieurs workers atteignent la
  même branche ;
- provoquer une exception au milieu d'un lot et vérifier zéro `n_in_flight` et zéro
  `pending` ;
- exécuter plusieurs recherches, `update_root()`, un coup adverse et une nouvelle
  recherche ;
- vérifier le budget exact de simulations malgré terminaux, hits TT, collisions et lot
  final partiel.

Les tests n'exigent pas qu'une même position logique ne soit jamais évaluée deux fois dans
des nœuds distincts. Ils exigent qu'une réservation de nœud ne soit jamais terminée deux
fois.

### 10.3 Stress et invariants

Les campagnes couvrent 2, 4 et 8 workers, batch 1 et 8, les positions de
`search_bench.py`, et au moins cent recherches courtes par configuration. Après chaque
recherche, `inspect_tree()` doit être propre et le coup principal légal.

MSVC et la chaîne CMake actuelle ne fournissent pas ThreadSanitizer. Les tests
déterministes et le stress répété sont donc obligatoires sous Windows. Une campagne
optionnelle clang/ThreadSanitizer pourra renforcer la validation, sans bloquer l'étape 1.

Une exécution multicœur ne doit pas être identique bit à bit au mono-worker : l'ordre des
réservations change naturellement l'arbre. La correction repose sur les invariants, le
budget exact et le banc de qualité, pas sur l'identité de la policy finale.

## 11. Mesure du gain

### 11.1 Référence et débit

Avant la modification fonctionnelle, le rapport de référence est produit avec le modèle
ONNX réel, le même GPU, h0, la même taille de table, cinq passages et les trois positions
du banc. Il mesure séparément `mcts_search()` et `step_analysis()`.

Le même protocole compare ensuite :

| Configuration               | Rôle                                          |
| --------------------------- | ---------------------------------------------- |
| 1 worker, batch 8           | référence actuelle                           |
| 2 workers, batch 8          | premier effet de concurrence                   |
| 4 workers, batch 8          | montée en charge                              |
| 8 workers, batch 8          | cible initiale                                 |
| 16 workers, batch 8 puis 16 | exploratoire, seulement si 8 reste CPU-limité |

Chaque worker peut produire plusieurs feuilles par vague. La taille maximale du lot n'est
donc pas limitée par le nombre de workers.

Le rapport contient : médiane, minimum, maximum, simulations/s, positions réseau/s,
appels GPU/s, remplissage moyen, hits TT, collisions, temps par phase, temps d'attente et
utilisation CPU/GPU observée. Après échauffement, chaque configuration exécute au moins 30
appels chronométrés par position. L'acceptation demande une amélioration reproductible de
la médiane et aucune hausse supérieure à 5 % du p95 de latence par rapport au mono-worker.

Le banc contrôlé sous-estime les hits TT des parties réelles, surtout après réutilisation
de racine en finale. Cette limite est documentée. Elle ne justifie pas de remplacer le
banc isolé par de longues parties, plus coûteuses et moins reproductibles.

### 11.2 Qualité de recherche

La qualité est comparée d'abord à nombre de simulations égal, sans bruit de Dirichlet et
avec le même modèle. Le banc de puzzles utilise les mêmes puzzles appariés pour 1 et 8
workers.

Le critère est une non-infériorité explicite : la borne basse de l'intervalle de confiance
à 95 % sur la différence appariée du taux de résolution ne doit pas passer sous -1 point
de pourcentage. L'intervalle est obtenu par bootstrap apparié déterministe. Le test exact
de McNemar et les paires discordantes restent rapportés comme diagnostics, mais une
p-value supérieure à 0,05 n'est pas considérée comme une preuve d'équivalence.

Une première passe stratifiée de 500 puzzles sert de barrière rapide. La campagne de 2500
puzzles est réservée au candidat qui franchit cette barrière et tous les invariants.

Après validation à simulations égales, une mesure à temps égal quantifie le gain pratique
en simulations supplémentaires et en taux de résolution.

## 12. Phasage et critères de sortie

### Étape 0 : observabilité

- ajouter les chronométrages désactivables ;
- produire la référence mono-worker h0 ;
- vérifier un surcoût désactivé inférieur ou égal à 2 %.

### Étape 1A : fondations testables

- évaluateur interchangeable et faux évaluateur ;
- cache TT thread-safe par snapshots ;
- états de feuille et garde RAII ;
- exécuteur persistant et contextes de worker.

Chaque composant passe ses tests isolés avant intégration.

### Étape 1B : vagues multicœurs

- collecte parallèle et traitement séquentiel après barrière ;
- campagnes d'invariants et d'exceptions ;
- comparaison 1/2/4/8 workers ;
- banc de puzzles à simulations égales, puis à temps égal.

La recherche par vagues peut devenir le mode UCI par défaut seulement si elle est plus
rapide sur le banc réel, respecte tous les invariants et franchit la non-infériorité.
Sinon, le mode mono-worker reste le défaut.

### Étape 2 : décision sur la production continue

Les temps par phase déterminent si le chevauchement restant justifie le chantier. Si oui,
la synchronisation des statistiques est sélectionnée par microbenchmark, documentée, puis
fait l'objet d'un plan séparé. L'étape 1 n'est pas jetée : tous ses composants hors
barrière sont réutilisés.

## 13. Hors périmètre

- paralléliser davantage le self-play ;
- modifier les 119 plans, l'encodage des coups ou les règles du plateau ;
- changer la politique h0 choisie pour la table ;
- dédupliquer les évaluations de positions identiques dans des nœuds distincts ;
- régler `c_puct` ou rechercher la force maximale avant stabilisation de la correction ;
- optimiser le softmax sur les seuls coups légaux, optimisation indépendante consignée
  dans `docs/backlog.md` ;
- remplacer l'arbre partagé par plusieurs arbres indépendants.

## 14. Références étudiées

- Leela Chess Zero 0.32.1 et branche master au commit
  `043e7233e115b1ff44f892513b91b6fc1fa49eb6` : collecte par minibatches, compteur
  `n_in_flight`, workers de recherche et agrégation des requêtes réseau.
- KataGo au commit `76369bf535b5fe9310ef93fdfd3f108b93a44e7f` : plateaux privés par thread,
  virtual loss posé avant la descente, file centrale d'évaluation et verrous séparant
  statistiques et structure.
- Stockfish : table très compacte acceptant des snapshots volontairement relâchés. Cette
  stratégie n'est pas reprise ici, car une entrée de policy LapZero est bien plus grande
  et ne peut pas être lue de façon partiellement écrite.

Ces références orientent l'architecture, mais ne remplacent pas les tests propres à la
représentation, à la TT h0 et au cycle de vie UCI de LapZero.
