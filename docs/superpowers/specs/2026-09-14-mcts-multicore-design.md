# Recherche MCTS multicœur : conception

Date : 2026-09-14

Statut : spec prête pour revue

## 1. But

La recherche UCI actuelle regroupe déjà plusieurs feuilles pour une seule inférence GPU.
Elle reste toutefois séquentielle : un seul cœur CPU descend dans l'arbre, prépare les
tenseurs et attend l'inférence. Le but est de faire travailler plusieurs cœurs sur le
même arbre afin de remplir plus vite les lots GPU et d'augmenter le nombre de simulations
par seconde, sans dégrader la correction de l'arbre ni la qualité à nombre de simulations
identique.

Le premier réglage cible est 8 workers CPU et un lot GPU de 8. Les deux valeurs restent
paramétrables par appel. Le chemin actuel à un worker reste disponible comme référence et
comme solution de repli.

## 2. Ce qui est déjà acquis

Le chantier du batching a déjà apporté les fondations utiles :

- `n_in_flight`, le virtual loss de type LC0, est séparé de `visit_count` et de Q.
- Les feuilles d'un lot capturent déjà le tenseur, les coups légaux et le hash avant que
  le plateau ne soit remis à la racine.
- `step_analysis` protège la durée entière de l'appel contre `update_root`.
- `inspect_tree()` détecte les enfants dupliqués, les pointeurs parents incohérents, les
  prior incohérents, les visites incohérentes et un virtual loss non annulé.
- Les compteurs séparent appels réseau, positions réseau, succès et échecs de table, et
  terminaux.
- `search_bench.py` mesure `mcts_search` et `step_analysis` sur trois positions, sur
  plusieurs passages, avec le modèle GPU réel.
- Le banc de puzzles compare les sorties appariées à simulations égales.

Ce travail ne rend pas le MCTS courant thread-safe. Les champs de `MCTSNode`, le vecteur
`children` et les entrées de la table sont aujourd'hui lus et écrits sans synchronisation.
Il constitue néanmoins un excellent protocole de vérification pour la suite.

## 3. Audit de `Chessboard`

`src/chessboard.cpp` a été lu intégralement avant cette conception. Sa conclusion est sans
ambiguïté : un `Chessboard` ne peut pas être partagé entre workers, même en apparence en
lecture seule.

`getLegalMovesForSquare()` appelle `isMoveSafe()`. Cette dernière déplace temporairement
une pièce, retire éventuellement le pion pris en passant, modifie temporairement les
coordonnées du roi, appelle `isInCheck()`, puis restaure le tout. Deux appels concurrents
sur le même plateau créeraient donc une course même s'ils ne font que demander les coups
légaux.

En plus, `movePiece()` met à jour le plateau, le trait, les droits de roque, la prise en
passant, le compteur des 50 coups, le hash Zobrist, `m_moveHistory`, `m_boardHistory` et
`m_snapshotHistory`. `undoMove()` restaure tous ces éléments. `getAlphaZeroTensor()` lit
précisément ces historiques pour les 8 snapshots et les plans de répétition. Une copie
partielle du plateau serait donc incorrecte, même si le FEN courant était identique.

La copie implicite de `Chessboard` est adaptée au besoin initial : ses membres sont des
valeurs et des vecteurs, sans pointeur possédé. Chaque worker recevra donc une copie
complète du plateau racine, historique compris. Une simulation ne modifie que sa copie,
revient à sa racine locale avec `undoMove()`, puis passe à la suivante.

## 4. Approches envisagées

### A. Arbre partagé, plateaux privés, service GPU unique, retenue

Des workers CPU descendent en parallèle dans un arbre MCTS partagé. Chaque worker possède
son `Chessboard` privé. Lorsqu'il atteint une vraie feuille, il capture les données déjà
préparées par le batching, pose le virtual loss, dépose une requête dans une file, puis
attend le résultat. Un seul service d'inférence regroupe les requêtes jusqu'à la taille de
lot demandée, appelle `evaluate_batch()`, puis développe les feuilles et effectue les
backups synchronisés.

Cette architecture réutilise l'arbre, la table de transposition et le batching existants.
Elle est la seule des trois qui vise directement la latence actuellement observée.

### B. Verrou global autour de toute descente, écartée

Un worker prendrait un verrou global, descendrait, le relâcherait pendant l'inférence, puis
le reprendrait pour le backup. C'est facile à rendre correct mais les descentes, la
génération légale et les accès à l'arbre resteraient en série. Le gain serait faible et
serait surtout limité au service GPU déjà présent.

### C. Un arbre indépendant par cœur, écartée

Chaque cœur pourrait chercher dans son propre arbre puis agréger les visites de racine.
Cela évite presque tous les verrous, mais perd la réutilisation profonde, multiplie les
misses de table et ne définit pas naturellement la réutilisation de racine après le coup
adverse. La somme des visites n'aurait pas le même comportement qu'un arbre unique.

## 5. Architecture retenue

### 5.1 Le precedent du self-play, et la surface publique

`SelfPlayManager::generate_games()` utilise déjà jusqu'à 8 threads OpenMP dans sa phase 2.
Il s'agit cependant de racines distinctes, une par partie. Chaque itération `i` possède son
propre `Chessboard` et son propre sous-arbre. Durant cette phase parallèle,
`advance_to_leaf()` ne fait que lire la table de transposition, les écritures ayant lieu
plus tard dans `execute_gpu_batch()`, une phase séquentielle.

Ce précédent valide l'isolement des plateaux et le regroupement GPU. Il ne rend pas
`advance_to_leaf()`, `select_leaf()` ou `expand_and_backup_prepared()` utilisables sur une
racine partagée : ces fonctions lisent et modifient les mêmes compteurs et `children` sans
verrou. Le chemin UCI parallèle sera donc un nouveau noyau synchronisé, sans modifier le
chemin du self-play.

`step_analysis()` et `mcts_search()` reçoivent un paramètre additionnel
`worker_count`, par défaut à 1. `worker_count == 1` conserve exactement le noyau actuel,
sans création de thread ni changement de comportement. Le mode parallèle demande
`worker_count >= 2` et un `batch_size >= 1`.

L'activation UCI ne sera faite qu'après les mesures. La première valeur candidate est :

```python
MCTS_BATCH_SIZE = 8
MCTS_WORKER_COUNT = 8
```

Les deux constantes restent côte à côte dans `uci.py`. Aucun changement ne concerne le
self-play.

### 5.2 Session de recherche parallèle

Une invocation parallèle crée une session temporaire contenant :

- un compteur atomique des simulations à réclamer ;
- une file de `FeuilleCollectee` protégée par mutex et condition variable ;
- un worker GPU unique ;
- `worker_count` threads CPU ;
- un état d'arrêt et une exception éventuelle partagés.

Chaque worker clone le plateau racine au début de l'appel. Il réclame une simulation, fait
une descente, traite immédiatement les feuilles terminales et les succès de table, ou
capture une vraie feuille. Pour cette dernière, il annule ses coups locaux, la met dans la
file et attend sa réponse. Il n'a jamais plus d'une requête GPU en vol. Le nombre de
descentes dont le résultat est différé est donc borné par `worker_count`.

Le service GPU déclenche une inférence lorsque la file atteint `batch_size`. Il doit aussi
vider un lot partiel lorsqu'il n'existe plus assez de producteurs actifs pour le compléter.
Il annule le virtual loss de chaque requête, développe la feuille depuis sa capture et
fait le backup avant de réveiller les workers concernés.

À la sortie de `run_search_parallel`, tous les workers et le service GPU sont joints. Ce
join est obligatoire avant le retour de `step_analysis()`, donc avant qu'un futur
`update_root()` puisse déplacer ou détruire une racine.

### 5.3 Synchronisation de l'arbre

Chaque `MCTSNode` reçoit un mutex propre. Il protège ensemble :

- `visit_count` et `total_value` ;
- `n_in_flight` ;
- `is_terminal` ;
- l'état d'expansion et le vecteur `children`.

Les opérations d'un nœud prennent son mutex pour lire ou modifier ces données, puis le
relâchent avant de passer à un autre nœud. Aucun chemin ne détient simultanément les
verrous parent et enfant, ce qui élimine un ordre de verrouillage cyclique. Les pointeurs
des enfants sont stables pendant une session : les enfants vivent dans des `unique_ptr` et
la racine n'est jamais déplacée pendant la recherche.

L'expansion utilise un état explicite `unexpanded`, `pending`, `expanded` ou `terminal`.
Le worker qui passe atomiquement de `unexpanded` à `pending` est le seul propriétaire de
la requête GPU. Une autre descente qui rencontre `pending` compte une collision, remet son
plateau local à la racine et choisit de produire un lot partiel plutôt que de développer le
même nœud une deuxième fois.

Le virtual loss est posé sous les verrous des nœuds concernés, un à un, puis est annulé
avant le backup. Une garde RAII garantit l'annulation sur toute sortie anticipée ou
exception. Q continue de dépendre seulement des visites terminées.

### 5.4 Table de transposition

La table directe est partagée. Une écriture et une lecture concurrentes sur un `TTEntry`
sont une data race C++, même si l'entrée est de taille fixe. La solution retenue est un
petit tableau fixe de verrous de bandes, par exemple 4096. L'accès à l'entrée
`hash % m_tt_size` prend le verrou `hash % 4096`, copie les données nécessaires dans une
valeur locale, puis le relâche. Aucune référence vers une entrée de table ne survit au
verrou.

Cela évite plusieurs millions de mutex, garde les collisions locales et ne modifie pas la
politique de remplacement actuelle. Les résultats de table restent des accélérations, pas
une source de vérité sur les 8 plans d'historique, limitation préexistante.

## 6. Invariants et cas limites

À la fin d'une recherche réussie :

- tous les `n_in_flight` sont nuls ;
- chaque simulation demandée est terminée exactement une fois ;
- les enfants restent uniques et les priors de chaque nœud développé somment à 1 ;
- la racine a reçu exactement le nombre de backups correspondant aux simulations ;
- le plateau reçu par `step_analysis()` n'a pas changé, FEN, hash, historique et tensor
  compris ;
- un coup extrait des résultats est un coup légal sur le plateau racine.

Un échec d'inférence arrête la session, réveille tous les workers, annule les virtual loss
encore actifs, joint tous les threads puis relaie l'exception. Il ne doit jamais laisser un
arbre réutilisable avec une feuille `pending` ou un compteur en vol non nul.

Les compteurs existants restent comptés par position logique, pas par thread. Des compteurs
supplémentaires distinguent les collisions de feuilles, les lots GPU effectifs, leur taille
moyenne et le temps d'attente CPU pour l'inférence. Ils expliquent un gain ou son absence
sans modifier la décision du moteur.

## 7. Stratégie de test et de mesure

### 7.1 Tests déterministes avant la concurrence

Un nouveau test de copie de `Chessboard` construit une position avec historique, joue une
suite incluant roque, prise en passant et promotion quand les positions sont disponibles,
clone le plateau, puis vérifie FEN, hash Zobrist, coups légaux et tensor identiques. Il
joue et annule ensuite des coups sur une copie sans modifier l'autre.

Le chemin `worker_count == 1` doit produire la même policy que le chemin actuel, avec la
même graine et les mêmes entrées. Les tests existants de virtual loss, d'invariants d'arbre,
de compteurs et de race UCI restent exécutés sans modification de leur exigence.

### 7.2 Tests concurrents

Les tests de MCTS couvrent les workers 2, 4 et 8, les trois positions de
`search_bench.py`, des tailles de lot 1 et 8, et au moins cent courtes recherches par
configuration. Ils vérifient après chaque recherche `inspect_tree()`, zéro `en_vol`, la
conservation des visites et la légalité du coup principal.

Un test d'arrêt et de réutilisation répète : recherche, extraction de résultat, mise à jour
de racine, coup adverse, recherche. Il vérifie que la session se termine avant la mutation
de racine et qu'aucun worker ne conserve de pointeur invalide.

MSVC et la chaîne CMake actuelle ne fournissent pas ThreadSanitizer. Les campagnes de stress
répétées et `inspect_tree()` sont donc la barrière obligatoire sous Windows, sans prétendre
remplacer un détecteur de data races. Une passe optionnelle sous clang avec ThreadSanitizer
peut compléter cette barrière plus tard, mais elle ne bloque pas ce chantier.

### 7.3 Mesures de débit

Avant toute modification, un rapport de référence est produit avec le modèle ONNX réel, le
GPU, un processus, cinq passages et les mêmes trois positions. Il enregistre le matériel,
la taille de table, le batch et le nombre de workers. Il mesure séparément
`mcts_search` et `step_analysis`.

Après chaque étape significative, le même protocole compare au minimum :

| Configuration | Rôle |
|---|---|
| worker 1, batch 8 | référence du batching actuel |
| worker 2, batch 8 | premier effet de concurrence |
| worker 4, batch 8 | pente de montée en charge |
| worker 8, batch 8 | cible initiale |
| worker 16, batch 8 puis 16 | mesure exploratoire, seulement si 8 est utile |

Le rapport donne médiane, minimum, maximum, simulations par seconde, positions réseau par
seconde, appels GPU par seconde, remplissage moyen, taux de table, collisions et attente
GPU. Il ne promet pas un gain fixe : la limite peut devenir le GPU, les verrous ou le CPU.

Le banc contrôlé, arbre froid et petite table, sous-estime les succès de table rencontrés
en partie réelle après `update_root`. Cette réserve est documentée mais ne justifie pas un
benchmark de partie complet, plus long et moins isolé. Une courte séquence avec déplacement
de racine peut être ajoutée comme observation complémentaire, jamais comme unique preuve
de débit.

### 7.4 Niveau de jeu

Le banc de puzzles est exécuté sans bruit de Dirichlet, avec des MCTS neuves et à nombre de
simulations égal. La comparaison est appariée entre worker 1 et worker 8, puis analysée par
les paires discordantes et le test exact de McNemar. Une première passe stratifiée de 500
puzzles sert de barrière rapide. La campagne de 2500 puzzles est réservée au candidat qui
franchit cette première barrière sans régression visible ni violation d'invariant.

Un gain de débit ne suffit pas à conclure : il doit être accompagné d'une absence de baisse
statistiquement crédible à budget de simulations identique. Ensuite seulement, une mesure
à temps fixe peut traduire ce gain en nombre de simulations supplémentaires et en niveau
pratique.

## 8. Hors périmètre

- Paralléliser le self-play, déjà structuré autour de parties distinctes.
- Modifier les 119 plans, l'historique, l'encodage des coups ou les règles du plateau.
- Refaire la table de transposition ou corriger sa clé historique.
- Régler `c_puct`, la taille de batch ou le nombre de workers pour la force avant que les
  mesures de correction et de débit soient stables.
- Remplacer l'architecture par plusieurs arbres indépendants.

## 9. Décision attendue après implémentation

Le nombre de workers par défaut ne sera pas choisi sur intuition. Il sera celui dont le
rapport montre le meilleur compromis entre débit médian, remplissage GPU, stabilité et
absence de régression au banc de puzzles. Si 8 workers ne bat pas matériellement le mode
mono-worker batché, le code conserve ce dernier par défaut et le résultat est tout de même
utile : il aura localisé la limite suivante du moteur.
