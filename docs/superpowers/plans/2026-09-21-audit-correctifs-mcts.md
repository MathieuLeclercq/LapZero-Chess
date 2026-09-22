# Correctifs et tests issus des deux reviews MCTS : plan d'implémentation

> Pour les agents chargés des correctifs : exécuter les lots séparément, en ligne, avec le skill `superpowers:executing-plans` si disponible. Aucun sous-agent. Les cases ci-dessous suivent l'avancement : cochées quand le lot est fait et validé.

**Objectif :** traiter les dix points soulevés lors des deux passes de review, avec des tests capables de détecter les défauts, sans refondre le moteur ni modifier arbitrairement ses réglages de jeu.

**Architecture :** conserver la recherche mono-worker, le batching et les vagues multicoeurs. Corriger les contrats locaux : réservation, expansion, fin de partie, entrée/sortie de l'évaluateur, identité UCI et cycle de vie du self-play. Renforcer le harnais avant de réinterpréter les mesures de performance.

**Technologies :** C++17, CMake/CTest, OpenMP, pybind11, Python 3.13, pytest, ONNX Runtime. Les tests de correction doivent utiliser principalement l'évaluateur C++ contrôlable, sans modèle ni GPU.

**Références :** les deux reviews de cette conversation, `docs/superpowers/specs/2026-09-14-mcts-multicore-design.md`, `docs/superpowers/plans/2026-09-16-mcts-multicore-waves.md`, `docs/superpowers/specs/2026-09-15-tt-evaluation-key-design.md`.

**Plans de suivi :** le rebalayage de divergence apres R1 est planifie dans `docs/superpowers/plans/2026-09-21-rebalayage-virtual-loss.md` ; la fin de lot self-play de R9 est planifiee dans `docs/superpowers/plans/2026-09-21-fin-de-lot-self-play.md`. Ces deux plans remplacent des mesures ou des explications non interpretables, ils ne doublonnent pas les correctifs de ce document.

**Base relue :** commit `f3ae347`, avec notamment `973a1bd` pour le changement de taille du pool self-play. Les noms de fonctions sont plus durables que les numéros de ligne.

**Statut du document :** recommandations issues d'une lecture statique. Aucun test, compilation, entraînement, benchmark ou programme du projet n'a été exécuté pour le rédiger. Aucun correctif n'est appliqué par ce document. Les scénarios proposés restent à implémenter puis à exécuter.

Les chemins ci-dessous sont relatifs à la racine du dépôt.

## Contraintes globales

- Ne pas changer les valeurs par défaut de `virtual_loss`, FPU, taille de batch, workers, profondeur de clé TT ou budget de recherche pour faire passer un test.
- Conserver les tenseurs de 119 plans, l'encodage de politique à 4672 indices et l'historique réel des puzzles.
- Conserver l'amnésie à 1 %, le renfort tactique du premier coup et le retour au budget/bruit normal ensuite.
- Ne pas remplacer le moteur par `python-chess`. Son usage éventuel reste limité à la validation externe d'une fixture, jamais à la recherche ni au self-play de production.
- Ne pas ajouter de dépendance nécessaire à la production pour faciliter un test.
- Ne pas convertir ce chantier en refactor général de `Chessboard`, du MCTS ou de l'inférence.
- Préserver les fichiers utilisateur, checkpoints, replay buffers et résultats existants. Ne pas vider ou migrer des données d'entraînement.
- Ne pas réécrire les rapports historiques comme si leurs mesures avaient été refaites. Ajouter un avertissement ciblé lorsque leur interprétation est affectée.
- Pas de campagne GPU longue, de parties Lichess ou de self-play réel automatiquement : les validations lourdes demandent une décision explicite du propriétaire.
- Aucun commit, push ou merge automatique n'est autorisé par ce document. Si des commits sont demandés ensuite, ne jamais ajouter `Co-Authored-By`.
- Ne pas utiliser de tiret cadratin dans les textes ajoutés.
- Ne jamais présenter une absence d'échec en partie réelle comme une preuve d'absence de bug. Inversement, ne pas présenter un cas limite comme une dégradation mesurée du niveau actuel.

## 1. Couverture des constats

| ID | Point soulevé | Nature du constat | Livrable attendu |
|---|---|---|---|
| R1 | Réservation de plusieurs unités, libération d'une seule | Défaut établi à la lecture, `virtual_loss > 1` | Réservation pondérée équilibrée et tests de toutes les sorties |
| R2 | Invariants incomplets et options non transmises | Deux défauts du harnais | Détection des réservations résiduelles et configuration réellement testée |
| R3 | Mat à 100 demi-coups évalué comme nulle | Défaut préexistant dans la recherche | Même priorité mat/nulle dans tous les chemins MCTS |
| R4 | Bruit de Dirichlet perdu sur une racine issue de TT | Défaut préexistant d'expansion/bruit | Bruit appliqué une fois au bon moment, TT intacte |
| R5 | Sorties réseau malformées insuffisamment contrôlées | Fragilité avec accès potentiellement hors limites | Validation avant lecture, stockage TT ou publication |
| R6 | Tests de padding avec sorties identiques | Lacune de test, pas permutation démontrée | Vérification de l'association position/politique/valeur |
| R7 | Expansion limitée aux 128 premiers coups | Défaut préexistant conditionnel, cas limite rare, priorité basse | Tous les coups dans l'arbre, cache contourné pour la position qui dépasse, capacité 128 conservée |
| R8 | Réutilisation UCI sans comparer la position initiale | Défaut préexistant | Réutilisation conditionnée par la base ET les coups |
| R9 | Explication du gain self-play incorrecte, parties abandonnées | Constat établi, ampleur du biais non mesurée | Mesures correctement décrites et politique de fin de lot explicite |
| R10 | Réservations persistantes après échec de fusion d'une vague | Défaut exceptionnel de durée de vie | Nettoyage garanti avant destruction d'un arbre |

Le fait que le bot joue bien est compatible avec ces constats : plusieurs concernent des paramètres non utilisés par défaut, des entrées inhabituelles, le self-play ou un échec d'allocation.

### Ordre recommandé

1. R2 : rendre les diagnostics fiables.
2. R1 : corriger le déséquilibre des réservations.
3. R10 : fermer le chemin de nettoyage exceptionnel.
4. R5 et R6 : sécuriser puis vérifier l'association des sorties réseau.
5. R3 : corriger les scores terminaux.
6. R7 : découpler les enfants MCTS de la capacité TT, priorité basse et aucun gain Elo attendu.
7. R4 : réparer l'expansion/bruit des racines avec les contrats précédents stabilisés.
8. R8 : réparer l'identité des positions UCI, lot indépendant.
9. R9 : rectifier les diagnostics self-play, puis traiter séparément la politique de fin de génération.

R2 doit pouvoir être validé seul avec une réservation artificiellement laissée active. Ne pas attendre que R1 soit corrigé pour prouver que le diagnostic détecte une fuite.

## 2. Méthode de validation commune

### 2.1 Ce qu'un bon test doit prouver

Pour chaque défaut établi :

- [ ] Construire le plus petit scénario déterministe qui atteint le chemin fautif.
- [ ] Vérifier que le test échoue sur le comportement actuel pour la raison attendue, sans exiger un crash.
- [ ] Appliquer le correctif local.
- [ ] Vérifier que le scénario passe et que les chemins voisins gardent leur contrat.
- [ ] Consigner le scénario et le résultat observé. Ne pas écrire seulement « tous les tests passent ».

Pour une fragilité sans bug observé, comme R6 :

- [ ] Ajouter le test de contrat.
- [ ] Prouver sa sensibilité, par exemple avec une permutation volontaire des sorties dans un adaptateur de test.
- [ ] Si le code actuel respecte le contrat, ne pas le modifier artificiellement.

Une mutation destinée à vérifier un test doit rester isolée, ne pas être commitée et ne pas toucher aux changements utilisateur. Une variante volontairement fautive du faux évaluateur est préférable à une modification temporaire de la production.

### 2.2 Réutiliser l'outillage existant

- `tests/cpp/controlled_evaluator.hpp` : évaluateur sans ONNX ni GPU, avec échecs contrôlés.
- `tests/cpp/mcts_test_access.hpp` : accès de test aux parties privées du MCTS.
- `tests/cpp/test_support.hpp` : assertions existantes.
- `tests/cpp/test_reservation.cpp`, `test_wave_collection.cpp`, `test_wave_search.cpp` : points naturels pour les réservations et vagues.
- `tests/cpp/test_selfplay_shared_core.cpp` : noyau partagé et gestionnaire self-play.
- `python_src/tests/test_search_bench.py` : faux MCTS et contrôle de la transmission des arguments.
- `python_src/tests/test_uci_race.py` : moteur UCI avec dépendances injectées.

Étendre ces fichiers avant de créer une nouvelle infrastructure. Un nouveau fichier de tests n'est justifié que si un ensemble cohérent devient trop volumineux, par exemple les règles terminales de recherche. Dans ce cas, enregistrer explicitement son exécutable et son test dans `CMakeLists.txt`.

Les hooks supplémentaires doivent être propres à une instance, inactifs en production et non exposés comme fonctions utilisateur Python. Ne pas utiliser de variable globale mutable pour déclencher une panne dans plusieurs recherches concurrentes.

### 2.3 Invariants communs au repos

Après une recherche terminée et après la propagation d'une exception :

- aucun nœud `Pending` laissé sans propriétaire ;
- aucun `n_in_flight` résiduel ;
- aucun enfant dupliqué, parent incohérent ou statistique non finie ;
- état du plateau d'entrée préservé, y compris historique et informations utilisées par la clé d'évaluation ;
- sur un appel réussi, nombre de simulations terminées égal au budget demandé ;
- sur un appel échoué, pas d'obligation d'annuler les simulations déjà complètement validées, mais aucune simulation incomplète ne doit être comptée comme réussie.

Ne pas imposer `visites_parent == somme_visites_enfants` partout : l'expansion d'une feuille peut apporter une visite propre. Préserver l'encadrement déjà défini par `inspect_tree`.

Ne pas inspecter un arbre en cours de mutation en contournant son verrou de session. Les assertions de repos se font après retour ou synchronisation explicite.

### 2.4 Matrice commune, sans explosion combinatoire

La suite rapide doit couvrir au moins :

| Dimension | Valeurs ciblées |
|---|---|
| Chemin | séquentiel historique, batch mono-worker, vagues multicoeurs, noyau self-play |
| Workers | 1, 2 et 8 sur les cas pertinents |
| Batch | 0 uniquement avec 1 worker ; 1 et 8 ; 32 sur un test de lot partiel |
| Budget | 0, 1, 3, 8 et 17 |
| Virtual loss | 1, 2, 3 et 8 ; 32 en test unitaire de frontière |
| Padding | désactivé et activé |
| Cache | froid, réutilisation valide, rejet sémantique et collisions forcées |

Ne pas multiplier toutes ces dimensions dans chaque test. Choisir quelques croisements qui atteignent réellement le défaut. Garder le test exhaustif de petites structures, et limiter le nombre de recherches complètes.

Une TT de 3 entrées sert à provoquer des collisions. Une TT de 8192 entrées convient à beaucoup de tests de contrat. Aucune des deux ne représente à elle seule les performances d'une grande TT en partie réelle.

## 3. R1 : libérer exactement les unités réservées

**Statut (2026-09-21) : fait et validé.** `PathReservation` enregistre le nombre exact d'unités par entrée de chemin et `release()` les soustrait une par une. Vérification : tests pondérés 1, 2, 3, 8 et 32 unités, propriétaires simultanés, déplacements et exceptions, recherches réelles aux amplitudes 1, 2, 3 et 8 avec workers 1, 4 et 8, échec d'évaluateur à amplitude 3 puis reprise, et passage réel sur les trois positions avec `--virtual-loss 3 --worker-counts 8` (zéro nœud en vol). Détection par mutation prouvée. Commit : `Libere exactement les unites reservees`.

### Constat et portée

Dans `src/mcts_reservation.cpp`, `PathReservation::reserve(node, units)` ajoute `units` à `n_in_flight`, mais ne conserve que le pointeur du nœud. `release()` retire ensuite 1 par entrée.

Avec `virtual_loss = 1`, les opérations s'équilibrent. Avec 2, 3 ou davantage, des unités restent après chaque descente. Il ne s'agit donc pas d'un simple choix de force d'exploration : la recherche suivante hérite de pénalités qui auraient dû disparaître.

### Fichiers et contrat

- Production : `src/mcts_reservation.hpp`, `src/mcts_reservation.cpp`.
- Vérification des appelants : `src/mcts.cpp`, `src/mcts_batch.cpp`, `src/mcts_wave.cpp`.
- Tests : `tests/cpp/test_reservation.cpp`, `tests/cpp/test_wave_search.cpp`, `tests/cpp/test_wave_collection.cpp`.
- Contrat public à conserver : `reserve(MCTSNode*, std::uint32_t units = 1)`, déplacement sans copie, `release()` et destruction sans exception.

### Implémentation recommandée

- [x] Remplacer chaque pointeur stocké dans le chemin par une entrée contenant le pointeur ET le nombre exact d'unités réservées lors de cet appel.
- [x] Enregistrer cette entrée avant d'incrémenter l'atomique. Si l'allocation du vecteur échoue, aucun compteur ne doit avoir été modifié.
- [x] Dans `release()`, soustraire les unités de chaque entrée, jamais une valeur globale relue dans `SearchTuning`.
- [x] Préserver le transfert exclusif de propriété dans les déplacements. Une affectation par déplacement doit d'abord libérer l'ancien chemin du destinataire.
- [x] Préserver l'idempotence de `release()` : un second appel ne fait rien.
- [x] Préserver le comportement existant pour un pointeur nul ou `units == 0` : aucune réservation.

Ne pas corriger en remettant directement `n_in_flight` à zéro : plusieurs propriétaires peuvent réserver simultanément le même nœud. Ne pas augmenter les vraies visites ni modifier `total_value` pour compenser. Le mécanisme actuel doit rester un compteur séparé.

### Tests précis

**R1-T1, réservation pondérée simple.** Pour 1, 2, 3, 8 et 32 unités, réserver une racine et un enfant. Vérifier les incréments exacts, puis le retour exact à zéro. Vérifier que `visit_count`, `total_value` et Q sont inchangés.

**R1-T2, propriétaires simultanés.** Deux gardes réservent le même nœud pour 2 et 3 unités. Attendre 5, puis 3 après libération du premier, puis 0. Ce test interdit explicitement la fausse correction par remise à zéro.

**R1-T3, même propriétaire et unités différentes.** Un garde réserve un nœud pour 2 unités, un autre nœud pour 8 et, si l'API l'autorise, le premier de nouveau pour 3. Chaque ajout doit être exactement compensé.

**R1-T4, déplacements et exceptions.** Reprendre les scénarios existants de constructeur de déplacement, affectation sur un garde déjà propriétaire, publication et exception, avec des unités différentes de 1. Le garde déplacé ne doit rien libérer deux fois. Un état publié ne doit pas être annulé par un perdant.

**R1-T5, vraie recherche.** Exécuter des recherches avec `virtual_loss` 1, 2, 3 et 8, batch 8, workers 1 puis 4 ou 8, budgets 3 puis 17. Vérifier les invariants de repos après chaque appel sur le même arbre. Vérifier le delta de simulations, pas seulement le dernier compteur global.

**R1-T6, collisions et échec réseau.** Forcer une collision de feuille et une exception au deuxième appel de l'évaluateur avec `virtual_loss = 3`. Après nettoyage, aucune unité ne doit rester, même sur les ancêtres communs. Reprendre ensuite une recherche sur la même instance.

### Acceptation et historique des mesures

- [x] Tous les tests pondérés passent sans modifier la définition d'UCB.
- [x] Le défaut est détecté avant correction pour au moins `units = 2`. Vérifié par mutation : libération à 1 unité, `reservation_tests` échoue sur « concurrent reservations leaked with units 3 » et `wave_search_tests` sur « tree retained in-flight nodes », puis repasse après restauration.
- [x] Les résultats de tuning obtenus avec `virtual_loss > 1` sont marqués comme non interprétables en l'état. Ne pas les présenter comme une comparaison fiable de qualité. Avertissement ajouté à la section 4 du rapport du coût de calcul.
- [x] Aucune nouvelle campagne de tuning n'est lancée automatiquement.
- [x] Le rebalayage de `virtual_loss` après cette correction est planifié dans `docs/superpowers/plans/2026-09-21-rebalayage-virtual-loss.md` : configurations re-mesurées, barrière qualité, propagation au self-play. Il remplace le balayage de septembre, sans le réécrire.

## 4. R2 : rendre les invariants et le harnais fiables

**Statut (2026-09-21) : fait et validé.** `n_in_flight` non nul est désormais une violation détectée par l'inspection. Le harnais configure le MCTS par un point commun (`configurer_mcts`), applique explicitement les défauts, contrôle chaque tranche, affiche `en_vol` et `pending` et les expose en JSON. Vérification : `node_state` en CTest, 265 pytest, et deux passages réels sur les trois positions (8 et 17 simulations, tranches de 8). Commit : `Rend les invariants et le harnais fiables`.

### Deux défauts à corriger ensemble, mais à tester séparément

1. `src/mcts_observe.cpp` compte les nœuds en vol dans `TreeReport.en_vol`, mais n'ajoute pas de violation pour ce seul motif. Le harnais peut donc annoncer « aucune violation » malgré une fuite.
2. La branche `--invariants` de `python_src/search_bench.py` crée son MCTS puis recherche sans lui appliquer les options de lot fixe et de tuning pourtant inscrites dans le contexte du rapport.

### Fichiers et interfaces

- Production diagnostic : `src/mcts_observe.cpp`, éventuellement les commentaires de `TreeReport` dans `src/mcts.hpp`.
- Harnais : `python_src/search_bench.py`.
- Tests : `tests/cpp/test_node_state.cpp` ou `test_reservation.cpp`, `python_src/tests/test_search_bench.py`.
- Conserver le sens de `en_vol` : nombre de nœuds concernés, pas somme des unités réservées.

### Implémentation recommandée

- [x] Lors de l'inspection au repos, tout `n_in_flight != 0` doit ajouter une violation et un message explicite. Garder le plafond de messages existant.
- [x] Garder les violations `Pending` : elles ne sont pas équivalentes au compteur en vol et peuvent révéler un autre défaut.
- [x] Extraire un petit point commun de configuration du MCTS utilisé par les mesures ET les invariants : lot fixe, virtual loss, FPU, facteur de tentatives de collision et activation éventuelle du chronométrage.
- [x] Appliquer explicitement les valeurs demandées sur un MCTS réutilisé, y compris les valeurs par défaut. Éviter qu'un ancien `fixed_batch = true` survive à une configuration qui demande `false`.
- [x] Vérifier les autres arguments : batch, workers, taille TT, profondeur de clé et budget doivent être ceux réellement utilisés.
- [x] Pour `--slices`, soit vérifier les invariants après chaque tranche du même arbre d'analyse, soit refuser clairement cette combinaison. Recommandation : vérifier chaque tranche ; ne pas afficher une granularité non exécutée.
- [x] Faire échouer le programme avec un code non nul si les invariants de repos échouent. Afficher aussi `en_vol` et `pending` pour rendre le diagnostic visible.
- [x] Si une sortie JSON est demandée en mode invariants, y inclure les résultats d'invariants et les réglages effectifs, pas seulement une liste vide de mesures de débit.

### Tests précis

**R2-T1, faux négatif actuel.** Construire un nœud valide par ailleurs, réserver 3 unités puis appeler l'inspection contrôlée sans recherche concurrente. Attendre `en_vol == 1`, `violations > 0` et un message identifiable. Libérer ensuite le garde avant destruction du nœud. Le test actuel doit échouer sur la condition `violations > 0`.

**R2-T2, absence de faux positif.** Inspecter un arbre vide, un arbre propre et un arbre nettoyé après exception. Aucun résidu ne doit être signalé. Un nœud `Pending` doit toujours être détecté indépendamment.

**R2-T3, transmission réelle des options CLI.** Avec les faux objets déjà utilisés dans `test_search_bench.py`, invoquer la logique de `main()` en mode invariants, sans vrai modèle ni ONNX, avec lot fixe activé, virtual loss 3, FPU 0,5, facteur 8, batch 8 et 4 workers. Enregistrer l'ordre des appels. Les setters doivent être appelés avant la recherche avec les valeurs exactes.

**R2-T4, statut de sortie.** Faire renvoyer par le faux inspecteur un rapport violé, puis un rapport propre. Attendre respectivement un code non nul et zéro, et des rapports cohérents avec ces statuts.

**R2-T5, options réutilisées et tranches.** Configurer successivement un même faux MCTS avec des réglages non standards puis standards. Vérifier qu'ils sont réellement rétablis. Pour 17 simulations en tranches de 8, attendre 8, 8, 1 sur le même arbre, sans remise à zéro intermédiaire, avec contrôle au repos après chaque tranche si cette option est supportée.

### Acceptation

- [x] Un résultat vert implique explicitement zéro nœud en vol et zéro nœud `Pending`.
- [x] Le rapport décrit ce qui a été exécuté, pas seulement ce qui figurait dans les arguments.
- [x] La collecte des invariants reste hors des boucles de mesure du débit.
- [x] Les tests du harnais ne chargent ni checkpoint ni GPU.

## 5. R3 : donner la priorité au mat sur la règle des 50 coups

**Statut (2026-09-21) : fait et validé.** `terminal_value_for` dans `src/search_terminal.hpp` classe toute position terminale avec priorité au mat : chemin rapide sans génération de coups quand le roi n'est pas en échec, contrôle des coups légaux quand il l'est. Tous les chemins cités passent par lui : `select_leaf`, `expand_node_single`, `advance_to_leaf`, `expand_and_backup`, les deux branches terminales de `run_search`, et la branche de nulle de règle des vagues. La valeur est initialisée dans le nœud avant la publication de l'état. Vérification : mat chargé à la racine dans les trois chemins (Q racine -1, aucune inférence, aucun enfant, 8 `terminal_hits`), mat porté depuis 99 par une vraie recherche séquentielle (feuille -1, racine positive), classification des cas voisins (mat avant 100, pat, matériel insuffisant, nulle des 50 coups, échec avec échappatoire, répétition). Détection par mutation prouvée sur la priorité au mat. Commit : `Donne la priorite au mat sur la regle des 50 coups`.

### Constat

La logique de `Chessboard` donne déjà la priorité au mat. Plusieurs chemins MCTS testent en revanche `half_move_clock >= 100` avant de déterminer qu'il n'existe aucun coup légal et que le roi est en échec. Ils peuvent donc remonter 0 au lieu de -1 pour le camp maté.

Le test Python existant `test_un_mat_au_centiemes_demi_coup_reste_un_mat` valide le plateau, pas tous les chemins de recherche.

### Fichiers concernés

- `src/mcts.cpp` : `select_leaf`, `expand_node_single`, `advance_to_leaf`, `expand_and_backup`.
- `src/mcts_batch.cpp` : chemins terminaux séquentiel et batché.
- `src/mcts_wave.cpp` : `is_rule_terminal`, `terminal_value`, `collect_wave_leaf`.
- Tests : `python_src/tests/test_endgame_rules.py` comme référence de fixtures ; `tests/cpp/test_wave_search.cpp`, `test_wave_collection.cpp`, `test_selfplay_shared_core.cpp` pour les chemins sans réseau réel.

### Implémentation recommandée

- [x] Définir une classification terminale cohérente : mat = -1 pour le joueur au trait, pat = 0, nulle de règle = 0.
- [x] Vérifier le mat avant de conclure à une nulle lorsque les conditions se chevauchent.
- [x] Réutiliser les coups légaux déjà calculés lorsqu'ils sont disponibles. Ne pas ajouter une génération complète de coups à chaque sélection d'un nœud déjà développé. Les sites qui viennent de calculer une liste vide gardent leur ternaire `isInCheck` ; le nouvel appel n'a lieu qu'après une nulle de règle et seulement si le roi est en échec.
- [x] Pour le raccourci « nulle de règle », vérifier au minimum qu'il ne masque pas un mat, par exemple via échec et absence de coup légal. Conserver le chemin rapide pour les nulles qui ne peuvent pas être des mats.
- [x] Remplacer les décisions contradictoires dans TOUS les chemins cités, pas seulement dans les vagues.
- [x] Lorsqu'une valeur terminale est stockée dans le nœud, l'initialiser avant de publier son état comme disponible aux autres workers. Ne pas confondre cette valeur avec un Q déjà accumulé.
- [x] Préserver le comptage des visites et la convention de signe de `backup`.

Ne pas utiliser aveuglément un état de partie mémorisé si le chemin de simulation ne le recalcule pas. La classification doit être correcte sur les plateaux réellement manipulés par la recherche.

### Tests précis

**R3-T1, mat directement chargé à la racine.** Utiliser `7k/6Q1/5K2/8/8/8/8/8 b - - 100 1`. Les noirs sont au trait et matés. Avec un faux évaluateur qui échoue s'il est appelé, vérifier : aucune inférence, état terminal, Q de racine égal à -1 après des simulations, aucune création d'enfant. Couvrir séquentiel, batch mono-worker et vagues.

**R3-T2, coup matant depuis 99.** Reprendre `7k/8/5KQ1/8/8/8/8/8 w - - 99 1`, jouer `g6g7` dans une fixture de descente contrôlée, puis vérifier la valeur -1 de la feuille et +1 remontée à son parent. Ne pas se contenter du `game_state` de `Chessboard`. Une fixture d'arbre à branche imposée est acceptable pour isoler le backup ; elle ne doit pas prétendre tester l'exhaustivité de la génération des coups.

**R3-T3, vraie nulle à 100.** Position non matée, avec du matériel suffisant et au moins un coup légal : valeur 0, pas d'inférence, pas de fuite. Inclure un roi en échec qui dispose d'une échappatoire pour éviter la fausse correction « en échec à 100 implique mat ».

**R3-T4, cas voisins.** Mat avec compteur inférieur à 100, pat, répétition et matériel insuffisant. Le correctif ne doit pas transformer toutes les positions sans coup en victoire/défaite ni ignorer les nulles déjà prises en charge.

**R3-T5, self-play.** Faire passer le même mat à 100 par `advance_to_leaf` et `expand_and_backup` selon le chemin concerné. Vérifier le signe du résultat de recherche et celui de l'issue finale enregistrée par le gestionnaire.

**R3-T6, terminal_hits.** Conserver un test séparant découverte d'une feuille terminale et réutilisation d'un terminal déjà connu. Pour une racine préalablement classée terminale, toutes les simulations qui la revisitent doivent être comptées selon le contrat actuel. Ne pas imposer `terminal_hits == completed_simulations` sur une recherche générale.

### Acceptation

- [x] Une même position terminale reçoit la même valeur dans tous les chemins.
- [x] Aucun appel NN n'est ajouté sur un terminal connu.
- [x] Les tests n'exigent pas qu'un modèle réel trouve le mat : ils vérifient la règle et le backup.
- [x] Le seuil de 100 actuellement choisi par le moteur n'est pas modifié dans ce lot.
- [x] R3-T5 est couvert : la classification finale du gestionnaire self-play est extraite dans `conclure_partie` et testée sur les deux camps matés (+1 et -1), le pat, la répétition, les cinquante coups, le matériel insuffisant et la longueur maximale. Mutation vérifiée sur le signe.

## 6. R4 : appliquer le bruit de racine même après un hit TT

**Statut (2026-09-21) : fait, deux tests de bord restent.** Un hit TT dans `expand_node_single` matérialise désormais les enfants et publie `Expanded` via `make_children_from_probe`, partagé avec l'expansion paresseuse (`src/search_children.hpp`), sans inférence ni backup. Côté self-play, le bruit de la racine du coup est porté par `m_pending_epsilon` : posé à `reset_game` et au changement de coup, consommé exactement une fois dès que les enfants existent, dans la phase séquentielle (`apply_pending_noise`, appelé depuis `reset_game`, `play_best_move` et `execute_gpu_batch`). Le RNG n'est utilisé que dans ces phases séquentielles. Vérification : un hit TT donne des enfants sans appel réseau et sans backup, la politique du cache reste intacte après un bruit de 0,5, le même germe reproduit le même bruit, epsilon nul ne change rien, les priors restent normalisés, et les deux racines du self-play sont `Expanded` avec bruit consommé dès la construction. Détection par mutation prouvée sur l'expansion depuis la table. Commit : `Applique le bruit de racine apres un hit de table`. R4-T2 est couvert depuis par une comparaison stricte racine froide / racine chaude au même germe. R4-T6 est couvert aussi : avec une fixture de puzzle, le premier coup reçoit le budget tactique de 4000 simulations et le slow move, le bruit 0,30 est consommé, puis le coup suivant retrouve un budget normal et un epsilon au plus de 0,12 ; le cas où la racine du puzzle est servie par la table materialise ses enfants et consomme le bruit de la même façon. Mutation vérifiée sur l'expansion depuis la table.

### Constat

Dans `expand_node_single`, le hit TT peut rendre sa valeur sans créer d'enfants. Le garde de publication remet alors le nœud non publié à `Unexpanded`. L'appel immédiat à `add_dirichlet_noise` ne fait rien, puisque la liste d'enfants est vide.

Les enfants sont matérialisés plus tard pendant la descente. Le bruit attendu a été perdu. C'est particulièrement important pour les nombreuses racines identiques de départ en self-play.

### Fichiers et contrats

- `src/mcts.cpp` : expansion depuis le cache et `add_dirichlet_noise`.
- `src/selfplay_manager.cpp` et éventuellement `.hpp` : préparation de racine, changement de coup, application différée du bruit.
- Tests : `tests/cpp/mcts_test_access.hpp`, `test_node_state.cpp`, `test_selfplay_shared_core.cpp`.
- La TT doit continuer à stocker la politique du réseau, jamais une politique bruitée pour une partie particulière.

### Implémentation recommandée

- [x] Faire en sorte qu'une expansion explicite de nœud depuis un hit TT construise ses enfants, initialise sa valeur réseau et publie `Expanded`, sans ajouter de backup ni de simulation fictive.
- [x] Réutiliser la construction existante des enfants depuis `TTProbe`, avec le contrat R7 sur la complétude de la politique. Extraction dans `src/search_children.hpp`, partagée par l'expansion paresseuse et l'expansion explicite.
- [x] Vérifier le chemin `mcts_search(add_dirichlet = true)` : la racine doit avoir ses enfants avant l'application du bruit, même à budget de recherche nul. L'expansion de racine est immédiate depuis ce lot.
- [x] En self-play, couvrir non seulement `reset_game`, mais aussi une racine extraite d'un enfant encore non développé. Le bruit en attente est appliqué dès que la racine a des enfants, y compris après une matérialisation par la table pendant une descente.
- [x] Recommandation pour le self-play : représenter explicitement le fait que le bruit de ce coup reste à appliquer. Préparer les racines et appliquer ce bruit dans les phases séquentielles, puis consommer ce drapeau exactement une fois.
- [x] Avant de descendre dans une nouvelle racine non développée, garantir son expansion et l'application du bruit prévu. Une expansion depuis TT n'exige pas de NN. Si la racine doit être évaluée, préserver autant que possible le passage par le batch existant.
- [x] Si une première version choisit une expansion scalaire anticipée pour une racine encore non développée, mesurer et documenter ce coût séparément. Non retenu : la racine non développée reste évaluée par le lot existant, le bruit est différé au lieu d'être anticipé.
- [x] Ne jamais faire tirer le bruit par plusieurs workers à partir du même `m_noise_rng` non protégé. Les callbacks de publication ne doivent pas introduire une course sur le RNG. Le bruit n'est tiré que dans les phases séquentielles.
- [x] Ne pas ajouter le bruit à chaque vague, à chaque reprise d'analyse ou à chaque hit de table. L'unité est la racine d'un coup de self-play.

### Tests précis

**R4-T1, racine froide et racine chaude.** Préchauffer la TT avec une politique non uniforme, créer ensuite une nouvelle racine identique. Sans bruit, attendre les mêmes priors normalisés et tous les enfants. Vérifier l'absence d'inférence supplémentaire sur le hit et l'absence de backup ajouté par l'expansion.

**R4-T2, bruit déterministe.** Ajouter au seul accès de test une manière de réinitialiser `m_noise_rng` avec une graine connue. Partir d'une politique identique et d'un ordre d'enfants identique ; comparer les priors après bruit d'une racine froide et d'une racine chaude, avec tolérance numérique. Réinitialiser la graine entre les essais. Aucun setter de graine public n'existe à supposer implicitement.

**R4-T3, preuve d'application.** Avec au moins plusieurs coups et une fixture non dégénérée, vérifier que le bruit change les priors, conserve leur somme à 1 et ne les rend pas négatifs ou non finis. Vérifier aussi qu'avec epsilon nul la politique est inchangée.

**R4-T4, TT non polluée.** Bruiter une racine, en créer une autre depuis la même TT avec bruit désactivé, puis vérifier qu'elle retrouve les priors non bruités. Une autre partie ne doit pas hériter de l'exploration de la précédente.

**R4-T5, une application par coup.** Vérifier une racine issue de TT, une issue d'une inférence et une racine réutilisée. Plusieurs vagues pour le même coup ne doivent pas recomposer le bruit. Après changement de coup, le nouveau bruit est permis une fois.

**R4-T6, renfort puzzle.** Pour une partie puzzle : premier coup avec 4000 simulations et epsilon 0,30 ; coup suivant avec budget normal choisi par `roll_next_move` et epsilon 0,12. Tester aussi lorsque le premier coup est un hit TT. Ne pas changer le taux d'amnésie ni supprimer l'historique.

### Acceptation

- [x] Le hit TT n'empêche plus le bruit demandé.
- [x] La politique en cache reste indépendante du bruit.
- [x] Aucun RNG partagé n'est utilisé sans synchronisation depuis les workers.
- [x] Les compteurs et budgets ne gagnent pas de visite artificielle.
- [x] Tout coût supplémentaire de préparation des racines est visible dans les mesures futures, pas supposé négligeable. Aucun appel réseau ajouté ici : l'expansion depuis la table est gratuite et le coût de matérialisation des enfants est celui de l'expansion habituelle, déplacé avant la recherche.

## 7. R5 : contrôler les sorties de l'évaluateur avant toute utilisation

**Statut (2026-09-21) : fait et validé.** `validate_network_output` dans `src/evaluator.hpp` valide forme et finitude avant tout accès, et l'adaptateur scalaire l'appelle avant `values[0]`. Les trois consommateurs (mono batché, vagues, `execute_gpu_batch`) passent par lui ; le self-play ne peut plus lire un lot court de façon hors bornes. Vérification : corruption ciblée par appel et par ligne (vide, trop long, non fini) sur l'adaptateur scalaire, la racine, la dernière ligne d'une vague et le premier lot self-play, avec reprise après rejet. Détection par mutation prouvée : sans validation, le test scalaire échoue, le test de vague échoue et le test self-play plante en accès mémoire. R5-T2 est couvert : la clé du noeud rejeté n'est pas un hit, la même position est réévaluée et le cache reçoit une entrée finie. R5-T6 est couvert : chaque rejet porte un message non vide qui nomme l'évaluateur. Coût : la vague scannait déjà ses sorties, le chemin mono batché et le self-play gagnent un parcours de 4672 flottants par ligne, non mesuré séparément car dominé par l'inférence. Commit : `Valide les sorties de l evaluateur avant usage`.

### Constat et danger

La boucle de vagues valide les tailles et la finitude des sorties. D'autres entrées sont moins protégées. En particulier, `Evaluator::evaluate` lit `values[0]` après l'appel virtuel, sans vérifier qu'une valeur existe. `expand_node_single` peut ensuite indexer une politique trop courte ou stocker une valeur non finie dans la TT.

La validation doit précéder le premier accès, pas seulement l'expansion des enfants. Le test actuel avec un NaN global peut finir par échouer dans une vague après avoir déjà laissé entrer la mauvaise valeur à la racine ; il ne prouve donc pas que l'entrée racine est protégée.

### Fichiers

- `src/evaluator.hpp` : adaptateur scalaire et contrat commun de sorties.
- `src/mcts.cpp`, `src/mcts_batch.cpp`, `src/mcts_wave.cpp` : consommateurs.
- `src/selfplay_manager.cpp` : `execute_gpu_batch`.
- Tests : `tests/cpp/controlled_evaluator.hpp`, `test_evaluator.cpp`, `test_wave_search.cpp`, `test_selfplay_shared_core.cpp`.

### Implémentation recommandée

- [x] Définir une validation commune recevant les vecteurs de politiques, de valeurs et la taille physique du batch évalué.
- [x] Exiger exactement `batch_size * 4672` éléments de politique et `batch_size` valeurs, avec calcul de taille sûr.
- [x] Vérifier la finitude des politiques ET des valeurs. Garder explicite la convention d'entrée du consommateur : il reçoit les probabilités de politique, pas des logits à softmaxer ici.
- [x] Appeler cette validation dans l'adaptateur scalaire avant `values[0]`.
- [x] Appliquer le même contrat sur les autres chemins avant lecture par indice, création d'enfants ou insertion TT. Réutiliser le contrôle existant des vagues plutôt que scanner deux fois leurs mêmes sorties.
- [x] Valider la sortie entière d'un batch avant d'en consommer le premier élément. Un NaN en dernière ligne ne doit pas être découvert après stockage des premières lignes de ce même batch.
- [x] Rejeter explicitement une sortie invalide. Ne pas remplacer silencieusement un NaN par 0, une politique tronquée par une politique uniforme ou un échec NN par une nulle.
- [x] Préserver les évaluations valides déjà présentes dans la TT ; ne pas vider tout le cache pour masquer l'absence de contrôle à l'entrée.
- [x] Vérifier les nettoyages après rejet : réservations, état `Pending` et retour des plateaux à leur position d'entrée. Ajouter un garde de rollback local là où une exception de ce nouveau contrôle révélerait une restauration manquante.

La politique de fallback uniforme lorsque la somme des probabilités légales vaut zéro est un contrat séparé. Ne pas la supprimer sans décision distincte. Les contrôles supplémentaires de bornes numériques doivent être discutés séparément s'ils changent des entrées jusque-là tolérées.

### Extension du faux évaluateur

Permettre de cibler le numéro d'appel fautif, le type de corruption et, pour un batch, sa ligne. Conserver les modes existants. Ajouter les cas « vecteur vide », « trop long » et « NaN/infini à un emplacement choisi ».

### Tests précis

**R5-T1, premier appel scalaire.** Sur une racine non terminale et une TT froide, injecter séparément : aucune valeur, une politique vide, politique de 4671 éléments, politique trop longue, deux valeurs pour batch 1, NaN de valeur, infini de politique. Chaque cas doit produire une exception descriptive avant accès hors bornes. Un budget de recherche nul permet d'isoler l'expansion initiale.

**R5-T2, pas d'empoisonnement du cache.** Après chaque rejet initial, vérifier avec l'accès de test que la clé fautive n'est pas un hit utilisable. Rétablir le faux évaluateur, reprendre la même position et vérifier qu'une nouvelle évaluation valide a lieu et que les priors/valeurs sont finis.

**R5-T3, dernier élément d'un batch invalide.** Racine initiale valide, deuxième appel contenant plusieurs feuilles, corruption sur la dernière. Vérifier qu'aucune feuille de ce batch invalide n'a été publiée comme une expansion réussie ni insérée dans la TT par ce batch. Les résultats de batches précédents peuvent rester.

**R5-T4, tous les consommateurs.** Reprendre le contrôle sur batch mono-worker, vagues et `execute_gpu_batch`. Le self-play doit libérer les réservations et restaurer les plateaux en attente sans lecture de sortie invalide.

**R5-T5, reprise et destruction.** Après le rejet : rechercher de nouveau, remettre l'analyse à zéro et détruire l'instance dans des tests séparés. Pas de nœud en vol, pas de dépendance envers un état partiellement publié.

**R5-T6, aucune correction silencieuse.** Vérifier le type/message d'exception. Le test ne doit pas considérer comme succès un retour d'une politique uniforme lorsque l'évaluateur a violé son contrat.

### Acceptation

- [x] Le premier accès à `values[0]` est protégé.
- [x] Aucune valeur invalide n'atteint la TT ni l'arbre comme résultat validé.
- [x] Les invariants et plateaux sont propres après erreur.
- [x] La validation sur les chemins normaux n'est pas exécutée deux fois inutilement ; son éventuel coût sera mesuré, pas supposé nul. La vague scannait déjà une fois ; le coût ajouté aux autres chemins reste à chiffrer si un doute apparaît.

## 8. R6 : tester l'association des lignes du batch et le padding

**Statut (2026-09-22) : fait et validé.** `DiscriminatingEvaluator` identifie chaque ligne par la signature exacte de son tenseur et rend des valeurs et politiques distinctes par tenseur. La vérification `verifier_association` rejoue chaque chemin, recalcule le tenseur et exige pour chaque nœud la value et les priors de sa propre ligne ; les tenseurs non servis (transposition sur hit TT) sont ignorés. Couverts : R6-T1 (24 simulations, lot fixe 8), R6-T2 (racine fabriquée avec un enfant terminal, une seule ligne consommée par feuille réseau), R6-T3 (budgets 1, 3, 7 à forme 8, et forme 32), R6-T4 (racine matée, aucun lot envoyé), R6-T5 et R6-T6 (sentinelles de padding ignorées, `nn_batches` et `nn_calls` cohérents), R6-T8 (une variante qui permute deux lignes fait échouer la vérification, donc la fixture est discriminante), et R6-T7 (deux parties distinctes dans un même lot du gestionnaire, value, politique, signe du backup et plateaux restaurés vérifiés par partie ; l'inversion des lignes du lot fait échouer le test). Aucun correctif de production n'a été nécessaire. Commit : `Teste l association des lignes du batch et le padding`, complété par `Termine exactement les parties demandees`.

### Nature du point

Aucune permutation des sorties n'a été démontrée lors de la review. Le problème est que `ControlledEvaluator` renvoie la même politique uniforme et la même valeur pour chaque ligne : un mauvais décalage peut alors être invisible.

Ce lot est d'abord un renforcement de tests. Ne modifier le code de production que si un test révèle réellement une erreur.

### Fichiers

- Support : `tests/cpp/controlled_evaluator.hpp` ou un évaluateur de test spécialisé dans le même répertoire.
- Tests : `test_wave_search.cpp`, `test_wave_collection.cpp`, `test_selfplay_shared_core.cpp`.
- Production à observer : assemblage/consommation dans `src/mcts_batch.cpp`, `src/mcts_wave.cpp` et `src/selfplay_manager.cpp`.

### Évaluateur discriminant recommandé

- Identifier une ligne par son tenseur effectivement reçu, pas par l'ordre supposé des workers.
- Pour un petit ensemble de fixtures, enregistrer les tenseurs attendus et leur associer des sorties distinctes : par exemple des valeurs -0,6, -0,2, 0,3 et 0,7 et des politiques non uniformes différentes.
- Ne pas se contenter de lire une seule case souvent identique. Comparer les tenseurs enregistrés, ou utiliser une signature dont l'unicité est explicitement vérifiée dans la fixture.
- Consigner les entrées dans l'ordre réellement présenté à l'évaluateur afin de reconstruire l'association attendue indépendamment de l'ordonnancement.
- Par défaut, une ligne dupliquée pour padding doit recevoir la même sortie, puisqu'elle représente le même tenseur.
- Pour un test spécifique d'ignorance des lignes de padding, un mode de test peut volontairement renvoyer des sentinelles finies différentes sur ces lignes supplémentaires. Cette dérogation sert seulement à détecter leur consommation accidentelle.

### Tests précis

**R6-T1, feuilles distinctes.** Plusieurs positions réseau dans une vague reçoivent des sorties différentes. Après expansion, vérifier pour chaque feuille son `network_value`, ses priors légaux normalisés et le backup attendu selon sa profondeur. Ne pas vérifier uniquement le meilleur coup ou la somme des visites.

**R6-T2, terminaux intercalés.** Préparer une vague contenant feuilles réseau et feuilles terminales. Seules les feuilles réseau doivent consommer une ligne de sortie. La présence d'un terminal entre deux feuilles réseau ne doit pas décaler l'index de politique/valeur de la seconde.

**R6-T3, lots partiels.** Avec batch nominal 8, forcer 1, 3 et 7 feuilles réseau réelles. Vérifier la forme physique 8 lorsque le padding est actif, la duplication correcte des entrées et la consommation des seules lignes réelles. Reprendre un cas avec batch nominal 32.

**R6-T4, aucune feuille réseau.** Une vague entièrement terminale ou satisfaite sans inférence ne doit pas envoyer un batch factice au GPU. Pas de lecture d'un dernier tenseur inexistant pour remplir le padding.

**R6-T5, sentinelles de padding.** Renvoyer des sorties différentes, mais finies et de dimensions valides, pour les seules lignes de padding. Les priors, valeurs, visites et insertions TT des feuilles réelles doivent rester identiques à la version sans sentinelles.

**R6-T6, compteurs utiles.** Les lignes artificielles ne sont ni des simulations terminées ni des feuilles utiles supplémentaires. Vérifier les deltas de `completed_simulations` et de `nn_calls`, en tenant compte de l'évaluation séparée de la racine. `nn_batches` compte les appels, pas les lignes physiques.

**R6-T7, mélange des parties self-play.** Forcer plusieurs parties distinctes à rejoindre le même batch. Vérifier l'association à `game_idx`, la politique affectée, le signe du backup et le nombre de coups annulés sur le bon plateau après traitement.

**R6-T8, preuve de sensibilité.** Une variante de test qui permute deux lignes de sortie doit faire échouer les assertions R6-T1 ou R6-T7. Si elle ne le fait pas, les fixtures ne sont pas assez discriminantes.

### Acceptation

- [x] Tous les contrôles portent sur l'identité des positions et leurs sorties, pas seulement sur des tailles.
- [x] Aucune hypothèse sur l'ordre d'arrivée des workers n'est nécessaire. La vérification se fait par rejeu de chemin et signature de tenseur, jamais par index de collecte.
- [x] Pas d'exigence d'arbre bit à bit identique entre exécutions multicoeurs.
- [x] Si le code actuel passe les nouveaux tests discriminants, le livrable est un renforcement de couverture, pas un correctif fictif. C'est le cas : aucun code de production n'a changé pour R6.

## 9. R7 : ne jamais tronquer les coups légaux à la taille du cache

**Statut (2026-09-22) : fait et validé.** `make_children_from_policy` construit les enfants sur toute la liste légale et normalise les priors sur cette liste. `EvaluationCache::store` refuse désormais une politique de plus de 128 coups : c'est une tentative de mise en cache documentée, aucune entrée tronquée n'est publiée. `TT_MAX_MOVES` reste à 128, aucune allocation par entrée, aucune campagne GPU. Vérification : frontière cache 129 (aucun hit) et 128 (relu intégralement), expansion préparée de 129 enfants avec somme des priors à 1 et 129e coup dominant présent au prior attendu, invariants d'arbre respectés sur ce nœud, et R7-T5 sur une position réelle à 218 coups légaux vérifiée avec python-chess (218 trouvés par les deux, FEN identique au retour) : tous les enfants présents, priors normalisés, et la deuxième racine rappelle le réseau au lieu de relire une entrée tronquée. Détection par mutation prouvée sur les deux sites. Commit : `Ne tronque plus les coups legaux a la taille du cache`.

### Constat

`make_children_from_policy` dans `src/mcts.cpp` limite la liste à `TT_MAX_MOVES`, actuellement 128. `EvaluationCache::store` tronque également sa politique à cette capacité.

Une limite de stockage TT devient ainsi une limite de l'arbre de recherche, même sur une évaluation fraîche. Le perft peut rester correct puisque le générateur, lui, renvoie la liste complète.

Deux précisions de priorité, issues de la revue du 2026-09-21 :

- Le cas est rare et aucun élément ne montre que le moteur l'ait rencontré. Une position ordinaire ne dépasse virtuellement jamais 128 coups légaux ; des positions construites atteignent 144 coups sans promotions, mais rien n'établit qu'elles apparaissent en partie ou en self-play. Aucun gain Elo n'est attendu de ce correctif.
- La capacité de 128 n'est pas un accident. Elle permet une entrée TT compacte et préallouée : deux nombres par coup, allocation unique de la table. La question n'est donc pas « pourquoi 128 ? » mais « que faire quand une évaluation dépasse la capacité ? ».

La conclusion retenue : conserver 128, garder tous les coups dans l'arbre, et contourner la mise en cache pour la seule position qui dépasse. Agrandir la capacité est écarté par les chiffres. À 4 millions d'entrées et environ 48 + 8 x capacité octets par entrée (estimation 64 bits, sans compilation pour mesurer `sizeof`), passer de 128 à 256 ferait passer la TT d'environ 3,99 à 7,81 Gio, soit +3,81 Gio, sans aucun hit supplémentaire sur les positions ordinaires, avec des insertions plus lourdes sous verrou et une pression de cache accrue. À mémoire constante, 256 emplacements obligeraient à réduire la table à environ 2,05 millions d'entrées, donc davantage de collisions. Les valeurs intermédiaires (160 : +0,95 Gio, 218 : +2,68 Gio) n'apportent pas de meilleur compromis.

Classement : priorité basse, robustesse sur cas extrêmes, derrière R1 et R2.

### Fichiers

- `src/mcts.cpp` : construction d'enfants depuis la politique réseau.
- `src/evaluation_cache.cpp` et `.hpp` : contrat d'insertion complète.
- Tests : `tests/cpp/test_evaluation_cache.cpp`, `test_node_state.cpp` ou `test_wave_search.cpp`, avec `mcts_test_access.hpp` pour atteindre l'expansion préparée.

### Implémentation recommandée

- [x] Construire les enfants MCTS à partir de TOUTE la liste `legal_indices` ; normaliser les priors sur cette même liste complète.
- [x] Conserver `TT_MAX_MOVES = 128`. Aucune augmentation de capacité, aucune allocation dynamique par entrée TT, aucune campagne GPU pour ce point.
- [x] Recommandation simple : si une politique dépasse la capacité TT, ne pas la mettre en cache. L'arbre garde tous ses enfants et la prochaine rencontre pourra refaire une évaluation. C'est ce contournement qui préserve l'optimisation mémoire, pas un agrandissement de la table.
- [x] Une politique partielle ne doit jamais être publiée comme un hit complet. Le test du dépassement doit avoir lieu avant toute publication d'entrée tronquée.
- [x] Conserver le stockage normal pour 128 coups ou moins.
- [x] Vérifier que les deux consommateurs de `TTProbe`, mono-worker et vagues, continuent de recevoir des snapshots complets. Le contrat de `TTProbe` est inchangé, seule l'écriture est refusée au-delà de la capacité.

Le contournement d'une insertion trop grande n'exige pas de nouvelle sémantique de succès. Si l'API reste `void`, documenter qu'il s'agit d'une tentative de mise en cache. Ne pas inventer un hit partiel « suffisamment bon ».

### Tests précis

**R7-T1, frontière cache.** Avec des listes synthétiques de 127, 128 et 129 indices distincts dans `[0, 4671]`, vérifier : les deux premières se relisent intégralement ; la troisième ne produit pas de hit tronqué dans un cache initialement vide. Tester aussi une autre entrée valide déjà présente pour vérifier l'absence de corruption.

**R7-T2, frontière expansion.** Passer une liste synthétique de 129 coups à l'expansion préparée via l'accès de test, avec une clé de test et une réservation correctement acquise. Attendre exactement 129 enfants, indices identiques à la liste d'entrée et somme des priors égale à 1.

**R7-T3, dernier coup dominant.** Donner au 129e indice la probabilité dominante. Vérifier qu'il existe dans l'arbre avec le prior attendu. Ce test empêche une correction qui compterait 129 enfants mais normaliserait encore sur les 128 premiers.

**R7-T4, répétition de l'expansion hors cache.** Répéter la préparation d'une nouvelle racine avec cette même liste. Vérifier que le bypass TT ne ramène jamais une politique partielle et que tous les coups restent présents.

**R7-T5, intégration échiquéenne complémentaire, optionnelle.** Ajouter une position de fixture ayant effectivement plus de 128 coups légaux si une source vérifiée est disponible. Le test doit d'abord affirmer que le moteur en trouve plus de 128, puis comparer l'ensemble des enfants à la liste légale. Ne pas inventer une FEN non vérifiée pour faire croire à une couverture de partie réelle. La position construite citée en review (jusqu'à 144 coups sans promotions) reste à vérifier avec le moteur avant d'entrer dans le dépôt ; ce cas limite ne justifie pas de retarder le lot. **Fait depuis** : `R6R/3Q4/1Q4Q1/4Q3/2Q4Q/Q4Q2/pp1Q4/kBNN1KB1 w - - 0 1`, position construite par Nenad Petrovic, 218 coups légaux confirmés par python-chess et par le moteur, FEN inchangée au retour ; le test vérifie les 218 enfants, les priors normalisés, les invariants et le rappel du réseau à la deuxième racine.

Les indices synthétiques servent à tester le contrat de liste de l'expansion, pas à jouer des coups sur un plateau. Ils ne doivent pas être parcourus comme s'ils étaient des coups légaux d'une position quelconque. R7-T1 à T4 suffisent à détecter la troncature sans attendre la fixture complémentaire.

### Acceptation

- [x] Aucun enfant légal fourni à l'expansion n'est perdu.
- [x] Aucun hit TT ne représente une politique incomplète.
- [x] La mémoire par entrée TT reste celle de 128 emplacements ; aucun budget RAM supplémentaire n'est demandé.
- [x] Le lot reste classé en robustesse de cas limite, sans gain de force attendu ; il ne bloque ni R1 ni R2 et les mesures courantes ne servent pas à nier l'existence du cas rare.

## 10. R8 : inclure la position initiale dans l'identité UCI

**Statut (2026-09-21) : fait et validé.** `UCIEngine` mémorise `base_identity` : `startpos` explicite, ou les six champs FEN normalisés sur les espaces. Les chemins « ne rien changer » et « avancer d'un coup » exigent la même base ; sinon le plateau et l'arbre sont reconstruits. L'identité est mise à jour à chaque commande et remise à zéro par `ucinewgame` ; elle ne suit jamais le plateau courant. Vérification dans `python_src/tests/test_uci_position.py` : compteurs différents, faux prolongement par préfixe, roques différents, vraies réutilisations, cycle du bot avec enregistrement de son propre coup, base sans coups et changement de base. Détection par mutation prouvée : en forçant `meme_base`, les trois tests d'identité échouent. Commit : `Inclut la position de base dans l identite UCI`.

### Constat

`UCIEngine.parse_position` compare `new_move_list` et `last_move_list` pour garder ou décaler l'arbre. La position de base, `startpos` ou FEN, n'entre pas dans cette décision.

Deux commandes `position fen ... moves ...` avec une même liste non vide peuvent donc être traitées comme identiques alors que leur base diffère, notamment sur le compteur de demi-coups ou les droits de roque.

### Fichiers

- Production : `python_src/uci.py`.
- Tests : `python_src/tests/test_uci_race.py`, ou un nouveau `test_uci_position.py` si nécessaire pour séparer les responsabilités.

### Implémentation recommandée

- [x] Mémoriser un identifiant de position de base en plus de la liste de coups.
- [x] Recommandation conservatrice : une identité explicite pour `startpos` ; pour `fen`, les six champs normalisés seulement pour les espaces. Ne pas ignorer le trait, les roques, l'en passant ou les compteurs.
- [x] Exiger une identité de base égale avant le chemin « ne rien changer » ou « avancer d'un coup ».
- [x] Si la base change, reconstruire le plateau et réinitialiser l'arbre, même si les listes de coups sont identiques ou partagent un préfixe.
- [x] Mettre à jour l'identité mémorisée après une reconstruction réussie. La réinitialiser sur une nouvelle partie selon la logique existante.
- [x] Conserver `stop_search()` avant toute modification du plateau ou de l'arbre.
- [x] Conserver la mise à jour de `last_move_list` après le coup joué par le moteur. L'identité de la base ne doit pas devenir l'identité du plateau courant.

Traiter un `startpos` et sa FEN équivalente comme deux identités différentes est acceptable : cela reconstruit davantage, mais reste correct. Une canonicalisation plus ambitieuse n'est pas nécessaire pour réparer le défaut.

### Tests précis

**R8-T1, même liste, compteur différent.** Deux FEN avec le placement initial classique, l'une aux compteurs `0 1`, l'autre `36 19`, suivies toutes deux de `g1f3`. Après la seconde commande, le plateau doit correspondre à une reconstruction fraîche de la seconde base : compteur 37, pas 1. Vérifier également le décompte `real_ply` et l'appel de réinitialisation de l'arbre.

**R8-T2, faux prolongement.** Première base suivie de `g1f3`, puis seconde base suivie de `g1f3 g8f6`. Malgré le préfixe de coups, la seconde commande doit reconstruire. Comparer la FEN et le tenseur à ceux obtenus par chargement/rejeu indépendants.

**R8-T3, vraies réutilisations.** Même base et même liste : pas de reconstruction inutile. Même base et exactement un coup supplémentaire : `update_root` puis application du coup, dans le bon ordre après l'arrêt de la recherche.

**R8-T4, autres changements de base.** Couvrir une variation de placement et de droits de roque avec des coups de fixture légaux dans les deux bases. Une différence significative ne doit pas être masquée par les listes.

**R8-T5, cycle du bot.** Charger une position, simuler l'enregistrement de son propre coup comme le fait la boucle UCI, puis recevoir le coup adverse. Vérifier que la réutilisation normale en partie n'est pas désactivée par le correctif.

**R8-T6, nouvelle partie et absence de coups.** Vérifier la réinitialisation de l'identité et le chargement d'une FEN sans champ `moves`. Aucun état de la partie précédente ne doit subsister.

### Acceptation

- [x] Même base + prolongement légal conserve la réutilisation.
- [x] Base différente interdit la réutilisation, même avec listes identiques.
- [x] Les tests utilisent le moteur UCI injecté sans charger de modèle.
- [x] Aucune régression de l'arrêt préalable de la recherche.

## 11. R9 : corriger l'interprétation et la fin des lots self-play

**Statut (2026-09-22) : R9-A et R9-B faits, mesure de débit différée.** Compteurs `SelfPlayStats` (`games_started`, `games_completed`, `active_slots`, `new_plies`, `replayed_plies`) accessibles par `get_stats`. Le constructeur ne démarre plus de partie : `generate_games(N)` initialise `min(C, N)` places, ne relance que tant que `started < N`, désactive les places sans départ et attend la collecte des N parties engagées. Places inactives exclues des descentes, de `all_blocked` et du batch. N = 0 sans démarrage ni évaluation, quotas négatifs et places nulles refusés, appels successifs indépendants. L'en-tête et les libellés du banc de remplissage sont corrigés, `docs/devlog.md` consigne l'erreur d'interprétation sans réécrire l'entrée historique. Vérification : matrice C = 4 avec N = 0, 1, 3, 4, 5, 7 et couple (2, 3), `started == completed == N`, `active == 0`, aucun appel réseau à N = 0, deux générations successives indépendantes. Détection par mutation prouvée sur la condition de relance. Commit : `Termine exactement les parties demandees`. Restent non faits : la caractérisation chiffrée d'avant correctif (établie par lecture, C + N - 1), et la mesure avant/après de débit (tâche 5 du plan, GPU). T1 est couvert par un levier de test `m_forced_end_plies`, inactif en production : pour C = 2 et N = 3, une place finit à 2 plies et l'autre à 8, les trois résultats collectés ont exactement les longueurs 2, 2 et 8, donc chaque partie démarrée est retrouvée une fois ; mutation vérifiée. T3 est couvert par un test dédié : une seule place active finit par lots d'une ligne, sans attendre un lot plein et sans racine nulle. T4 est couvert : une place joue un puzzle au premier coup à 4000 simulations pendant qu'une partie normale finit à 2 plies, et les deux sont collectées, longueurs 2 et 4. T6 est couvert par le contrôle de cohérence entre issue finale et raison de fin. T7 est couvert : `SelfPlayStats` est exposé au Python par une variante de binding de diagnostic, et le rapport du banc distingue départs, fins, places actives, plies nouveaux, historique rejoué et exemples ; le débit n'utilise que les plies nouveaux, et deux tests purs le verrouillent sur des statistiques artificielles.

### Ce qui est établi, et ce qui ne l'est pas

`SelfPlayManager::generate_games` relance une partie à chaque fin tant que le nombre de résultats terminés reste inférieur au quota. Ce renouvellement se produit aussi lorsque `total_games == concurrent_games`.

Le texte de `973a1bd` et l'explication de `python_src/dev_tools/selfplay_refill_bench.py` opposent donc à tort un pool qui se vide à un pool qui se renouvelle.

Dans le fonctionnement actuel, pour un appel normal qui termine et un pool de C places initialisées, N résultats demandés conduisent à C + N - 1 parties démarrées. Les C - 1 autres ne sont pas récupérées lors de la destruction du gestionnaire local dans le binding. Certaines peuvent être proches de leur fin sans être encore collectées.

Ainsi, C = 256 et N = 512 donnent 767 départs pour 512 résultats récupérés. Ce constat ne mesure ni la quantité exacte de calcul perdue ni un biais de qualité. Il rend plausible une sélection en faveur des parties rapides à terminer ; cet effet doit être quantifié avant d'être affirmé comme une dégradation d'entraînement.

### Fichiers

- `src/selfplay_manager.cpp` et `.hpp` : initialisation, slots actifs, compteur de départs et arrêt.
- `src/bindings.cpp` : vérifier la durée de vie du gestionnaire et préserver le type de retour existant.
- `python_src/dev_tools/selfplay_refill_bench.py` : descriptions et interprétation des mesures.
- `python_src/train_self_play.py` : paramètres seulement si une décision ultérieure les change.
- Tests : `tests/cpp/test_selfplay_shared_core.cpp` ; nouveau test Python du rapport si la logique de présentation est extraite.

### R9-A : corriger d'abord le diagnostic, sans changer le scheduler

- [x] Renommer les comparaisons en configurations explicites `(concurrent, total)` ou « horizon de génération », pas « pool vide/renouvelé ».
- [x] Ajouter un avertissement au compte rendu concerné : le gain observé n'est pas invalidé par principe, mais l'explication causale par un renouvellement nouvellement activé n'est pas démontrée. En-tête du banc et entrée de devlog.
- [x] Distinguer débit d'inférence, débit de simulations, parties terminées et positions effectivement conservées pour l'entraînement. Fait par le rapport du banc : départs, fins, places actives, plies nouveaux, historique rejoué et exemples sont des colonnes distinctes, et le débit n'utilise que les plies nouveaux. La campagne chiffrée reste, avec la mesure différée.
- [x] Introduire des compteurs internes de diagnostic, au moins `started`, `completed` et `active`, accessibles aux tests. Documenter leur mise à jour à chaque transition.
- [x] Caractériser le comportement actuel avec de petites parties scriptées : constater les départs supplémentaires et la non-récupération des parties restantes, sans faire tourner un entraînement réel. Sans objet depuis le correctif : l'arithmétique C + N - 1 est documentée par lecture, et le comportement corrigé est verrouillé par la matrice de tests et par T1 avec fins imposées.

Attention au dénominateur des mesures : `GameResult.total_real_moves` inclut l'historique de la partie source d'un puzzle. Ce n'est pas automatiquement le nombre de nouveaux demi-coups générés pendant cet appel. Pour une mesure de génération utile, compter séparément les nouveaux coups, les exemples retenus et, le cas échéant, le rejeu de l'historique. Ne pas rebaptiser un compteur existant sans adapter sa définition.

### R9-B : recommandation de correction pour une génération finie

Plan de suivi : `docs/superpowers/plans/2026-09-21-fin-de-lot-self-play.md` (compteurs de diagnostic, caractérisation du comportement actuel, génération finie, mesure avant/après et décision). Le présent document fixe le contrat et les tests ; le plan de suivi décrit l'exécution.

Recommandation : démarrer exactement N parties, renouveler les places tant qu'il reste des départs à effectuer, puis laisser finir les parties restantes. Ce choix simplifie le contrat et évite de jeter des parties engagées.

- [x] Séparer `games_started` et `games_completed`.
- [x] Ne pas démarrer automatiquement C parties dans le constructeur alors que le quota N n'est connu qu'à `generate_games`. Le constructeur peut préparer les conteneurs et le MCTS partagé ; le démarrage effectif doit respecter N.
- [x] Introduire l'état actif/inactif d'un slot. Initialiser au plus `min(C, N)` slots.
- [x] Après une fin : enregistrer le résultat une fois, incrémenter les fins, puis redémarrer ce slot uniquement si `games_started < N`. Sinon, le désactiver.
- [x] Arrêter seulement après collecte des N résultats. Les slots inactifs ne doivent participer ni aux descentes, ni à la condition `all_blocked`, ni à la taille utile du batch.
- [x] Accepter les lots GPU plus petits pendant la fin de génération. Ne pas créer de nouvelles parties pour les remplir artificiellement.
- [x] Définir le comportement de N = 0 : aucun démarrage ni évaluation et un résultat vide. Rejeter les quotas négatifs et le nombre de places non positif avant allocation.
- [x] Définir les appels successifs sur le même gestionnaire : recommandation, une génération indépendante par appel après la fin de la précédente, avec résultats et compteurs remis à zéro et sans anciennes parties actives.
- [x] Préserver le format des exemples, les résultats de fin de partie et l'API Python qui renvoie les parties.

Ce choix change le profil de débit de fin d'itération et doit être mesuré. Il n'est pas garanti qu'il réduise tous les temps muraux : terminer une partie longue coûte plus que la jeter. Le critère est d'abord la production correcte de N parties intégrales avec une comptabilité honnête du travail utile.

Alternative possible, hors de ce lot : conserver un gestionnaire persistant et ses parties actives entre appels. Cela exige de définir les transitions de modèle, de TT, de cibles d'entraînement et de reprise après interruption. Ne pas improviser cette architecture pour éviter la fin de lot.

### Tests précis

**R9-T1, transitions déterministes.** Ajouter un accès de test permettant d'imposer des fins de parties à des moments connus, sans réseau réel ni attente aléatoire. Pour C = 2 et N = 3, faire terminer un slot rapidement et l'autre lentement. Après correction, exactement trois identités de partie doivent avoir été démarrées et récupérées une seule fois chacune.

**R9-T2, matrice de quotas.** Tester C = 4 avec N = 0, 1, 3, 4 et 7. Attendre `started == completed == N` et `active == 0` à la fin. Aucun slot inutile ne doit être initialisé pour N < C.

**R9-T3, vidage final sans blocage.** Une seule partie encore active, les autres slots inactifs. Elle doit continuer avec des lots partiels jusqu'à son résultat, sans attendre un hypothétique batch plein et sans accès à une racine nulle.

**R9-T4, pas de perte de partie lente.** Imposer une partie longue ou un premier coup tactique plus coûteux. Vérifier qu'elle est récupérée malgré la fin rapide des autres. Cela vérifie le contrat de collecte, pas une amélioration Elo.

**R9-T5, appels successifs.** Sur le même gestionnaire, demander deux générations finies. La seconde ne doit ni renvoyer les résultats de la première, ni compter ses départs, ni garder ses réservations.

**R9-T6, qualité des exemples inchangée.** Conserver les contrôles existants de dimensions et de finitude ; ajouter l'unicité des identités de partie en test et la cohérence de l'issue finale. Une partie inactive ne doit fournir aucun exemple supplémentaire.

**R9-T7, rapports honnêtes.** Sur des statistiques artificielles, vérifier que le rapport distingue configurations, départs, fins, exemples utiles et nouveaux coups générés. L'historique rejoué d'un puzzle ne doit pas gonfler un compteur présenté comme du nouveau jeu produit.

### Acceptation

- [x] Le diagnostic n'affirme plus qu'un pool auparavant non renouvelé a été rendu renouvelable par le seul changement 512 vers 256.
- [x] Les mesures historiques sont conservées avec leur réserve d'interprétation. L'entrée de devlog du 2026-09-20 reste intacte, une nouvelle entrée la corrige.
- [x] Si R9-B est appliqué, chaque appel réussi renvoie exactement les N parties qu'il a démarrées, sans abandon implicite.
- [x] La décision de conserver ou changer la taille du pool se fonde ensuite sur des mesures comparables. Ne pas modifier cette taille dans le même correctif fonctionnel.

## 12. R10 : nettoyer les réservations si la fusion d'une vague échoue

**Statut (2026-09-21) : fait et validé.** `WaveResultsGuard` vide les résultats des contextes à toute sortie de `collect_wave`, fusion comprise, après le retour de `SearchExecutor::run` (aucun worker ne touche plus aux résultats). La capacité maximale de `merged` est réservée avant la collecte, donc le transfert ne peut plus allouer. Deux hooks de test (`before_merge`, `after_transfer`) permettent d'injecter `std::bad_alloc` sur une instance MCTS sans l'exposer aux bindings. Vérification : nettoyage après échec de fusion, transfert partiel, panne par `step_analysis` puis reprise et `update_root`, panne par `mcts_search` avec preuve `pending_results == 0` avant destruction de la racine locale, unités pondérées à 3. Détection par mutation prouvée. Commit : `Nettoie les reservations si la fusion d une vague echoue`.

### Chaîne de défaillance

Dans `collect_wave`, le `try/catch` protège l'exécution des workers et nettoie leurs résultats si elle échoue. La construction du vecteur `merged`, notamment son `reserve`, vient ensuite hors de cette protection.

Si cette allocation échoue après la collecte, les `WorkerContext` persistants gardent des `PathReservation`. Dans `mcts_search`, la racine locale est détruite pendant la remontée d'exception. Une libération ultérieure des réservations peut donc accéder à des nœuds déjà détruits.

Sur `step_analysis`, la racine persiste d'abord, mais une remise à zéro ou une destruction ultérieure peut reproduire le problème. Il faut garantir la durée de vie, pas seulement rendre le chemin habituel sans exception.

### Fichiers et contrat

- `src/mcts_wave.cpp` : `collect_wave`, `run_search_waves`.
- `src/mcts_wave.hpp` : hooks de test si nécessaire.
- `tests/cpp/mcts_test_access.hpp`, `test_wave_collection.cpp`, `test_wave_search.cpp`.
- Vérifier le contrat de `SearchExecutor::run` : aucun worker ne doit continuer à toucher les résultats après son retour, y compris exceptionnel.

### Implémentation recommandée

- [x] Étendre la garantie de nettoyage à toute la durée de `collect_wave`, fusion comprise.
- [x] Recommandation : un garde local nettoie les `results` des contextes participants à toute sortie. Après transfert réussi, les conteneurs sont vides ou ne contiennent que des objets déplacés sans propriété.
- [x] Le nettoyage ne doit intervenir qu'après la fin de tous les workers. Ne pas libérer un chemin pendant qu'un worker le parcourt encore.
- [x] Réserver éventuellement la capacité maximale de `merged` avant la collecte pour déplacer l'allocation risquée avant l'acquisition de réservations. C'est une réduction du risque, pas un substitut au nettoyage général.
- [x] Préserver les déplacements `noexcept` de `PathReservation` et vérifier ceux des objets qui la contiennent.
- [x] Sur une exception pendant le transfert, les éléments déjà transférés sont nettoyés par le vecteur local et les autres par les contextes. Aucune réservation ne doit être possédée deux fois.
- [x] Ne pas se limiter à un nettoyage dans le destructeur de `MCTS` : dans `mcts_search`, la racine locale peut être détruite bien avant lui.
- [x] Ne pas masquer `std::bad_alloc` en renvoyant un résultat de recherche partiel comme s'il était complet.

### Injection de panne recommandée

Ajouter un hook de test juste après la collecte et avant la fusion, capable de lever `std::bad_alloc`. Il doit être inactif par défaut et associé à l'instance ou à l'appel testé. Il est préférable à un épuisement réel de la RAM et à un remplacement global de `operator new`.

Pour atteindre les points d'entrée publics, prolonger l'accès de test existant afin que cette injection puisse être installée sur une instance MCTS, sans l'exposer dans les bindings utilisateur. Ne pas faire varier la disposition mémoire de la classe entre la bibliothèque et le binaire de tests par des macros incohérentes.

### Tests précis

**R10-T1, nettoyage après collecte réussie.** Collecter au moins deux feuilles, dont certaines possèdent `Pending`, puis lever au point de fusion. Attendre que tous les `context.results` soient vides, que chaque réservation soit libérée et que les plateaux workers soient restaurés avant de détruire la racine.

**R10-T2, transfert partiel.** Injecter une exception de test après transfert d'un premier résultat. Vérifier que ni le vecteur local ni les contextes ne laissent de propriétaire, et qu'aucune unité n'est retirée deux fois.

**R10-T3, racine locale de mcts_search.** Déclencher la panne par le vrai point d'entrée, intercepter l'exception, puis relancer une recherche sur le même MCTS. Vérifier le nettoyage des contextes avant la fin de vie de la racine au moyen des accès de test, pas uniquement l'absence de crash.

**R10-T4, analyse persistante.** Déclencher la même panne avec `step_analysis`, inspecter au repos, puis tester séparément `reset_analysis`, `update_root` lorsque pertinent, une nouvelle recherche et la destruction du MCTS.

**R10-T5, unités pondérées.** Reprendre un cas avec `virtual_loss = 3` une fois R1 corrigé. Le nettoyage de durée de vie doit aussi libérer exactement les bonnes unités.

**R10-T6, outil mémoire complémentaire.** Lorsqu'une configuration AddressSanitizer compatible est disponible, utiliser ces scénarios ciblés pour rechercher des accès après libération. Ce contrôle complète les assertions de propriété ; il ne remplace pas leur preuve déterministe et n'est pas un prérequis pour commencer.

### Acceptation

- [x] Aucun résultat persistant ne possède une réservation après le retour exceptionnel de la collecte.
- [x] La durée de vie de toute racine couvre la libération des pointeurs qui la référencent.
- [x] Les erreurs de workers et les erreurs de fusion disposent de la même garantie de nettoyage.
- [x] Aucune panne réelle de mémoire machine n'est nécessaire pour tester le cas.

## 13. Validation finale des correctifs, à planifier après leur implémentation

**Statut (2026-09-22) :** Niveau A fait au fil des lots. Niveau B fait sauf l'analyse UCI de bout en bout (ligne suivante). Niveaux C et D : ce sont les campagnes GPU en attente, rebalayage et mesure de débit self-play.

### Niveau A : correction rapide, sans modèle

- [x] Construire les cibles C++ existantes concernées et lancer les tests ciblés de réservation, états de nœuds, collecte, recherche, cache, évaluateur et noyau self-play.
- [x] Lancer les tests Python du harnais et des positions UCI avec dépendances factices.
- [x] Si un fichier de test C++ a été ajouté, vérifier son enregistrement CTest, son timeout et la copie de DLL utilisée par les autres cibles. Aucun nouveau fichier de test C++ dans cet audit ; les tests ajoutés vivent dans les cibles existantes, et le perft roundtrip est enregistré en CTest.
- [x] Utiliser des conditions/barrières déterministes pour les tests concurrents, avec timeout de sécurité ; ne pas s'appuyer sur des pauses aléatoires pour provoquer une collision.
- [x] Ne pas considérer les tests Python sautés faute de checkpoint comme une validation de leurs contrats. Les nouveaux cas critiques doivent justement être couverts sans ce checkpoint. Les tests ajoutés n'en dépendent pas.

Les noms CTest existants utiles sont `reservation`, `node_state`, `wave_collection`, `wave_search`, `evaluation_cache`, `evaluator_contract`, `selfplay_shared_core`, `search_executor` et `multicore_stress`. Vérifier les noms au moment de l'exécution si l'arborescence a évolué.

### Niveau B : non-régression du moteur et intégration

- [x] Relancer le perft rapide si les règles terminales ou interfaces de plateau ont été touchées, sans prétendre qu'il valide les bugs spécifiques au MCTS.
- [x] Conserver les tests de copie de plateau et de clé TT : compteur des 50 coups, contexte de répétition, droits de roque et profondeur d'historique.
- [x] Vérifier une analyse UCI complète avec historique, arrêt, reprise et déplacement de racine. Les tests automatisés doivent rester locaux et ne pas se connecter à Lichess. Fait par un pilote local : uciok et readyok, historique de quatre coups, déplacement de racine avec un coup de plus, changement de base, arrêt d'un go infinite ; les quatre bestmove sont légaux selon python-chess et le journal par coup s'écrit.
- [x] Vérifier une génération self-play minuscule avec faux évaluateur, contenant une partie normale et une partie puzzle contrôlée. Fait : la partie lente à 4000 simulations et la partie normale sont générées et collectées ensemble dans le même test.

### Niveau C : performance, seulement après validation fonctionnelle

Ne pas lancer une grosse campagne à chaque lot. Commencer par le harnais de débit déjà présent et un petit ensemble fixe de positions, avec mêmes modèle, provider, taille TT, profondeur de clé, batch, workers et budget avant/après.

Mesurer séparément :

- simulations terminées par seconde ;
- positions réseau utiles par seconde ;
- nombre d'appels NN et taille physique des batches, padding compris ;
- temps mur total et phases instrumentées ;
- premier appel avec création des workers et régime établi ;
- pour le self-play, données réellement conservées plutôt que seules inférences calculées.

Alterner l'ordre des versions/configurations, conserver plusieurs passages et publier la dispersion. Un écart comparable au bruit de mesure n'est ni une amélioration ni une régression prouvée. Ne pas additionner les temps cumulés des threads comme s'ils étaient du temps mur.

Les parties réelles réutilisent arbre et TT sur plusieurs coups. Les tests froids ne décrivent pas entièrement ce régime, particulièrement en finale. Cette limite ne justifie ni de jeter les benchmarks froids ni de lancer automatiquement une longue campagne de parties chaudes.

Attention aux mesures découpées : `get_last_timing()` fournit le dernier appel. Avant d'interpréter un chronométrage par phase de plusieurs tranches, vérifier que le harnais les cumule correctement, ou utiliser un appel unique pour cette comparaison. Ne pas mélanger temps total de la recherche et temps interne d'une seule tranche.

### Niveau D : niveau de jeu, budget explicitement choisi

- [ ] Réutiliser un sous-ensemble déterministe du puzzle bench, avec vrai historique et même modèle.
- [ ] Comparer les mêmes puzzles avant/après, pas deux échantillons indépendants.
- [ ] Séparer score au premier coup recherché et réussite de la ligne complète.
- [ ] Comparer d'abord à nombre de simulations égal pour isoler une éventuelle dégradation logique, puis à temps égal si l'objectif est le bénéfice pratique de débit.
- [ ] Pour le multicoeur, accepter l'ordonnancement non déterministe ; ne pas exiger les mêmes compteurs détaillés ou la même distribution de visites bit à bit.
- [ ] Ne pas présenter une petite variation sur un petit échantillon comme un gain Elo établi. Conserver les désaccords par puzzle pour analyse.
- [ ] Ne pas mélanger correction R1 et nouveau tuning de virtual loss dans la même comparaison.

R4 change l'exploration en self-play, pas le réseau entraîné déjà chargé par le bot. Un correctif du bruit ne promet donc pas une hausse immédiate du niveau du checkpoint actuel.

## 14. Critères de clôture et compte rendu attendu des agents

Pour chaque lot, fournir :

1. Le défaut exact ou la lacune de couverture traitée, avec les fonctions concernées.
2. Le scénario qui échouait avant, ou la mutation de test qui prouve sa sensibilité.
3. Le choix d'implémentation et les comportements volontairement laissés inchangés.
4. Les tests réellement exécutés, leur résultat, ainsi que les tests non exécutés et pourquoi.
5. Les effets éventuels sur mémoire, coût d'inférence et comparabilité des anciens rapports.
6. Les limites restantes, sans annoncer « aucun bug possible ».

### Checklist de couverture globale

- [x] R1 : unités de virtual loss équilibrées sur sorties normales et exceptionnelles.
- [x] R2 : invariants complets, options réellement transmises et statut d'échec fiable.
- [x] R3 : mat à 100 correctement évalué dans toutes les recherches et le self-play.
- [x] R4 : bruit présent sur TT hit, une fois par coup, sans pollution du cache.
- [x] R5 : sorties invalides rejetées avant accès et insertion, reprise propre.
- [x] R6 : tests discriminants de correspondance des lignes et de padding.
- [x] R7 : totalité des coups conservée, jamais de hit de politique tronquée.
- [x] R8 : identité UCI fondée sur la base et les coups, réutilisation normale conservée.
- [x] R9 : rapports corrigés et décision explicite sur la collecte des parties engagées. Génération finie et compteurs faits ; mesure de débit et décision de taille de pool différées.
- [x] R10 : aucune réservation ne survit à l'arbre qu'elle référence après échec de fusion.

Ne pas clore un lot uniquement parce que le bot joue une bonne partie. Ne pas refuser un correctif démontré au motif que le cas est rare. L'objectif est de préserver le bon comportement courant tout en fermant les cas limites et en rendant les preuves de validation plus solides.
