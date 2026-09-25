# Audit de débit du self-play

Date : 25 septembre 2026. Référence du code : `15d349f60d3839e0332d04133288d1d4c4aade04`.

Objectif unique : augmenter la production de jeu et d'exemples d'entraînement sur le PC fixe, dont le débit communiqué est d'environ **9,84 positions/s**, sans dégradation significative de la qualité.

Méthode : lecture du C++, des appelants Python, des bancs et de résultats déjà archivés. Aucun test, benchmark, entraînement ou sous-agent lancé. Aucun code de production modifié. Ce document est un audit, pas une spécification d'implémentation approuvée.

## 1. Avant toute optimisation : mettre en place le banc de répartition des temps

**Cette étape passe avant les leviers de l'audit.** Le livrable attendu est un rapport expliquant où passe le temps d'une génération réelle sur le PC fixe. Pas un classement obtenu en remplaçant le self-play par huit recherches identiques, ni une estimation déduite de l'utilisation GPU affichée dans le gestionnaire de tâches.

Le protocole ci-dessous décrit un travail futur. Il n'est ni implémenté ni exécuté dans cet audit.

### 1.1. Brancher le banc sur le vrai chemin de production

Le pilote doit réutiliser `SelfPlayManager::generate_games`, le même évaluateur ONNX et la conversion Python de production. Ajouter des diagnostics optionnels au chemin existant plutôt que recopier sa boucle dans un exécutable de benchmark. La variante de binding avec statistiques et le banc de remplissage donnent un point de départ, mais pas encore la décomposition nécessaire.

Séparer deux usages :

- **Observer une génération de production** : instrumentation désactivable, agrégée, sans changement de recherche. C'est la référence pour expliquer les 9,84.
- **Comparer deux versions sur un modèle figé** : pilote autonome de génération uniquement. Pas d'entraînement, d'écriture dans le replay buffer, d'évaluation Stockfish ou d'envoi réseau. Inclure la conversion des exemples si le débit annoncé est celui de bout en bout.

Le second usage ne doit pas prétendre reproduire parfaitement le premier sans vérifier ses différences d'initialisation et de mémoire. Pendant la boucle de production, le modèle PyTorch et l'optimiseur restent notamment présents sur le GPU alors que le self-play utilise ONNX.

Prévoir au minimum trois modes de diagnostic : désactivé, phases globales, puis détail CPU optionnel. Ne pas activer d'emblée un chronomètre à chaque nœud.

### 1.2. Figer et enregistrer la configuration

Dans les métadonnées de chaque passage, enregistrer :

- révision du code, binaire C++ réellement chargé et configuration de compilation ;
- chemin et empreinte du modèle ONNX, fichier de puzzles et empreinte ;
- CPU, GPU, RAM/VRAM, versions du runtime et du pilote, nombre effectif de threads ;
- parties demandées, places du pool, simulations slow/fast, ratios, bonus tactique, TT et politique de clé ;
- options de batch, de précision et de diagnostic ;
- graine si le banc en dispose, état froid/chaud de la session et périmètre du chronomètre.

Commencer avec la configuration effectivement utilisée pour les 9,84, et non avec des simulations réduites pour aller plus vite. Les valeurs 512/256 et 700/100 sont des défauts du dépôt, pas une preuve de la commande du PC fixe.

Si une graine de diagnostic est ajoutée, elle doit piloter les deux sources aléatoires : choix de parties/coups dans le gestionnaire et bruit de Dirichlet dans le MCTS. Une graine ne rend pas automatiquement identiques deux ordonnancements concurrents ; ne pas faire dépendre toute la comparaison de cette promesse.

### 1.3. Installer des chronomètres emboîtés, avec des frontières explicites

Utiliser une horloge monotone : `steady_clock` côté C++, `perf_counter` côté Python. Conserver les durées agrégées dans des champs nommés avec leur unité. Les compteurs doivent repartir de zéro par génération.

**Niveau Python :** mesurer séparément création de l'évaluateur, appel C++ complet, conversion des résultats et nettoyage. Le total doit couvrir le même périmètre que la métrique de production. Émettre le rapport de benchmark après ce chronomètre, sans inclure son écriture dans la durée comparée.

**Niveau C++ :** distinguer construction du gestionnaire (TT, buffers, lecture des puzzles), génération, puis retour/destruction et coût de passage vers Python. Dans la génération, la partition de premier niveau doit être la suivante :

| Champ proposé | Début et fin de la mesure | Ce qu'il faut y laisser |
|---|---|---|
| `initial_slots_wall_ns` | Avant le premier départ, après le dernier départ initial | Rejeu puzzle, racines, bruit, éventuelles inférences unitaires |
| `move_management_wall_ns` | Autour de la phase 1, pour chaque tour | Coups joués, déplacement/destruction des arbres, résultats, redémarrages |
| `collection_wall_ns` | Avant l'entrée OpenMP, après la sortie de région | Descentes, tenseurs, travail terminal, ordonnanceur et barrières |
| `assembly_dispatch_wall_ns` | Autour de la fusion des buffers et de la décision de lancement | Copies, scans des places, décision de différer un batch |
| `batch_evaluator_wall_ns` | Juste avant `evaluate_batch`, juste après son retour | Préparation ONNX, `Run`, softmax et récupération des sorties |
| `batch_validation_wall_ns` | Autour de la validation du lot de sorties | Dimensions et finitude, sans retrait des contrôles |
| `batch_consume_wall_ns` | Autour de la boucle consommant les feuilles | Coups légaux, clé, TT, enfants, backup et undo |
| `batch_finalize_wall_ns` | Après consommation, jusqu'à la fin du nettoyage | Bruit différé, libération des réservations et listes d'attente |
| `generation_other_wall_ns` | Résidu explicite par rapport au total de génération | Préparation non classée, contrôle de boucle, retour et autres frais |

Ces catégories doivent être **disjointes au premier niveau**. Par exemple, arrêter le chronomètre d'assemblage avant d'appeler `execute_gpu_batch`, sinon l'inférence sera comptée deux fois.

Les expansions initiales font déjà partie de `initial_slots` ou de `move_management`. Leur `Run` et leur softmax seront aussi utiles comme sous-diagnostics, mais ne doivent pas être ajoutés une seconde fois à la somme principale. Pour répondre à « combien de temps ONNX au total ? », produire un autre regroupement exclusif qui retire ces sous-durées de leurs parents.

Afficher un résidu plutôt que forcer les pourcentages à totaliser 100 %. Un résidu important impose de préciser les frontières ; un résidu négatif significatif signale un double comptage ou des périmètres incohérents. Le pourcentage de chaque phase doit préciser son dénominateur : génération C++ ou génération complète Python.

### 1.4. Décomposer correctement l'évaluateur ONNX

L'[évaluateur possède déjà](C:/Users/mlecl/Documents/LapZero-Chess/src/onnx_evaluator.hpp:9) un mode de timing qui distingue `run_ns` et `softmax_ns`. Le banc doit les **accumuler après chaque appel**, y compris ceux des racines, avec le type d'appel et la taille du batch. Ne pas utiliser seulement `get_last_timing` à la fin d'une génération.

Mesurer aussi la durée totale de `evaluate_batch` pour identifier ses autres frais, sans compter cette durée totale et ses sous-durées comme des postes additionnels. Si plusieurs appels deviennent concurrents ultérieurement, un unique champ partagé « dernier timing » ne sera plus une interface suffisante.

Nommer le poste « appel ONNX Run », pas « calcul GPU pur ». Il peut contenir copies, synchronisations et travail CPU du runtime. Un profil détaillé du runtime ou des événements GPU ne devient nécessaire que si l'on doit départager ces composantes. Le premier banc n'a pas besoin de cette complexité pour repérer un softmax ou un post-traitement CPU coûteux.

### 1.5. Ne pas confondre temps mural et durée cumulée des workers

Le premier rapport utilise `collection_wall_ns`, chronométré **à l'extérieur** de la région OpenMP. Si cette phase est importante, activer ensuite un détail par thread : descente, clé/TT, attente TT et construction du tenseur.

Chaque worker écrit dans sa propre structure de timing. Les agréger après la barrière, sans mutex global ni incrément atomique de chronomètre à chaque nœud. Le [SearchTiming existant](C:/Users/mlecl/Documents/LapZero-Chess/src/search_timing.cpp:21) peut servir de base, mais sa fusion additionne les durées des workers : ce résultat n'est pas une durée murale. Activer `MCTS::set_timing_enabled` ne suffit pas non plus à profiler toute la boucle self-play.

Exemple d'interprétation : huit workers occupés pendant 5 ms peuvent représenter 40 ms cumulées pour environ 5 ms murales. Il ne faut ni ajouter 40 ms au diagramme des phases, ni appeler automatiquement ces 40 ms du « temps CPU système » : les chronomètres monotones incluent aussi les attentes.

Conserver somme, maximum et distribution par worker pour repérer le déséquilibre. Des sous-timers déjà emboîtés, comme la sélection et certaines opérations qu'elle englobe, ne doivent pas être additionnés sans clarification de leur exclusivité.

### 1.6. Compteurs nécessaires pour expliquer les durées

Le rapport doit accompagner les temps de compteurs peu coûteux :

- parties démarrées/terminées, places actives, coups réellement joués, historique rejoué et exemples sauvegardés ;
- simulations terminées, dont terminales ; nombre d'expansions initiales séparé des simulations ;
- requêtes logiques de feuilles, appels réseau et lignes physiques, y compris appels unitaires ;
- histogramme de taille utile/physique et nombre de changements de taille entre appels ;
- tours avec un batch prêt mais non envoyé, durée de ces tours et âge des requêtes ;
- coût des départs et de la conversion finale ; pics mémoire si disponibles.

Le nombre de parties actives et le nombre de parties en attente du réseau sont distincts. Un pool de 256 places actives peut n'avoir que 255 requêtes prêtes ; c'est précisément un cas à rendre visible.

Pour l'attente, utiliser par exemple le maximum d'âge d'une requête et des histogrammes, pas la somme des attentes de 256 parties présentée comme du temps mural. Décomposer les mesures entre départ, régime renouvelé et vidage après le dernier départ autorisé. Une vue par tranches de places actives aide à expliquer les petits batches de fin.

Ne pas ajouter le calcul systématique de hashes de tous les tenseurs au profil de base : le diagnostic de doublons exacts peut être un mode séparé ou échantillonné, avec sa couverture indiquée. Sinon le banc ajouterait lui-même le coût d'A4 avant d'en avoir démontré l'intérêt.

### 1.7. Déroulement d'une campagne de mesure

1. **Vérification de l'instrumentation.** Sur une petite charge contrôlée, établir que les compteurs, les frontières et la restitution des résultats sont corrects. Ce contrôle prépare le banc ; il ne sert pas à prédire le débit de production.
2. **Premier profil représentatif.** Une génération complète avec le vrai modèle, le vrai pool et les vrais budgets, sans autre charge GPU concurrente. Garder les temps de démarrage et de fin. Une seule génération peut repérer un poste majeur, pas prouver un gain de quelques pourcents.
3. **Coût du profilage.** Comparer instrumentation activée/désactivée avec le même travail autant que possible. Rapport minimal et aucune sortie par nœud. Si l'écart atteint l'ordre du gain recherché, alléger ou échantillonner le détail avant de conclure.
4. **Baseline répétée.** Répéter pour connaître la variabilité du PC fixe, avec modèle inchangé et graines consignées. Ne pas comparer directement deux itérations d'entraînement utilisant des modèles différents.
5. **Une optimisation à la fois.** Comparaisons alternées A/B ou A/B/B/A, mêmes paramètres et plusieurs passages. Rapporter médiane et dispersion, nombre de coups/exemples et durée complète, pas seulement le meilleur passage.

Pour la métrique de production, conserver le démarrage d'une nouvelle session et d'une nouvelle TT comme aujourd'hui. Une campagne avec session déjà chaude est possible en diagnostic séparé, mais son gain ne doit pas être vendu comme un gain de bout en bout. Documenter également le régime thermique et les autres charges importantes de la machine.

Ne pas contourner le vidage final en arrêtant le benchmark dès que le quota de résultats arrive plus vite : toutes les parties engagées doivent rester collectées, sans réintroduire le biais précédemment corrigé.

### 1.8. Livrables et critères pour considérer le banc exploitable

Prévoir un résultat structuré par passage et un résumé lisible contenant : configuration, trois débits distincts, partition des temps avec dénominateurs, tailles de batch, attente, fin de génération, mémoire et variabilité. Les timings bruts seront des données de diagnostic, pas de gros fichiers ajoutés automatiquement au dépôt.

Avant d'utiliser ce banc pour choisir une optimisation, vérifier :

- départs = fins = quota, places actives finales nulles ;
- les compteurs réseau incluent bien les appels directs du gestionnaire, pas seulement ceux du MCTS ;
- les sous-durées ne sont pas comptées deux fois et le résidu est expliqué ;
- le mode diagnostic ne change ni budgets, ni bruit, ni exemples produits dans les cas contrôlables ;
- erreurs d'évaluation et sorties non finies continuent à échouer proprement, plutôt que produire un rapport partiel annoncé comme réussi ;
- une permutation volontaire de sorties est détectable avec l'évaluateur discriminant existant ;
- le coût propre du profilage est connu et faible devant le gain que l'on cherche à mesurer.

**Lire le résultat avant de décider :** si l'attente et les variations de forme dominent, commencer par A1 ; si le post-traitement domine, A2 ; si le softmax domine, A3 ; si seul le calcul/transfert ONNX domine, examiner surtout les tailles de batch, A4 et A12. A10 n'est intéressant que s'il existe du travail CPU indépendant à recouvrir.

Utiliser enfin la borne d'Amdahl : supprimer entièrement un poste de 10 % du temps ne peut donner que 1 / 0,90, soit environ +11 % de débit. C'est une borne illustrative, pas une estimation du projet. Elle évite de transformer une accélération locale spectaculaire en promesse globale injustifiée.

## 2. Conclusion de l'audit de code

Le self-play dispose déjà du bon principe : paralléliser les parties, avec une seule feuille en attente par arbre, et mutualiser leurs inférences. Il ne faut pas lui appliquer aveuglément les optimisations de recherche concurrente du bot UCI.

Les premières pistes sont :

1. Mieux déclencher et dimensionner les batches, sans attendre systématiquement la dernière partie.
2. Paralléliser la préparation et le traitement des feuilles, aujourd'hui en partie séquentiels.
3. Réduire le softmax CPU et les calculs de préparation répétés.
4. Dédupliquer les entrées réseau strictement identiques dans un batch.
5. Envisager ensuite le chevauchement CPU/GPU entre groupes de parties.

Le passage de 256 à 512 parties est également une option simple, déjà favorable dans une mesure archivée sur la machine de développement. Ce résultat ne prédit pas celui du PC fixe.

**La présence de coûts évitables est établie par le code ; leur importance relative sur le PC fixe ne l'est pas.** Il serait prématuré d'annoncer un gain global ou de désigner définitivement le CPU, le GPU ou la mémoire comme facteur dominant.

## 3. Baseline et métrique à conserver

### Ce que mesure le compteur Python

Dans [train_self_play.py](C:/Users/mlecl/Documents/LapZero-Chess/python_src/train_self_play.py:267), `saved_positions_per_sec` est le nombre d'exemples conservés divisé par le temps de `generate_games`. Si les 9,84 proviennent de cette métrique, ce ne sont pas tous les coups joués.

Le temps inclut la création de la session ONNX, du gestionnaire et de sa table, la génération, le retour vers Python, la conversion des exemples et le nettoyage. Il exclut l'écriture ultérieure des shards, l'entraînement et l'export suivant. Le débit affiché par la progression C++ a encore un autre périmètre : son chronomètre démarre après l'initialisation des premières parties.

Conserver trois mesures distinctes :

- **Coups nouveaux/s** : `SelfPlayStats.new_plies`, sans le rejeu des historiques de puzzles.
- **Exemples conservés/s** : métrique utile à l'entraînement, avec le même périmètre temporel que les 9,84.
- **Évaluations réseau utiles/s** : diagnostic, pas objectif final à maximiser isolément. Éviter une inférence inutile peut améliorer le self-play en faisant baisser ce compteur.

Avec 25 % de coups à 700 simulations et 75 % à 100, on dépense en moyenne 250 simulations par coup, soit environ 1 000 simulations par exemple conservé. Ce calcul exclut les finales, les premiers coups tactiques et les particularités de réutilisation de l'arbre. Ce n'est pas une mesure du nombre d'inférences.

### Paramètres de qualité à figer

Les défauts du [lanceur](C:/Users/mlecl/Documents/LapZero-Chess/python_src/run_selftrain.py:46) sont 512 parties, 256 places, 700/100 simulations et un ratio slow de 0,25. Le gestionnaire ajoute :

- ratio slow d'au moins 0,60 avec six pièces ou moins ;
- injection de puzzles dans 20 % des départs si le fichier est disponible ;
- 4 000 simulations et epsilon 0,30 au premier coup du puzzle ;
- epsilon 0,12 ensuite, historique réel conservé, amnésie à 1 % ;
- échantillonnage des coups avant 30 demi-coups, puis argmax ;
- réutilisation de l'arbre et collecte de toutes les parties démarrées.

La commande, le modèle, le fichier de puzzles et le matériel effectivement employés sur le PC fixe n'ont pas été vérifiés dans cet audit. Les consigner avant toute comparaison. Ne pas augmenter le ratio slow pour obtenir artificiellement plus de « positions/s ».

## 4. Chemin critique actuel

Le [gestionnaire](C:/Users/mlecl/Documents/LapZero-Chess/src/selfplay_manager.cpp:352) fonctionne par étapes :

1. Séquentiel : jouer les coups dont le budget est fini, déplacer les racines, collecter les résultats et redémarrer les places.
2. OpenMP, au plus huit threads : descendre une fois dans chaque arbre disponible et construire son tenseur.
3. Séquentiel : fusionner les buffers, décider si le batch peut partir.
4. Appel ONNX bloquant, puis softmax CPU séquentiel.
5. Séquentiel : développer les feuilles, mettre à jour la TT, propager les valeurs et restaurer les plateaux.

Le GPU n'est pas alimenté pendant les étapes CPU qui entourent l'inférence. Un grand batch amortit les appels, mais ne supprime pas ces temps morts.

La TT est partagée entre les parties, les arbres ne le sont pas. Un hit rencontré dans `select_leaf` matérialise les enfants puis la descente continue : **un hit TT ne correspond donc pas systématiquement à une inférence évitée**. Les expansions explicites de racines sont un cas différent.

## 5. Priorités

P1 : premier ensemble de candidats après mesure. P2 : deuxième ensemble ou gain conditionnel. P3 : chantier plus invasif, à justifier par le profil. A0 est un prérequis d'observation, pas une optimisation.

| Réf. | Levier | Priorité / effort | Nature du gain possible |
|---|---|---|---|
| A0 | Mesurer les phases du vrai gestionnaire | Prérequis / faible à moyen | Éviter de travailler sur une phase marginale |
| A1 | Déclenchement borné et tailles de batch distinctes du pool | P1 / moyen | Réduire attente et changements de forme |
| A2 | Préparation et post-traitement parallèles | P1 / moyen | Réduire les sections CPU séquentielles |
| A3 | Softmax parallèle, puis éventuellement légal | P1 / faible puis moyen | Réduire les exponentielles et le travail mono-thread |
| A4 | Déduplication exacte des requêtes d'un batch | P2 / moyen | Éviter du calcul réseau répété, surtout en ouverture |
| A5 | Pool de 512 parties | P1, essai de configuration / faible | Gain archivé d'environ 9 %, autre machine |
| A6 | Clés, historique et tenseurs moins recopiés | P2 / faible à moyen | Réduire préparation, allocations et trafic mémoire |
| A7 | Réservation allégée propre au self-play | P2 / moyen | Éviter atomiques et parcours d'ancêtres inutiles |
| A8 | Initialisation des racines groupée | P2 / moyen | Éviter les inférences unitaires bloquantes |
| A9 | Résultats déplacés et conversion par blocs | P2 / faible à moyen | Réduire temps de sortie et pic mémoire |
| A10 | Chevauchement CPU/GPU entre groupes de parties | P3 / élevé | Recouvrir les phases plutôt que les additionner |
| A11 | Application des coups et allocation des nœuds | P3 / moyen à élevé | Réduire le coût CPU si celui-ci reste important |
| A12 | Buffers GPU et autre précision/backend d'inférence | P3 / variable | Réduire transferts ou calcul si ONNX domine |

Ces gains ne s'additionnent pas directement : plusieurs propositions attaquent les mêmes coûts.

## 6. Analyse des leviers

### A1. Ne pas immobiliser un batch prêt derrière une partie terminale

**Constat.** Le [lancement](C:/Users/mlecl/Documents/LapZero-Chess/src/selfplay_manager.cpp:522) exige `batch_full || all_blocked`. Une partie qui termine une simulation sans réseau reste disponible si son budget n'est pas épuisé. Elle empêche alors `all_blocked`, même si toutes les autres ont déjà une feuille prête.

Exemple possible : 255 feuilles attendent, tandis que la dernière partie revisite des feuilles terminales. Les tours de collecte CPU se succèdent jusqu'à ce que cette partie fournisse une requête ou atteigne son budget. Ce n'est pas un blocage infini, mais une attente évitable. Sa fréquence réelle n'est pas connue.

**Recommandation.** Découpler trois paramètres : parties simultanées, batch physique souhaité et délai maximal de collecte. Autoriser un départ suffisamment rempli, avec une borne de tours ou de temps. Conserver un mécanisme de vidage des petits batches en fin de génération.

L'[appel actuel](C:/Users/mlecl/Documents/LapZero-Chess/src/selfplay_manager.cpp:164) utilise la taille exacte, contrairement au chemin UCI à padding. Tester quelques tailles stables avec padding limité peut éviter des changements de forme coûteux. Ne pas imposer un batch de 256 aux trois dernières parties : le gaspillage pourrait dépasser le gain.

**Garde-fous.** Les lignes de padding ne doivent ni produire de visites, ni entrer dans la TT, ni compter comme simulations. Une partie garde exactement son budget. L'ordonnancement modifié peut changer l'ordre des tirages et l'utilisation du cache partagé, donc ne garantit pas des parties identiques au bit près.

**Mesure discriminante.** Nombre de tours avec requêtes prêtes mais sans appel réseau, âge des requêtes, histogramme des tailles utiles/physiques, temps de fin de génération. Valider ensuite un cas avec plusieurs requêtes prêtes et une partie effectuant seulement des simulations terminales.

### A2. Préparer les feuilles en parallèle et paralléliser leur traitement

**Constat.** La [boucle après inférence](C:/Users/mlecl/Documents/LapZero-Chess/src/selfplay_manager.cpp:172) traite toutes les parties sur un seul thread. `expand_and_backup` génère les coups légaux, recalcule la clé, construit les enfants et remonte la valeur.

**Recommandation.** Pendant la collecte OpenMP, capturer la liste légale et la clé de la feuille. Cela réutilise le principe de `expand_and_backup_prepared`, déjà présent dans le moteur. Après l'inférence, traiter les arbres distincts en parallèle.

Cette préparation permet aussi de détecter les nouvelles feuilles mat/pat avant le GPU. Actuellement, l'[absence de coups légaux](C:/Users/mlecl/Documents/LapZero-Chess/src/mcts.cpp:653) peut être découverte seulement après l'évaluation. Les nulles de règle déjà détectées pendant la descente n'ont pas ce problème. Conserver une simulation et le même backup terminal, mais sans requête réseau inutile.

**Garde-fous.** Une seule tâche propriétaire de chaque plateau/arbre. Valider tout le batch avant d'en publier les résultats. Capturer les exceptions dans les tâches et restaurer les plateaux/réservations avant de les propager. Garder le bruit de Dirichlet séquentiel ou isoler son générateur aléatoire, actuellement partagé. Si nécessaire, préparer en parallèle mais publier les entrées TT dans un ordre déterminé.

Huit threads sont imposés par le gestionnaire. Rendre ce nombre configurable ne suffit pas à accélérer les phases qui restent séquentielles. Une région OpenMP est ouverte à chaque collecte, mais cela ne prouve pas que le runtime recrée ses threads système à chaque tour.

**Validation future.** Association feuille/policy/value avec évaluateur discriminant, budgets exacts, plateaux restaurés, erreurs injectées en fin de batch, bruit une seule fois et aucune inférence pour une nouvelle feuille terminale.

### A3. Softmax : distinguer petits batches UCI et grands batches self-play

**Constat.** Le [softmax CPU](C:/Users/mlecl/Documents/LapZero-Chess/src/onnx_evaluator.cpp:77) parcourt séquentiellement 4 672 logits par position. À 256 positions, cela représente 1 196 032 exponentielles par appel. Le MCTS renormalise ensuite les probabilités des seuls coups légaux.

**Premier palier.** Paralléliser les lignes du batch, pas les réductions au sein d'une ligne. Cela conserve l'ordre arithmétique par position et limite le changement d'interface. Adapter le nombre de threads pour ne pas pénaliser les petits batches UCI.

**Second palier.** Calculer le softmax directement sur les coups légaux. En arithmétique réelle, « softmax global puis normalisation sur les coups légaux » est identique au softmax légal. L'interface doit distinguer explicitement logits et probabilités pour tous les consommateurs et la TT.

**Nuance importante.** Ce n'est pas toujours une simple différence d'arrondi en flottant : si un logit illégal domine au point que toutes les exponentielles légales sous-débordent vers zéro, le code actuel passe à une politique uniforme. Le softmax légal peut donner une distribution différente. Couvrir ce cas, les logits extrêmes et les politiques dégénérées. Continuer à rejeter les sorties non finies, y compris hors des coups légaux, au lieu de masquer silencieusement une erreur réseau.

Le softmax légal CPU ne réduit pas à lui seul le transfert des 4 672 logits depuis le GPU. Cela demanderait un traitement distinct côté GPU.

**Mesure discriminante.** Temps cumulé de softmax aux tailles réellement utilisées, pas extrapolation du batch 8. La conclusion historique « 0,2 à 2 % » ne peut pas être généralisée au self-play 256.

### A4. Dédupliquer les requêtes identiques dans le même batch

**Constat.** Pendant la collecte, plusieurs arbres peuvent manquer la TT sur la même entrée réseau. La première évaluation n'est stockée qu'après l'inférence du batch : elle n'empêche pas les doublons déjà collectés. Le marqueur `Pending` protège un nœud d'un arbre, pas une entrée réseau commune à plusieurs arbres.

Cela peut se produire fréquemment au début des parties normales, qui partagent position initiale et historiques de recherche. Aucun taux de doublons n'a été mesuré.

**Recommandation.** Construire une correspondance entre requêtes logiques et entrées réseau uniques. Évaluer chaque entrée unique puis distribuer policy/value à chaque demandeur. Conserver un backup par simulation et des statistiques d'arbre indépendantes.

**Identité indispensable.** Ne pas utiliser seulement le Zobrist ni la clé TT actuelle de profondeur historique zéro. Deux positions peuvent partager cette clé avec des tenseurs différents. Une première version prudente compare l'intégralité des 119 × 64 valeurs, avec hash d'accélération puis vérification d'égalité. Le modèle est exporté en mode évaluation et ne fait pas volontairement dépendre une ligne des autres lignes du batch.

Partager uniquement la sortie réseau complète, pas les priors déjà masqués : chaque plateau conserve ses coups légaux et son traitement terminal. Ne pas fusionner les arbres, les visites ou les bruits de racine. Avec un éventuel cache persistant de ces sorties, invalider au changement de modèle.

**Compromis.** Le hash et la comparaison ont un coût. Si presque toutes les entrées sont uniques, le gain peut être nul ou négatif. Si l'on déduplique puis padde jusqu'à la taille d'origine, on peut perdre tout le bénéfice GPU. Coordonner ce levier avec A1. Les variations numériques dues à la taille du batch doivent également être vérifiées.

**Validation future.** Des entrées identiques doivent recevoir la même évaluation tout en produisant chacune leur backup ; des historiques différents ne doivent pas être fusionnés. Mesurer doublons, lignes réellement calculées et débit global, séparément en ouverture et sur l'ensemble de la génération.

### A5. Augmenter le pool sans modifier le budget par partie

Une [mesure archivée](C:/Users/mlecl/Documents/LapZero-Chess/docs/superpowers/specs/2026-09-22-rebalayage-virtual-loss-resultats.md:108) à 700/100 simulations et 512 parties terminées donne :

| Pool | Coups nouveaux/s | Temps par coup |
|---|---:|---:|
| 256 | 32,8 | 30,47 ms |
| 512 | 35,6 | 28,09 ms |

Soit environ +9 % sur la machine de développement. Pas une nouvelle mesure de cet audit ni une prévision pour les 9,84 du PC fixe.

À considérer comme un essai de configuration peu coûteux. Surveiller RAM, VRAM et fin de génération. Garder le même nombre total de parties et le même modèle. Avec A1, un pool de 512 ne devra pas obligatoirement signifier un batch GPU de 512 : ce sont deux dimensions distinctes.

### A6. Réutiliser les clés et supprimer des copies de tenseurs

**Constat.** Sur une nouvelle feuille non terminale, `select_leaf` construit une clé et consulte la TT ; `advance_to_leaf` recommence ; `expand_and_backup` reconstruit la clé pour le stockage. La [construction de clé](C:/Users/mlecl/Documents/LapZero-Chess/src/evaluation_key.cpp:47) alloue et remplit un vecteur de hashes, même à profondeur zéro. La [construction du tenseur](C:/Users/mlecl/Documents/LapZero-Chess/src/chessboard.cpp:1549) reconstruit encore cet historique et compte les répétitions pour les huit snapshots.

**Recommandation.** Transporter la clé et le résultat de consultation avec la feuille ; parcourir les snapshots sans créer un vecteur temporaire, ou utiliser un scratch réutilisable. Un cache incrémental des répétitions est un palier ultérieur, plus délicat à restaurer lors d'undo.

La seconde consultation TT est particulièrement évitable dans le self-play actuel : les phases de collecte et d'écriture TT sont séparées. Ne pas supprimer aveuglément cette consultation dans un futur mode où d'autres groupes publient simultanément leurs résultats.

Le tenseur passe aussi par scratch, buffer de thread puis buffer global. À 256 places, chaque copie complète représente environ 7,44 Mio ; les huit buffers de threads réservent ensemble environ 59,5 Mio. Préférer une destination attribuée à la requête et des buffers persistants dimensionnés selon leur utilisation. Le `resize` alternant taille maximale et taille utile remet également à zéro la partie réagrandie du buffer, même quand la capacité est conservée.

**Garde-fous.** Garder exactement orientation, historique, amnésie, répétitions et compteurs. Distinguer capacité et longueur logique : le contrat de l'évaluateur doit continuer à décrire le vrai batch. Le but n'est pas de supprimer les protections ajoutées à la clé TT.

### A7. Alléger la réservation, pas retirer les garanties de nettoyage

**Constat.** `advance_to_leaf` réserve tous les ancêtres, avec un vecteur de chemin et des incréments/décréments atomiques. Pourtant aucune autre descente ne peut lire ce virtual loss avant la fin de l'inférence dans la même partie. C'est la raison de son amplitude inerte en self-play.

**Recommandation.** Autoriser un mode interne « une requête par arbre » conservant la propriété de la feuille et le nettoyage RAII, sans réservation de tous les ancêtres. Réutiliser autant que possible le noyau existant ; ne pas dupliquer tout le MCTS.

Ne pas appliquer cette simplification au bot UCI. Elle reste possible avec A10 si l'exclusivité par partie est maintenue, mais plus si plusieurs feuilles du même arbre sont produites simultanément. Le retour `nullptr` de `advance_to_leaf` est aujourd'hui compté comme une simulation achevée par le gestionnaire : ne pas laisser une collision `Pending` devenir possible et être comptée à tort.

**Validation future.** Comparaison à graine fixée avec évaluateur discriminant, absence de réservations résiduelles et restauration après exception. Gain attendu uniquement sur le coût CPU, pas sur la diversité de recherche.

### A8. Grouper les expansions initiales

**Constat.** [reset_game](C:/Users/mlecl/Documents/LapZero-Chess/src/selfplay_manager.cpp:119) appelle `expand_node_single` dans la phase séquentielle. Sur un miss, une inférence de taille 1 suspend la génération. Cela arrive au démarrage et au remplacement d'une partie terminée.

Les départs normaux identiques bénéficient rapidement de la TT : il serait faux de compter systématiquement 256 inférences unitaires au démarrage. Les départs puzzle, plus divers, sont davantage concernés. Le coût global est probablement secondaire, car payé par partie, pas par simulation.

**Recommandation.** Introduire une initialisation de racine pouvant rejoindre la collecte de batches. Réutiliser les hits TT immédiatement. Ne commencer les descentes qu'après matérialisation des enfants et application du bruit requis. Une expansion initiale ne doit pas consommer par accident une des 4 000 simulations tactiques ou modifier la convention de budget actuelle.

**Mesure discriminante.** Nombre et durée des appels de batch 1 dus aux départs et leur effet sur les formes ONNX. Ne pas sacrifier les historiques de puzzles pour accélérer leur initialisation.

### A9. Réduire copies et rétention des données terminées

**Constats vérifiés.** Dans la [fin de génération](C:/Users/mlecl/Documents/LapZero-Chess/src/selfplay_manager.cpp:411) :

- les vecteurs par coup sont recopiés dans les grands vecteurs de `GameResult` ;
- `m_finished_games.push_back(res)` copie ces grands vecteurs ;
- les données par coup d'une place devenue inactive restent conservées, car seule une nouvelle partie appelle leur `clear` ;
- `return m_finished_games` copie le membre, qui n'est pas une variable locale bénéficiant automatiquement d'un déplacement au retour.

Un exemple dense FP32 occupe 48 Kio pour état et politique, hors métadonnées. Les copies peuvent donc peser sur le pic mémoire. Leur effet sur le débit dépend de la RAM disponible, pas seulement du nombre d'octets copiés.

**Recommandation.** Déplacer les résultats devenus propriétaires, libérer les données sources après leur transfert logique, puis envisager une accumulation directement plate. En Python, convertir en FP16 par partie plutôt que par position et préserver le signe de la valeur selon le trait.

Les [propriétés NumPy du binding](C:/Users/mlecl/Documents/LapZero-Chess/src/bindings.cpp:286) sont déjà des vues avec un propriétaire qui maintient les données en vie. Le caster pybind11 local transmet aussi la catégorie de valeur des éléments. Ne pas présumer une copie supplémentaire de tout le dataset par pybind11, contrairement à une ancienne note du backlog.

Ne pas effectuer le calcul des valeurs à partir de données déjà libérées, ni modifier la précision utilisée par la recherche sous couvert d'optimiser le stockage.

### A10. Production en pipeline entre groupes de parties

**Recommandation architecturale.** Pendant que le GPU traite A, préparer B sur CPU ; pendant qu'il traite B, appliquer les résultats de A. A et B contiennent des parties distinctes. C'est une concurrence entre arbres, pas une nouvelle forme de virtual loss dans chaque arbre.

Conditions :

- une seule opération de recherche propriétaire d'une partie ;
- buffers d'entrée/sortie distincts et vivants jusqu'à consommation ;
- initialement un seul propriétaire de la session ONNX ;
- traitement correct des erreurs, arrêt et vidage final ;
- TT protégée, générateurs aléatoires non partagés sans discipline ;
- budgets, bruit et sauvegardes indépendants de la vitesse des groupes.

Prévoir assez de parties disponibles pour préparer B. Si toutes les 256 parties sont déjà en attente dans le batch A, il n'y a plus de travail de descente disponible. D'où l'intérêt de séparer taille du pool et taille du batch.

Pour le seul chevauchement, le coût idéal passe de `T_CPU + T_GPU` à `max(T_CPU, T_GPU)`, hors surcoûts. Le plafond théorique de ce mécanisme seul est donc deux, et beaucoup moins si une phase domine. Cela ne promet pas un doublement réel.

La TT et les tirages aléatoires rendent l'ordonnancement observable : conserver les mêmes budgets n'implique pas automatiquement des trajectoires identiques. Valider la qualité, sans exiger artificiellement le bit à bit de toute génération concurrente.

### A11. Coûts CPU plus profonds, seulement si nécessaires

- [movePiece](C:/Users/mlecl/Documents/LapZero-Chess/src/chessboard.cpp:1193) alloue des buffers et vérifie la légalité d'un coup déjà sélectionné parmi les enfants légaux. Séparer validation publique et exécution interne d'un coup validé est une piste, mais impose de conserver promotion, roque, en passant, hash, snapshots et undo. Ne pas supprimer la validation des coups entrants UCI/FEN/PGN.
- [checkInsufficientMaterial](C:/Users/mlecl/Documents/LapZero-Chess/src/chessboard.cpp:157) parcourt 64 cases après chaque coup de descente. Un retour anticipé conservateur dès qu'un pion, une tour ou une dame est trouvé, ou un compteur de matériel fiable, peut éviter ce parcours complet. Ne pas changer les règles de nullité à cette occasion.
- L'expansion alloue chaque enfant séparément ; les changements de racine détruisent les branches rejetées. Une allocation par blocs améliorerait potentiellement localité et coûts d'allocation/destruction. C'est plus invasif : stabilité des pointeurs et libération des sous-arbres conservés sont critiques.
- La table à quatre millions d'entrées représente de l'ordre de 4 Gio et est reconstruite à chaque génération. Mesurer initialisation, pic mémoire et attentes de verrous avant de la compacter ou de l'agrandir. Préserver tous les coups légaux et la gestion des positions dépassant 128 coups. Une TT plus grande ne garantit pas davantage de coups joués par seconde.
- La configuration CMake propose déjà une variante optimisée `RelWithDebInfo`. Vérifier ultérieurement le binaire réellement chargé sur le PC fixe, sans prétendre qu'il est actuellement en Debug. L'optimisation interprocédurale est une piste secondaire pour les petits accesseurs hors ligne ; éviter les options flottantes agressives qui invalideraient les garanties numériques.

### A12. Transferts et backend GPU

L'[évaluateur](C:/Users/mlecl/Documents/LapZero-Chess/src/onnx_evaluator.cpp:35) expose un buffer CPU à ONNX et récupère des sorties exploitées sur CPU. Il n'organise pas explicitement de double buffering ou de liaison à des sorties GPU persistantes.

Le wrapper d'entrée utilise déjà le pointeur existant : ce n'est pas une copie supplémentaire complète du tenseur côté C++. ONNX peut également réutiliser ses allocations internes. Ne pas déduire une allocation GPU coûteuse à chaque appel de la seule présence d'un objet `Ort::Value` local.

Si les mesures révèlent un coût important de transfert ou d'allocation, examiner buffers persistants, mémoire de transfert adaptée et I/O binding. Les API de liaison sont présentes dans les headers ONNX Runtime locaux. Elles ne rendent pas la recherche entièrement GPU : les arbres CPU ont toujours besoin de leurs résultats, et un softmax légal CPU ne réduit pas le volume de sortie réseau.

Si le calcul réseau domine après les optimisations précédentes, un backend mieux adapté ou une précision réduite devient un chantier légitime. L'ancien essai FP16 était défavorable sur des petits batches et la machine de développement ; il n'établit pas le résultat à 256/512 sur le PC fixe. Pas de promesse de gain, et validation numérique puis qualité obligatoire. Aucun changement d'architecture du réseau nécessaire pour les premières pistes de cet audit.

## 7. Mesures existantes : ce qu'elles ne prouvent pas

- Le [banc selfplay_phase_bench](C:/Users/mlecl/Documents/LapZero-Chess/tests/cpp/selfplay_phase_bench.cpp:17) utilise huit arbres, des recherches depuis la position initiale et une TT de 8 192 entrées. Il ne pilote pas une génération complète de `SelfPlayManager` avec puzzles, renouvellement et vidage final. Sa décomposition ne représente pas directement la production.
- Dans le mode à tailles fixes de [evaluator_batch_bench](C:/Users/mlecl/Documents/LapZero-Chess/tests/cpp/evaluator_batch_bench.cpp:119), `wall_ns` couvre plusieurs appels, alors que `run_ns` et `softmax_ns` sont ceux du dernier appel. Le mode variable, lui, les cumule. Ne pas soustraire ces champs comme s'ils décrivaient toujours le même intervalle, ni en tirer un partage CPU/GPU précis.
- Les données brutes archivées montrent des temps de softmax nettement supérieurs à 0,10 ms aux grandes tailles, mais avec une forte dispersion. Elles motivent A3 sans quantifier son gain de production.
- Les compteurs `nn_calls` et `nn_batches` du MCTS ne comptent pas les gros appels directs de `SelfPlayManager::execute_gpu_batch`. Une instrumentation de ce gestionnaire est nécessaire pour un dénominateur correct.
- Le banc de remplissage utilise bien `new_plies`, mais son chronomètre n'inclut pas exactement les mêmes opérations Python que la métrique de production.

## 8. Garde-fous pour les futurs changements

Aucun de ces contrôles n'a été exécuté dans cet audit.

1. **Intégrité mécanique.** Une simulation, un backup ; budgets inchangés ; aucune ligne de padding comptée ; bons indices après déduplication ; plateau restauré et aucune réservation persistante après erreur.
2. **Intégrité des données.** Même définition des exemples slow, même représentation de l'historique et du trait, même convention de résultat final. Toutes les parties démarrées sont terminées et collectées, y compris les longues parties et les puzzles à 4 000 simulations.
3. **Équivalence locale.** Avec un évaluateur discriminant, comparer exactement les opérations censées être équivalentes. Pour les kernels ou normalisations modifiés, contrôler les erreurs numériques et les distributions de visites sur des positions variées, pas seulement un coup forcé.
4. **Qualité.** Le banc de puzzles est un filtre utile, mais ne suffit pas à certifier la qualité de données self-play. Examiner aussi diversité, longueur, raisons de fin et cibles de politique des générations. Un gain de débit ne prouve pas un gain de niveau après entraînement.
5. **Débit honnête.** Même modèle et données, budgets et nombre de parties inchangés, comparaisons alternées et répétées sur le PC fixe. Rapporter durée complète, coups nouveaux/s et exemples/s, pas uniquement le régime favorable à pool plein.

Changements à ne pas mélanger avec les premiers lots : baisse des simulations, suppression des historiques, augmentation du ratio slow, réduction de la limite de longueur, abandon des dernières parties, relâchement de la clé TT, baisse des vérifications de validité ou plusieurs feuilles concurrentes par arbre.

Même « jouer immédiatement quand un seul coup est légal » n'est pas totalement neutre : la politique de ce coup est triviale, mais les simulations actuelles préparent le sous-arbre réutilisé au coup suivant. Cette économie serait une expérience de qualité séparée, pas une optimisation équivalente.

## 9. Ordre de travail recommandé

1. Fixer le périmètre des 9,84 et observer le gestionnaire réel sur le PC cible (A0).
2. Choisir le premier changement selon le temps mesuré : A1 pour l'attente/formes, A2 pour le post-traitement, A3 pour le softmax. Le pool de 512 peut être comparé séparément dès ce stade.
3. Ajouter A4 si les doublons sont suffisamment fréquents ; traiter A6 à A9 par petits changements indépendants.
4. Ne lancer A10 que si les temps CPU/GPU offrent un recouvrement utile. Préférer cette concurrence entre parties à une recherche UCI multifeuille greffée dans chaque partie.
5. Réserver A11 et A12 aux coûts qui subsistent réellement.

La recommandation centrale est de **mieux exploiter les parties indépendantes et supprimer du travail redondant avant de diminuer la qualité de la recherche**. Les premiers gains ne demandent ni réentraînement préalable, ni changement de modèle, ni abandon du moteur maison.
