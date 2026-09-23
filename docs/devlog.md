# Devlog

Décisions non évidentes, pièges techniques, bugs complexes et pistes abandonnées.
Une entrée par ligne : `AAAA-MM-JJ HH:MM : [Catégorie] Description courte`.
L'horodatage est celui du commit ou du document qui consigne la découverte ; le
journal est reconstruit après coup depuis `docs/superpowers/`, `docs/rapports/`
et `git log`. Les détails complets restent dans ces documents, pas ici.

2026-03-03 15:01 : [Réseau] Encodage AlphaZero, plateau retourné quand les Noirs sont au trait, sinon le réseau voit deux distributions.
2026-03-09 12:10 : [MCTS] Ébauche de MCTS batché retirée au profit du séquentiel, plus simple ; le batching ne reviendra qu'en 2026-09.
2026-03-11 02:26 : [Réseau] Toute la génération 1 avait été entraînée avec la mauvaise tête de policy ; tête corrigée et checkpoints repris.
2026-03-12 00:09 : [MCTS] Noeud non visité évalué avec le Q de son parent au lieu de zéro, précurseur de la FPU.
2026-03-12 22:32 : [Moteur] Détection de répétition fausse ; corrigée.
2026-03-19 21:47 : [Outil] Stockfish ancré par noeuds (200k) au lieu du temps, pour des mesures reproductibles.
2026-03-21 10:34 : [Entraînement] AdamW et replay buffer en deque ; la liste copiait le buffer à chaque itération.
2026-03-26 02:41 : [MCTS] FPU par réduction de 0.25, puis 0.30 (approche Lc0), retenu comme défaut.
2026-04-06 20:10 : [Réseau] Nouvelle génération SE à têtes élargies ; le transfert de poids (153/187 tenseurs) donne une value loss deux fois plus haute, lignée relancée depuis des poids aléatoires.
2026-04-07 22:12 : [Moteur] Nulle non détectée quand le coup joué donne échec.
2026-04-13 13:00 : [Perf] Self-play batché GPU ; plusieurs correctifs le même jour avant la première génération de parties correcte (15:50, 18:17).
2026-04-14 11:57 : [Perf] Enfants du noeud MCTS en std::vector au lieu d'une table unordered_map, sélection sur une structure contiguë.
2026-04-14 13:31 : [Bug] Correction multithread dans le self-play partagé.
2026-04-14 19:03 : [Entraînement] Bruit de Dirichlet corrigé à la racine, nulles forcées à 200 plies.
2026-04-15 20:31 : [Données] Replay buffer en shards sur disque, la RAM saturait.
2026-04-17 11:30 : [Perf] Optimisation du RNG et de la sélection du MCTS.
2026-04-21 21:02 : [Entraînement] Slow move plus fréquent en finale, la conversion y demande plus de simulations.
2026-04-23 13:06 : [Entraînement] Injection de puzzles Lichess en self-play, 20 % des parties.
2026-04-23 17:04 : [Entraînement] Premier coup d'un puzzle forcé en slow move, 4000 simulations et epsilon de Dirichlet 0.30.
2026-04-23 23:24 : [Entraînement] Nulles forcées portées de 200 à 300 plies.
2026-04-27 19:20 : [Réseau] Mode amnésie sur 5 % des positions, historique masqué dans le tensor.
2026-08-07 19:47 : [Perft] Fuzzer différentiel contre python-chess, 400k positions et 0 divergence ; le biais de roques est corrigé en écourtant les parties à 80 plies.
2026-08-07 19:56 : [Perft] 610M noeuds conformes sur les 6 positions, 0 bug du générateur ; seul défaut trouvé, perft divide jetait ses violations silencieusement.
2026-08-07 20:15 : [Build] requirements.txt incohérent (torch 2.4 sans wheel cp313 face à un CMake qui exige Python 3.13) et imports manquants (onnx, whr) ; remplacé par uv et pyproject.toml.
2026-08-07 20:47 : [Données] Décision de supprimer les plans d'historique, révisée le jour même (20:58, 21:25) : les 119 plans restent, le confondant se corrige à la source, pas dans l'architecture.
2026-08-07 21:41 : [Réseau] Amnésie conservée mais ramenée de 5 % à 1 % ; son but devient l'augmentation pure, plus la rupture d'un confondant.
2026-08-07 21:43 : [Données] Confondant historique / tactique mesuré sur la partie O7KXpd5v : prior du coup solution 39 % en FEN nue contre 0,05 % avec historique réel, facteur 800, et value fausse d'environ 1.3 pion.
2026-08-08 00:24 : [Données] Position d'un puzzle appariée par rejeu de la partie, jamais par le numéro de ply du GameUrl ; le procédé s'auto-valide et écarte les cas douteux.
2026-08-08 18:20 : [Données] 105 000 puzzles rejoués par le moteur, 0 rejet, historique moyen 54.8 coups.
2026-08-08 18:34 : [Banc] Un puzzle se présente avec son historique réel et une TT neuve par puzzle ; sans ça la mesure reproduit le confondant et les hits d'un puzzle contaminent le suivant.
2026-08-08 18:34 : [Banc] Le banc pilote mcts_search et jamais step_analysis, qui conserve sa racine entre appels ; le softmax et le nombre de plans restent internes au C++.
2026-08-14 16:48 : [Banc] La gaffe adverse doit figurer dans l'historique du puzzle, sinon la position présentée n'est pas celle du puzzle ; banc régénéré.
2026-08-14 16:53 : [Banc] Sous-échantillonnage à pas régulier et déterministe, le format n'a pas de PuzzleId et random.sample sans seed rendrait deux campagnes incomparables.
2026-08-14 17:54 : [Banc] Référence iter316 : 45.7 % réseau seul et 76.7 % à 700 simulations ; la recherche détruit 48 tactiques vues par la policy contre 823 qu'elle trouve seule.
2026-08-14 18:52 : [Banc] 800 contre 700 simulations, +0.7 point significatif en apparié : ne jamais mélanger les budgets, et la recherche n'est pas saturée, donc le débit se convertira en force.
2026-09-11 14:50 : [UCI] parse_position arrête la recherche avant de modifier le plateau ; la race avec update_root pouvait libérer l'arbre sous les pointeurs d'une descente.
2026-09-11 15:06 : [Perf] Référence avant batching : 286 à 299 simulations par seconde quelle que soit la position, signature d'un coût dominé par la latence d'inférence batch 1.
2026-09-11 15:10 : [TT] Grossir la table ne change presque rien à 400 simulations, saturée dès 65 536 entrées ; à rebalayer après le batching, la conclusion n'est pas acquise.
2026-09-11 15:11 : [Perf] Réserve d'Amdahl consignée : le batching n'accélère que la fraction des simulations qui appellent le réseau, le taux de hits en partie réelle est inconnu.
2026-09-11 17:38 : [MCTS] n_in_flight et détection de collision de feuilles ajoutés ; le virtual loss seul ne garantit pas N feuilles distinctes.
2026-09-14 12:15 : [MCTS] Le virtual loss n'entre que dans le dénominateur du terme U, jamais dans q_value(), sinon Q se dilue vers zéro et avantage les noeuds perdants.
2026-09-14 12:15 : [MCTS] Ne pas réutiliser backup pour appliquer le virtual loss, son signe alterne en remontant ; boucle dédiée qui ajoute +1 partout.
2026-09-14 13:14 : [Perf] Batching activé sur GPU seulement ; sur CPU il ne rend pas et le chemin reste séquentiel.
2026-09-14 13:23 : [MCTS] terminal_hits aligné entre chemin séquentiel et batché, sinon les deux régimes ne sont plus comparables.
2026-09-14 14:13 : [MCTS] Lot de 8 retenu ; le débit GPU sature dès 4 à 8 et la qualité ne bouge pas sur 2500 puzzles contre le séquentiel (McNemar p = 0.73).
2026-09-14 21:36 : [TT] Nulle évitable en finale expliquée : un hit de TT peut servir une value calculée sous un autre compteur de 50 coups ; arbre et TT recréés rétablissent le mat tour contre roi.
2026-09-14 21:36 : [Réseau] Le plan no-progress pèse fort dans le réseau : sur la même position, V passe de +0.64 à compteur 0 à -0.04 à compteur 99.
2026-09-15 09:08 : [TT] Clé sémantique distincte du Zobrist pour le cache réseau ; le Zobrist reste la clé des règles.
2026-09-15 12:56 : [TT] Droits de roque ajoutés aux contrôles de contexte de la clé, comme les répétitions et le flag amnésie.
2026-09-15 19:56 : [TT] h0 retenu : compteur des 50 coups exact, historique exclu de la clé ; 0.16 point sous h1 sur 2500 puzzles et mate la position causale en 31 demi-coups contre une nulle à 100 en legacy. Décision serrée, à confirmer par tournoi.
2026-09-16 17:38 : [MCTS] Propriété des feuilles formalisée par NodeState et PathReservation ; un noeud en attente ne peut plus être traversé ni développé deux fois.
2026-09-16 18:00 : [MCTS] Collecte parallèle des feuilles avec réservation précoce ; le drapeau is_pending de la spec devient un état de noeud, la collision de feuille devient un résultat de collecte compté.
2026-09-17 14:07 : [MCTS] Invariants concurrents verrouillés par un stress test : 1800 recherches, 115 200 simulations, zéro violation.
2026-09-19 15:09 : [Banc] Ordre des workers alterné à chaque passage ; les valeurs absolues varient de 15 à 30 % entre sessions, seuls les rapports intra-session sont exploitables.
2026-09-19 17:25 : [Perf] L'évaluateur représente 96.6 à 98.7 % du temps mural de la recherche ; le GPU reste le goulet, la préparation CPU est négligeable.
2026-09-19 18:28 : [MCTS] Recherche par vagues activée : médiane +8 à +26 % selon la position, p95 de latence sous +5 %, non-infériorité qualité sur 2500 puzzles.
2026-09-19 18:28 : [MCTS] Le gain multicœur plafonne dès 2 workers, et à tranches de 8 simulations 8 workers perdent 19 % par collisions ; le régime réel du bot est à tranches de 64.
2026-09-20 11:14 : [Perf] Lot de forme fixe activé dans l'UCI : x2.0 en ouverture, x3.2 en milieu, x2.5 en finale, p95 de latence divisé par 2 à 3.
2026-09-20 11:20 : [Perf] Décrochage élucidé : changer la forme du lot à chaque appel ONNX coûte 13.4 ms au lieu de 2.7 à 3.7 ms à forme fixe, facteur 4.
2026-09-20 11:23 : [Perf] FP16 rejeté, 18 à 28 % plus lent que FP32 sur le banc évaluateur, sans campagne qualité.
2026-09-20 12:08 : [Perf] Le softmax C++ vaut 0.2 à 2 % du coût de l'appel, ce n'est pas un levier d'optimisation.
2026-09-20 12:08 : [MCTS] Balayage de divergence rejeté : virtual loss 2 et FPU 0.45 donnent +13 à +25 % de débit mais -1.8 à -2.4 points de résolution sur 500 puzzles.
2026-09-20 12:08 : [Perf] Cas chaud mesuré : 700 appels réseau pour 700 simulations même à 81 % de hits de table, un hit matérialise les enfants et la descente continue ; la réserve d'Amdahl est sans objet.
2026-09-20 12:09 : [Perf] Buffers persistants non retenus : la mesure montre que le facteur dominant est la forme du lot, pas les allocations par appel.
2026-09-20 12:09 : [Perf] Production continue (file d'inférence) non décidée ; un microbenchmark de file avec producteurs synthétiques est exigé avant tout chantier de concurrence.
2026-09-20 15:25 : [Entraînement] Pool de self-play gardé plein à 256 parties concurrentes, x1.22 sur la génération.
2026-09-20 16:40 : [UCI] V statique du réseau publié dans les info UCI, distinct du Q moyen de la recherche.
2026-09-20 22:06 : [TT] V des noeuds matérialisés depuis la table renseigné, il manquait dans les info UCI.
2026-09-20 22:36 : [Entraînement] Chemin des puzzles passé depuis Python ; le chemin codé en dur dépendait du répertoire de lancement.
2026-09-22 09:50 : [MCTS] Fuite de réservation R1 élucidée : les unités étaient libérées par une constante globale, donc toute amplitude supérieure à 1 laissait +1 permanent par descente sur les noeuds chauds. Chaque entrée de chemin mémorise désormais ses unités exactes.
2026-09-22 10:03 : [MCTS] Une exception pendant la fusion d'une vague laissait des réservations propriétaires sur une racine locale détruite. Un garde vide les contextes des workers à toute sortie de collect_wave.
2026-09-22 10:09 : [Evaluateur] Les sorties réseau étaient consommées avant tout contrôle ; taille, finitude et ordre de grandeur sont désormais validés avant le premier accès par indice et avant l'insertion en table.
2026-09-22 10:24 : [MCTS] Mat contre règle des 50 coups : à 100 demi-coups, un mat était noté nulle quand les deux conditions se recouvraient. Le mat prime désormais dans la sélection, le backup, la conclusion self-play et l'UCI.
2026-09-22 10:35 : [MCTS] Bruit de racine perdu après un hit de table : le noeud restait Unexpanded, donc le bruit de Dirichlet ne trouvait aucune liste d'enfants, notamment sur les positions de départ. Les enfants sont matérialisés depuis la sonde avant l'application du bruit.
2026-09-22 10:59 : [MCTS] Coups légaux tronqués à la capacité du cache (128) : une position à 218 coups en perdait 90. La liste complète est conservée et la table est contournée au-delà de sa capacité.
2026-09-22 11:02 : [UCI] Identité d'une position fondée sur la base et la suite de coups, plus seulement sur la chaîne FEN reçue, pour que `position startpos moves ...` et la FEN équivalente partagent la même analyse.
2026-09-22 11:08 : [Self-play] Le banc de remplissage opposait à tort un pool qui se vide à un pool renouvelé : les deux régimes relançaient les places, seul le nombre de départs changeait. Le gestionnaire démarrait C + N - 1 parties pour N collectées et jetait les C - 1 dernières. La génération finie démarre exactement N parties, laisse finir celles qui sont engagées et les collecte toutes ; les compteurs started, completed, active et new_plies le rendent visible.
2026-09-22 13:58 : [Perft] Palier de référence figé par position, profondeurs adaptées, et contrôle d'aller-retour `encodeMove(decodeMoveIndex(i))` sur chaque coup légal ; les deux tournent en CTest.
2026-09-22 14:35 : [Binding] Encodage et décodage des coups unifiés en C++ ; la table Python ne servait que de doublon et un `is_black` incohérent lève désormais.
2026-09-22 17:29 : [Self-play] Première campagne sur le build R9-B : longueur moyenne des parties passée d'environ 110 à 128 plies et nulles par la règle des 50 coups de 0.3 à 1.1 pour cent. Ce n'est pas une dérive du réseau mais la fin d'un biais de sélection : l'ancien scheduler démarrait C + N - 1 parties et ne gardait que les N premières terminées, donc éliminait mécaniquement les finales longues, celles qui finissent sur les 50 coups. Le buffer contient désormais ces positions de finale, ce qui répond à une faiblesse de conversion observée en fin de partie, et ces deux courbes ne sont plus comparables aux runs d'avril.
2026-09-22 19:17 : [Binding] `py::make_tuple` dans une région `gil_scoped_release` provoquait une violation d'accès dans le binding self-play avec compteurs ; la valeur de retour est construite en C++ avant de relâcher le GIL.
2026-09-22 20:09 : [MCTS] Rebalayage après R1 : virtual loss 2 gagne 9 à 20 % de débit, abaisse le p95 de latence de 13 à 20 % et passe la barrière qualité sur 2500 puzzles, IC95 [-0,4 ; +0,56]. Le rejet de septembre venait de la fuite R1, pas du réglage ; il est activé dans le bot, le tournoi, la GUI et l'ancrage.
2026-09-22 20:09 : [Self-play] Matrice de pools à budget réduit : (128,256) 192,9, (256,512) 231,2 et (512,512) 254,4 plies nouveaux par seconde, départs = fins = total partout. Le plus grand pool est le plus rapide, contre l'intuition du genou à 128 ; confirmation à 700/100 requise avant de changer la production.
2026-09-22 20:09 : [Banc] Confondant mesuré sur iter436, 2500 puzzles : prior du coup solution 0,288 avec historique contre 0,516 sans, value -0,216 contre +0,930, résolution +6,6 points sans historique, p = 3e-13. C'est la référence à faire converger vers zéro après réentraînement.
2026-09-22 20:19 : [MCTS] Le virtual loss est inerte en self-play : chaque arbre n'y est descendu qu'une fois par vague, donc n_in_flight n'a aucun lecteur entre la réservation et sa libération. Seuls les chemins qui collectent plusieurs feuilles du même arbre, le mono batché et les vagues du bot, sont concernés. La propagation au self-play a été retirée.
2026-09-22 22:15 : [MCTS] Revue externe : une erreur de l'évaluateur en pleine descente scalaire laissait le plateau sur la feuille alors que l'arbre pointait la racine. Garde RAII armé par select_leaf, partagé avec les vagues et advance_to_leaf, plus un test d'erreur au deuxième appel réseau avec reprise sur le même arbre.
2026-09-22 22:15 : [Banc] La barrière qualité du virtual loss 2 avait tourné en mono batché (search_workers 1, seize processus Python), pas sur le chemin déployé. Les mêmes 500 lignes rejouées en vagues de 8 workers donnent 367 contre 369 réussites, IC95 [-0,4 ; +1,2], non-infériorité ; les 2500 en vagues restent la réserve.
2026-09-22 22:15 : [Self-play] Le bruit différé signalé par la revue est un filet, pas un trou : reset_game et play_best_move développent toujours la racine avant de poser le bruit, et extract_child ne rend qu'un nœud déjà visité donc développé. Un test verrouille l'invariant sur la racine réutilisée.
2026-09-23 14:05 : [Banc] Validation vagues complète : les 2500 puzzles rejoués avec 8 workers donnent 1924 contre 1931, IC95 [-0,2 ; +0,76], McNemar p = 0,32, quasi identiques au mono (1926 contre 1928). La réserve de validation du chemin déployé est levée.
2026-09-23 15:05 : [Self-play] Confirmation du pool à 700/100, horizon égal de 512 parties : (512, 512) donne 35,6 plies nouveaux par seconde et 28,09 ms par ply contre 32,8 et 30,47 pour (256, 512), soit x1,09, départs = fins = total partout. Le gain du grand pool se confirme au budget réel, du même ordre que la mesure à 100/20 (x1,10) ; la production reste 256 en attendant la décision.
