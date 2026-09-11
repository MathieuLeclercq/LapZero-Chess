# Pipeline de mesure et de vérification de la recherche : conception

Date : 2026-09-11
Statut : spec validée, prête pour le plan d'implémentation

Établir une référence reproductible du débit de la recherche MCTS, et un filet qui détecte
une régression, avant d'entreprendre le batching avec virtual loss décrit dans
`2026-08-08-uci-batching-notes.md`.

## 1. Pourquoi maintenant

Le batching est le plus gros levier de performance du projet, et il touche le coeur du
MCTS. Deux raisons de construire le filet d'abord.

**Quantifier.** Sans référence mesurée sur les mêmes positions et avec la même méthode, un
gain annoncé ne serait qu'une impression.

**Ne rien casser en silence.** Le piège principal du batching ne provoque pas de plantage.
`select_leaf` fait de l'expansion paresseuse et continue de descendre après un succès de
table de transposition, donc une feuille déjà collectée pour le GPU peut recevoir des
enfants pendant la même collecte, après quoi `expand_and_backup` en créerait un second jeu.
L'arbre serait corrompu et les priors dupliqués, sans le moindre symptôme visible.

## 2. Ce qui existe, et ce qui manque

`python_src/benchmark_release_debug.py` est **cassé** : il appelle
`chess_engine.MCTS(ONNX_PATH)` alors que le constructeur attend un `ONNXEvaluator` et non
une chaîne, et il pointe vers un ONNX absent. C'est du code mort d'une ancienne API.

Les mesures de la session précédente étaient ad hoc, dans un répertoire temporaire, et les
scripts n'existent plus. Il n'en reste que des chiffres, consignés dans les rapports :

| Mesure | Valeur |
|---|---|
| `mcts_search`, 400 simulations, un processus, GPU | 376 sims/s |
| `mcts_search`, 400 simulations, un processus, CPU | 373 sims/s |
| Débit agrégé, 16 processus, CPU | 2282 sims/s |

Ces chiffres suffisent à décider, pas à servir de référence avant et après. Rien ne mesure
`step_analysis`, qui est pourtant le chemin qu'emprunte `uci.py`.

## 3. Périmètre retenu

| Décision | Choix |
|---|---|
| Introspection de l'arbre | méthode C++ `inspect_tree`, avec recompilation |
| Instrumentation | compteurs C++ d'inférences et de succès de table |
| Chemins mesurés | `mcts_search` **et** `step_analysis` |
| Barrière qualité | deux niveaux, 250 puzzles en rapide, 2500 avant fusion |
| Benchmark cassé | supprimé |
| Race `parse_position` | **corrigée ici**, en première tâche |

### La race `parse_position` fait partie de ce chantier

Elle était jusqu'ici rangée en préalable implicite du batching (`docs/backlog.md` §3). Elle
devient la première tâche de ce pipeline, pour deux raisons.

`parse_position` (`uci.py:97`) appelle `mcts.update_root()` et reconstruit `self.board` sans
arrêter le fil de recherche, alors que `stop_search()` existe déjà (`uci.py:245`) et joint
proprement le fil. Le correctif tient en une ligne au début de la méthode.

Aujourd'hui c'est un bug latent qui se manifeste rarement. Avec le batching, la boucle
détiendra des `MCTSNode*` bruts pendant plusieurs millisecondes d'appel GPU, et `update_root`
détruit l'arbre sous ces pointeurs : cela deviendrait une écriture en mémoire libérée. La
corriger maintenant, pendant qu'on touche déjà à ce code et qu'aucune pression de
performance ne s'exerce, coûte moins cher que de la corriger en plein chantier de batching.

## 4. Architecture

### Côté C++, deux additions

**1. L'introspection de l'arbre.** `MCTS::inspect_tree() const` parcourt l'arbre d'analyse et
renvoie un `TreeReport`, sur le modèle exact de `PerftReport` déjà en place :

```cpp
struct TreeReport {
    uint64_t nodes = 0;
    uint64_t max_depth = 0;
    uint64_t violations = 0;
    std::vector<std::string> messages; // tronque, comme MAX_PERFT_MESSAGES
};
```

**2. Des compteurs, sans lesquels deux des trois grandeurs annoncées seraient
inobservables.** Le moteur n'a aujourd'hui **aucune instrumentation** : ni compteur d'appels
au réseau, ni compteur de succès de table. Le nombre d'inférences par seconde et le taux de
succès de la table ne peuvent donc pas être déduits de l'extérieur.

```cpp
struct SearchCounters {          // membres de MCTS, en std::atomic (voir ci-dessous)
    uint64_t nn_calls = 0;      // appels reels a m_evaluator->evaluate
    uint64_t tt_hits = 0;       // consultations de table reussies
    uint64_t tt_misses = 0;     // consultations echouees
    uint64_t terminal_hits = 0; // simulations arretees sur un noeud terminal
};
```

**Les compteurs doivent être des `std::atomic<uint64_t>` en `memory_order_relaxed`.** Le
self-play appelle `m_shared_mcts->advance_to_leaf` depuis une région OpenMP à 8 fils
(`selfplay_manager.cpp:376-387`) sur une instance de `MCTS` partagée. Des entiers simples y
seraient une course de données, donc un comportement indéfini et des comptes perdus.

Le coût est négligeable : un incrément atomique relâché vaut quelques dizaines de cycles,
face à une inférence de 2,7 ms. `get_counters()` renvoie une copie non atomique, un instantané
suffisant pour un rapport.

Avec `MCTS::get_counters() const` et `MCTS::reset_counters()`. Les compteurs sont de simples
incréments sur des chemins déjà existants, aux quatre sites suivants :

| Site | Compteur |
|---|---|
| `select_leaf:88`, expansion paresseuse, branche de succès | `tt_hits` |
| `expand_node_single:163`, branche de succès | `tt_hits` |
| `expand_node_single:176`, appel à `evaluate` | `nn_calls`, et `tt_misses` sur la branche |
| `advance_to_leaf:460`, chemin batché du self-play | `tt_hits` ou `tt_misses` |

Il y a bien **trois** sites de consultation de la table et non deux, l'expansion paresseuse
de `select_leaf` étant facile à oublier. Les compter tous les trois est nécessaire pour que
le taux affiché soit celui de la recherche et non d'une partie d'elle.

Aucune modification du code de recherche existant. Le parcours reste là où vit l'arbre, et
le côté Python se réduit à une assertion sur `violations`.

**`inspect_tree` porte sur `m_analysis_root`, et sur lui seul.** C'est le seul arbre qui
survit à un appel : `mcts_search` construit sa racine en variable locale (`mcts.cpp:237`)
et la détruit en revenant, il n'y aurait donc rien à inspecter après coup. La jambe
invariants s'exerce en conséquence sur le chemin `step_analysis`.

Ce n'est pas une limitation en pratique : `step_analysis` est le chemin réel du bot, et les
notes de batching prévoient déjà de brancher la boucle batchée sur lui en premier. La jambe
débit, elle, continue de mesurer les deux chemins.

Les deux alternatives écartées : exposer un vidage plat de l'arbre à Python, plus souple
mais dont le transfert deviendrait le coût dominant sur un arbre de plusieurs dizaines de
milliers de noeuds ; et un binaire autonome sur le modèle de `chess_perft`, qui ne pourrait
pas être piloté par le harnais Python et devrait recharger son propre ONNX.

### Côté Python

`python_src/search_bench.py` : mesure de débit, vérification d'invariants, CLI et rapport.
Il réutilise `puzzle_bench.resoudre_modele` pour l'export ONNX, donc aucune duplication de
la logique de chargement de modèle.

`python_src/benchmark_release_debug.py` est supprimé.

## 5. Ce qui est mesuré

**Trois positions fixes et commitées**, couvrant ouverture, milieu de jeu et finale. Un
débit mesuré sur la seule position de départ ne représente pas une partie : le nombre de
coups légaux, le taux de succès de la table et la profondeur de l'arbre y sont atypiques.

**Deux chemins, séparément.** `mcts_search`, qui repart d'un arbre neuf, et `step_analysis`
suivi de `update_root`, qui réutilise l'arbre et qui est le chemin réel du bot. Ils ne se
comportent pas pareil vis-à-vis du cache.

**Trois grandeurs, nommées sans ambiguïté.** Le mot « noeuds par seconde » est proscrit dans
ce harnais : le perft mesure la génération de coups à plus de 1,6 million de noeuds par
seconde, la recherche tourne à 375 simulations par seconde, et confondre les deux serait une
erreur d'un facteur 4000.

- **simulations par seconde** : le débit de la recherche ;
- **inférences par seconde** : le nombre d'appels réels au réseau, lu dans `nn_calls` ;
- **taux de succès de la table de transposition** : `tt_hits / (tt_hits + tt_misses)`, soit
  la part des descentes qui évitent le réseau. C'est la grandeur que le batching déplacera
  le plus, puisqu'il change le rapport entre descentes et évaluations.

Les deux dernières viennent des compteurs ajoutés en section 4. Elles ne se déduisent pas du
nombre de simulations : une simulation peut se terminer sur un nœud terminal ou sur un succès
de table, auquel cas elle ne coûte aucune inférence. C'est exactement ce rapport que le
batching va modifier, donc le mesurer avant est indispensable.

**La variance.** Chaque configuration est mesurée sur plusieurs passages, avec médiane et
étendue. Sans cela, un gain de 5 pour cent serait indistinguable du bruit.

## 6. Les invariants

Le vérificateur compte les violations. Il s'exerce sur l'arbre d'analyse, après un
`step_analysis`, pour la raison exposée en section 4. Les contrôles ci-dessous tiennent par
construction dans le code séquentiel actuel.

- **Aucun enfant dupliqué** dans un noeud, c'est-à-dire deux entrées partageant le même
  `move_idx`. C'est le contrôle qui attrape le piège décrit en section 1, et le plus
  important de tous.
- **Pointeurs parents cohérents** : `child->parent == node`.
- **Noeuds terminaux sans enfants.**
- **`move_idx` dans [0, 4671]** pour tout noeud non racine.
- **Priors sommant à 1** sur les enfants d'un noeud, `expand_node_single` divisant par
  `sum_legal`.

**Conservation des visites, sous forme encadrée.** Pour tout noeud,
`somme des visites des enfants <= visit_count <= 1 + somme des visites des enfants`.
L'encadrement est imposé par l'expansion paresseuse : selon qu'un noeud a été développé en
tant que feuille (défaut de table) ou traversé en créant ses enfants au vol (succès de
table), il a reçu ou non une visite propre. Un jeu d'enfants dupliqué violerait la borne
haute.

**Visites de la racine.** `step_analysis` crée la racine si elle est absente et la développe
par `expand_node_single` **sans backup** (`mcts.cpp:374-377`), puis chaque simulation remonte
exactement une fois par la racine. Donc après un `reset_analysis()` suivi d'un seul
`step_analysis(board, N)`, `m_analysis_root->visit_count` vaut exactement `N`, et `k` appels
successifs sans `update_root` donnent `k * N`.

C'est le contrôle qui détectera un virtual loss mal annulé, puisqu'un virtual loss résiduel
gonflerait ce compte sans rien casser d'autre. `mcts_search` a la même structure
(`mcts.cpp:237-238`), mais sa racine étant locale elle n'est pas inspectable.

### Passage de découverte

Le premier lancement du vérificateur porte sur la recherche séquentielle actuelle, qui fait
référence. Il n'est pas une barrière mais une mesure : tout invariant qui ne tiendrait pas
signale soit un bug latent, soit une erreur dans le modèle mental ci-dessus, et dans les
deux cas il vaut mieux l'apprendre avant de toucher au batching. Les invariants confirmés
deviennent ensuite la barrière.

## 7. Sorties et barrières

Un rapport markdown unique, `docs/superpowers/specs/<date>-search-bench-resultats.md`. Les
données tiennent en quelques dizaines de nombres, donc pas de CSV. Le fichier est régénéré
après le batching et **le diff git montre directement le gain**.

| Barrière | Commandes | Durée |
|---|---|---|
| Rapide, à chaque étape | `search_bench.py --invariants` puis `puzzle_bench.py --limite 250` | environ 1 min |
| Complète, avant fusion | `search_bench.py` puis `puzzle_bench.py` | environ 12 min |

## 8. Critère d'acceptation du batching

Non implémentable maintenant, mais acté ici pour que le chantier suivant s'y tienne.

**À taille de batch 1, la boucle batchée doit produire un résultat identique au bit près à
la boucle séquentielle.** Avec un seul élément, la collecte se réduit à une descente, le
virtual loss est posé puis annulé immédiatement, et le backup est le même. Ce test valide
d'un coup la machinerie de batch et la symétrie du virtual loss, et il ne coûte presque rien
à écrire le moment venu.

Il faut noter qu'au delà de batch 1 l'équivalence n'a plus lieu d'être : le virtual loss
change délibérément l'ordre des descentes. Comparer les sorties à nombre de simulations égal
serait donc une erreur de méthode, et c'est le banc de puzzles, pas une égalité, qui tranche
la qualité.

## 9. Ce que ce pipeline ne dira pas

- Rien sur la force en partie réelle. Un débit et un taux de résolution de puzzles ne sont
  pas un Elo.
- Rien sur la génération de coups, déjà couverte par la suite perft.
- Rien sur le self-play, qui batche déjà par 512 entre parties et n'est pas concerné par ce
  chantier.

## 10. Risques

**La recompilation devient obligatoire.** `inspect_tree` et les compteurs imposent de
reconstruire le module avant de pouvoir lancer le harnais. Le `.pyd` est suivi par git, donc
il doit être commité avec la source pour que les deux ne se désynchronisent jamais.

**Les compteurs touchent un chemin partagé.** Ils sont incrémentés depuis `select_leaf` et
`advance_to_leaf`, donc aussi pendant le self-play, en parallèle. D'où le choix d'atomiques
relâchés. Il faut vérifier après coup que le débit de self-play n'a pas bougé, sinon
l'instrumentation aurait dégradé ce qu'elle mesure.

**Le coût du parcours.** `inspect_tree` visite tout l'arbre, donc son coût croît avec le
nombre de simulations. Il ne doit jamais être appelé dans la boucle de mesure de débit, sous
peine de fausser la mesure qu'il est censé protéger. Les deux jambes sont lancées
séparément.
