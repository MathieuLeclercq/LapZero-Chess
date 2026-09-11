# Notes de conception : batcher la recherche côté UCI

Date : 2026-08-08
Statut : notes préparatoires, pas encore une spec validée

Le plus gros écart de performance du projet. Ce document consigne le diagnostic, la
mécanique du virtual loss, et les pièges spécifiques à ce code.

## Le diagnostic

Il existe exactement deux points d'entrée vers le GPU :

| Site | Batch | Utilisé par |
|---|---|---|
| `selfplay_manager.cpp:88` → `evaluate_batch(..., 512)` | **512** | self-play uniquement |
| `mcts.cpp:176` → `evaluate(...)` → `evaluate_batch(..., 1)` | **1** | `mcts_search` et `step_analysis`, donc le bot en partie, la GUI, le tournoi et le futur banc |

Le bot qui joue sur lichess fait donc **une inférence GPU par simulation**. Le
`BATCH_SIZE = 20` de `uci.py:17` n'est que la granularité de la boucle Python : il appelle
`step_analysis(board, 20, 1.4)`, qui fait 20 simulations séquentielles, chacune avec sa
propre inférence de batch 1. Le nom est trompeur et devrait changer.

**Reserve sur l'ampleur du gain, ajoutee le 2026-09-11.** Le taux de succes de la
table mesure par le banc de recherche, 2,7 a 10,7 pour cent, porte sur un arbre neuf.
En partie reelle `update_root` conserve le sous-arbre du coup joue, donc la recherche
suivante demarre chaude et le taux est plus eleve, particulierement en finale. Par la
loi d'Amdahl, le batching n'accelere que la fraction des simulations qui appellent
reellement le reseau : le gain reel sera donc inferieur a ce que laisse croire le
calcul ci-dessous. La direction ne change pas, seule l'ampleur. Voir la section
« Reserve sur le cas froid » de `2026-09-11-search-bench-resultats.md`.

Ordre de grandeur du gain : une inférence batch 1 sur un ResNet 10x128 est dominée par la
latence de lancement des noyaux, quelques millisecondes. À batch 32 ou 64, le coût par
position s'effondre. Attendre un facteur 10 à 40 sur le nombre de simulations à temps
constant, ce qui se traduit directement en Elo.

## La machinerie existe déjà

`SelfPlayManager` fait exactement ce travail, mais **entre parties**, pas dans un même
arbre. Les deux fonctions clés sont déjà écrites et testées :

- `MCTS::advance_to_leaf(root, board, c_puct, moves_played)` : descend, gère les cas
  terminaux et les hits de TT en interne (il fait le backup lui-même et renvoie `nullptr`),
  et sinon renvoie la feuille à évaluer en laissant le plateau **sur cette feuille**.
- `MCTS::expand_and_backup(leaf, board, policy, value)` : crée les enfants depuis la policy,
  écrit la TT, et remonte la value.

Ce qui manque pour batcher **dans un seul arbre** : le virtual loss. Sans lui, N descentes
successives sans mise à jour des statistiques choisissent toutes le même chemin et donc la
même feuille, et le batch ne contient que N copies de la même position.

## Le virtual loss, dérivé sur ce code précis

Ne pas recopier une formule d'ailleurs : la convention de signe dépend du code. Voici la
dérivation sur `MCTSNode::ucb_score` tel qu'il est écrit.

```cpp
float u = exploration_factor * prior / (1.0f + visit_count);
float exploitation = (visit_count == 0) ? (parent_q - fpu_reduction) : -q_value();
return exploitation + u;
```

Pour un enfant déjà visité, le score vu par le parent vaut `-q_value() + u`. La sélection
prend le maximum. Pour rendre un enfant **moins** attractif il faut donc **augmenter**
`q_value() = total_value / visit_count`.

Donc, sur chaque noeud du chemin descendu :

```cpp
n->visit_count += VIRTUAL_LOSS;
n->total_value += VIRTUAL_LOSS;   // +1 : le joueur au trait a ce noeud "gagne",
                                  // donc le coup qui y mene devient moins attractif
```

Les deux termes poussent le score vers le bas : `total_value` augmente donc `-q_value()`
diminue, et `visit_count` augmente donc `u` diminue. C'est cohérent à chaque niveau du
chemin, sans alternance de signe.

**Attention, on ne peut pas réutiliser `backup()` pour ça.** `backup(node, value)` alterne
le signe en remontant (`value = -value`), ce qui donnerait +1, -1, +1... alors qu'on veut
+1 partout. Il faut une boucle dédiée :

```cpp
void MCTS::apply_virtual_loss(MCTSNode* leaf) {
    for (MCTSNode* n = leaf; n != nullptr; n = n->parent) {
        n->visit_count += VIRTUAL_LOSS;
        n->total_value += static_cast<float>(VIRTUAL_LOSS);
    }
}

void MCTS::revert_virtual_loss(MCTSNode* leaf) {
    for (MCTSNode* n = leaf; n != nullptr; n = n->parent) {
        n->visit_count -= VIRTUAL_LOSS;
        n->total_value -= static_cast<float>(VIRTUAL_LOSS);
    }
}
```

Le virtual loss est retiré juste avant le backup réel, quand la value arrive du GPU.

`VIRTUAL_LOSS = 1` est le point de départ (une défaite pleine, les values étant dans
[-1, 1]). C'est un paramètre à régler : trop faible et les descentes se regroupent, trop
fort et la recherche s'éparpille.

Détail sur la racine : lui appliquer le virtual loss fausse transitoirement `parent_q`, qui
alimente le FPU. L'effet est négligeable et la plupart des implémentations l'appliquent
quand même, mais on peut choisir de n'incrémenter que `visit_count` à la racine.

## Structure de la boucle

```
tant que simulations restantes :
    // 1. Collecte
    feuilles.clear()
    tant que feuilles.size() < TAILLE_BATCH et tentatives < PLAFOND :
        feuille = advance_to_leaf(racine, plateau, c_puct, coups_joues)
        si feuille == nullptr :            // terminal ou hit de TT, deja traite
            simulations_faites++
            continuer
        si feuille est deja dans feuilles : // doublon malgre le virtual loss
            annuler la descente et sortir de la collecte
        appliquer_virtual_loss(feuille)
        construire le tenseur PENDANT que le plateau est sur la feuille
        enregistrer (feuille, coups_joues)
        annuler les coups_joues pour revenir a la racine

    // 2. Une seule inference
    evaluate_batch(tenseurs, policies, values, feuilles.size())

    // 3. Backup
    pour chaque feuille :
        rejouer son chemin OU ne pas en avoir besoin (voir piege 3)
        retirer_virtual_loss(feuille)
        expand_and_backup(feuille, plateau, policy_i, value_i)
        annuler le chemin
        simulations_faites++
```

## Pièges spécifiques à ce code

### 1. Une feuille collectée peut recevoir des enfants pendant la même collecte

`select_leaf` fait de l'expansion paresseuse : quand un noeud n'a pas d'enfants, il consulte
la TT et, en cas de hit, **crée les enfants et continue de descendre** (`mcts.cpp:83-105`).
Une feuille collectée pour le GPU peut donc se voir attribuer des enfants par une descente
ultérieure du même batch, après quoi `expand_and_backup` en créerait un **second** jeu.

Correctif : marquer les feuilles collectées comme « en attente » et faire que `select_leaf`
s'arrête dessus sans les traverser, à la manière du drapeau `is_terminal` existant. Ajouter
un `bool is_pending` à `MCTSNode` est le plus simple.

C'est le piège le plus sérieux de tout le chantier, et il ne se manifesterait pas par un
plantage mais par un arbre corrompu et des priors dupliqués.

### 2. La race UCI devient beaucoup plus dangereuse

Aujourd'hui `parse_position` (`uci.py:97`) modifie `self.board` et appelle
`mcts.update_root()` sans arrêter le thread de recherche (voir `docs/backlog.md` §3).

Avec le batching, la boucle détient des **pointeurs bruts `MCTSNode*`** pendant tout l'appel
GPU, qui dure plusieurs millisecondes. Si `update_root` s'exécute pendant ce temps, il
appelle `extract_child` puis détruit le reste de l'arbre par `unique_ptr`, et les pointeurs
collectés deviennent pendants. Écriture dans de la mémoire libérée.

**Corriger la race avant de batcher.** Un `self.stop_search()` en tête de `parse_position`,
plus tenir `m_mutex` sur tout le cycle collecte / inférence / backup, ou introduire un
compteur de génération invalidant un batch en cours.

### 3. Le plateau pendant le backup

`expand_and_backup` appelle `board.getLegalMoveIndices()` et `board.isInCheck()`, donc il
exige que le plateau soit **sur la position de la feuille**. Deux options :

- rejouer le chemin de chaque feuille avant son backup, puis l'annuler. Coût : deux
  traversées par feuille.
- capturer les indices légaux au moment de la collecte, quand le plateau y est déjà, et les
  passer à une variante de `expand_and_backup` qui ne touche plus au plateau. Plus rapide et
  plus sûr, mais demande de modifier la signature.

La seconde option est préférable et supprime un aller-retour make/unmake par feuille.

### 4. `mcts_search` et `step_analysis` partagent le même noyau

Les deux appellent `select_leaf` et `expand_node_single`. Batcher les deux d'un coup est
tentant mais double la surface. Ordre suggéré : écrire la boucle batchée comme une
troisième fonction, la brancher d'abord sur `step_analysis` (le chemin du bot), vérifier
le gain, puis basculer `mcts_search`.

Ne pas oublier que `expand_node_single` restera utilisé pour développer la racine, et que
lui reste en batch 1. Ce n'est pas grave, c'est un appel par coup joué.

### 5. Le cpuct peut devoir être retouché

Collecter N feuilles avant de mettre à jour les statistiques revient à retarder
l'information. L'exploration effective change avec la taille de batch, et les
implémentations existantes observent qu'un batch plus grand demande un cpuct légèrement
plus élevé. Le `1.4f` en dur dans `uci.py:288` et dans `selfplay_manager.cpp:345` devra
être revérifié.

### 6. Taille de batch et gestion du temps

Le batch remplit d'autant moins bien que l'arbre est petit : au tout début d'une recherche,
il n'y a pas N feuilles distinctes à trouver. Prévoir un plafond de tentatives et accepter
des batchs partiels, comme le fait déjà `SelfPlayManager` avec sa condition
`batch_full || all_blocked`.

Côté gestion du temps dans `uci.py`, le contrôle d'arrêt s'exerce entre deux batchs, donc la
granularité d'arrêt devient la durée d'un batch au lieu d'une simulation. Avec un batch de
32 c'est négligeable, mais `target_time` très court (fins de partie en blitz) mérite un
batch réduit.

## Comment mesurer le gain

Avant toute chose, **chiffrer l'état actuel** : nombre de simulations par seconde de
`step_analysis` sur une position fixe. C'est la mesure qui manque au dossier, faute d'un
modèle ONNX sur la machine où l'analyse a été faite. L'obtenir par
`lib.export_model_to_onnx` sur un checkpoint récent, puis chronométrer.

Ensuite, deux instruments :

- **simulations par seconde**, à position et temps fixés, pour le gain brut ;
- **le banc de puzzles**, pour vérifier que le gain se traduit en qualité de jeu et non
  seulement en volume. Le virtual loss modifie le comportement de la recherche, donc un
  gain en simulations ne garantit pas un gain en force. C'est précisément ce que le banc
  sert à trancher.

Le perft ne sert à rien ici : il ne valide que la génération de coups, pas la recherche.

## Ordre d'exécution recommandé

1. Corriger la race `parse_position` (une ligne, voir `docs/backlog.md` §3).
2. Mesurer les simulations par seconde actuelles.
3. Construire le banc de puzzles.
4. Ajouter `is_pending` à `MCTSNode` et le respect de ce drapeau dans `select_leaf`.
5. Ajouter virtual loss, boucle batchée, brancher sur `step_analysis`.
6. Mesurer les deux instruments, régler `VIRTUAL_LOSS`, `TAILLE_BATCH` et `c_puct`.
7. Basculer `mcts_search`.
