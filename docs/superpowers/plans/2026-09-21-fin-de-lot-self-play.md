# Fin de lot self-play (R9) : plan d'implementation

> **Pour les agents charges de l'execution :** executer les taches en ligne, une par
> une, avec le skill `superpowers:executing-plans` si disponible. Aucun sous-agent.
> Les commits listes sont crees au moment de l'execution, sur demande explicite.

**Objectif :** rendre la generation self-play honnete et finie : compter les departs, les
fins et les places actives, caracteriser les parties abandonnees, puis faire produire a
`generate_games(N)` exactement N parties entieres, sans jeter de parties engagees.

**Architecture :** le lot se fait en deux temps, comme le recommande le plan d'audit.
R9-A ajoute des compteurs et un banc de caracterisation, sans changer le scheduler. R9-B
remplace la relance systematique par un etat actif par place et un arret quand le quota de
departs est atteint. La recherche, le tuning, le format des exemples et l'API Python ne
changent pas.

**Technologies :** C++17, MSVC, CMake/CTest, OpenMP, pybind11, Python 3.13, pytest, uv,
ONNX Runtime 1.24.3 CUDA. Les tests de logique doivent tourner sans modele ni GPU.

**References :** `docs/superpowers/plans/2026-09-21-audit-correctifs-mcts.md` section 11
(R9, constats et recommandations), section 2 (methode de test) et section 13 (validation) ;
`python_src/dev_tools/selfplay_refill_bench.py` ; commit `973a1bd`.

**Base :** HEAD apres le commit des documents. R9 est independant de R1 et de R2 : il peut
etre execute avant, pendant ou apres le rebalayage de virtual loss.

**Statut du document :** plan prepare le 2026-09-21, a partir d'une lecture statique. Aucune
compilation ni campagne n'a ete executee pour le rediger.

## Contexte etabli

Dans `src/selfplay_manager.cpp` :

- le constructeur initialise C places et appelle `reset_game` pour chacune, donc C parties
  demarrent avant meme que le quota N soit connu ;
- `generate_games` relance la place de chaque partie terminee tant que
  `games_completed < N`, y compris quand `N == C` ;
- le nombre de departs vaut donc C + N - 1 pour N superieur a 0 ;
- le gestionnaire est local au binding `generate_self_play_games`
  (`src/bindings.cpp:293`), donc les C - 1 parties encore actives quand la boucle
  s'arrete sont detruites avec lui et jamais collectees ;
- les parties encore actives a la coupure sont, par construction, les plus longues. Le
  biais plausible vers les parties courtes dans les donnees conservees n'est pas quantifie.

Deux consequences :

1. Du travail d'inference est paye pour des parties qui ne produiront aucun exemple.
   En production actuelle, 256 places et 512 parties font 767 departs pour 512 resultats.
2. Les mesures du banc `selfplay_refill_bench.py` attribuent le gain a un renouvellement
   des places, alors que les deux regimes compares relancent les places. Ce qui change
   entre les deux, c'est le nombre de departs et l'horizon, pas l'existence du
   renouvellement. L'interpretation doit etre corrigee avant d'en tirer une conclusion
   d'entrainement.

## Decision de conception

Variante retenue : **generation finie**. Demarrer exactement N parties, relancer une place
tant qu'il reste des departs a effectuer, puis laisser les parties restantes se terminer et
les collecter toutes. Les places sans partie deviennent inactives et ne participent plus
aux descentes. Les lots GPU peuvent retrecir en fin de generation, c'est accepte.

Variante ecartee a ce stade : un gestionnaire persistant qui garde ses parties actives
entre les appels. Elle exige de definir les transitions de modele, de table de
transposition, de cibles d'entrainement et de reprise apres interruption. Elle ne doit pas
etre improvisee pour eviter la fin de lot.

## Contraintes globales

- Ne pas modifier la taille du pool, les simulations slow/fast, le ratio, le bruit de
  Dirichlet, l'amnesie a 1 % ni le renfort tactique du premier coup. Ces reglages se
  decident dans un lot separe, avec des mesures comparables.
- Ne pas modifier la recherche ni le noyau MCTS. Ce lot ne touche que l'ordonnancement des
  parties et la comptabilite.
- Preserver le format de `GameResult`, le type de retour Python de
  `generate_self_play_games` et les champs utilises par `convert_game_results`.
- Les tests C++ utilisent `tests/cpp/controlled_evaluator.hpp` et les acces de test
  existants, sans modele ni GPU. Une tache qui exige un modele reel le dit explicitement.
- Pas de campagne self-play longue automatique. Les mesures de la tache 5 se font a budget
  reduit (100 simulations lentes, 20 rapides) sur des matrices de tailles modestes, comme
  le fait deja `selfplay_refill_bench.py`.
- Un seul processus de mesure GPU a la fois, meme session pour les comparaisons, ordre des
  configurations alterne.
- Pas de tiret cadratin dans les textes ajoutes. Pas de ligne `Co-Authored-By`.
- Tests C++ en Release : controle par exception, jamais `assert()` supprime par `NDEBUG`.

## Fichiers

| Fichiers | Responsabilite |
|---|---|
| `src/selfplay_manager.hpp`, `src/selfplay_manager.cpp` | compteurs, places actives, generation finie |
| `src/bindings.cpp` | variante de diagnostic des compteurs, sans changer le retour de production |
| `tests/cpp/test_selfplay_shared_core.cpp` | transitions deterministes et matrices de quotas |
| `python_src/dev_tools/selfplay_refill_bench.py` | en-tete corrige, compteurs, matrice (C, N), profil de lots |
| `python_src/tests/` | test du rapport du banc si la logique de presentation est extraite |
| `python_src/train_self_play.py` | seulement si une decision ulterieure change les parametres |
| `docs/devlog.md`, `docs/backlog.md`, `CHANGELOG.md` | rectification et cloture |
| `docs/superpowers/specs/2026-09-21-fin-de-lot-self-play-resultats.md` | mesures et decision, cree en tache 6 |

## Commandes communes

```powershell
$cmakeExe = 'C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
$ctestExe = 'C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe'
& $cmakeExe -S . -B build
& $cmakeExe --build build --config Release
& $ctestExe --test-dir build -C Release --output-on-failure
uv run pytest python_src/tests -q
uv run python python_src/dev_tools/selfplay_refill_bench.py --sims 100 --fast 20 --paires "32:1,32:4,64:1,64:4"
```

## Tache 1 : compteurs de diagnostic

**Fichiers :** modifier `src/selfplay_manager.hpp`, `src/selfplay_manager.cpp`,
`src/bindings.cpp`, `tests/cpp/test_selfplay_shared_core.cpp`.

Interface proposee, a ajuster pendant l'implementation si le besoin est plus simple :

```cpp
struct SelfPlayStats {
    std::uint64_t games_started = 0;    // parties reellement demarrees
    std::uint64_t games_completed = 0;  // parties terminees et enregistrees
    std::uint32_t active_slots = 0;     // places avec une partie en cours
    std::uint64_t new_plies = 0;        // coups joues par la recherche, hors rejeu
    std::uint64_t replayed_plies = 0;   // historique de puzzle rejoue au reset
};
SelfPlayStats get_stats() const;
```

`new_plies` et `replayed_plies` repondent a la reserve de l'audit : dans
`GameResult.total_real_moves`, l'historique de la partie source d'un puzzle est inclus, ce
n'est donc pas un compteur de travail nouveau. Les increments se font sur les
transitions reelles : demarrage de partie, fin de partie, desactivation d'une place, coup
joue par `play_best_move`, rejeu d'un puzzle.

- [ ] **Ecrire le test en echec.** Avec un faux evaluateur et des budgets minimaux, lancer
      une generation de 3 parties sur 2 places. Attendre `games_started == 4`,
      `games_completed == 3`, `active_slots == 0`. Le test rougit d'abord par absence de
      `get_stats`, puis verrouille le comportement actuel une fois les compteurs ajoutes.
      La tache 4 mettra l'attente a `games_started == 3` : c'est la mesure du changement,
      pas un test a supprimer.
- [ ] **Test de sensibilite.** Une variante qui ne compte pas une transition doit faire
      echouer le test. Preferer une mutation locale du test, jamais un etat global mutable.
- [ ] **Implementer** les compteurs et un acces de diagnostic. Ne pas ajouter ces champs a
      `GameResult` et ne pas changer le type de retour de `generate_self_play_games` :
      passer par une variante de binding dediee aux tests ou par une sortie distincte.
- [ ] **Verifier** `ctest -R selfplay_shared_core` et la suite Python complete.
- [ ] **Commit :** `Compte les departs et les fins du self-play`.

## Tache 2 : caracterisation deterministe du comportement actuel

**Fichiers :** modifier `tests/cpp/test_selfplay_shared_core.cpp` ; mesures dans
`out/multicore/`.

- [ ] Figer par des parties scriptees, sans reseau :
      `(C, N) = (2, 3)` donne 4 departs et 3 collectes ;
      `(C, N) = (4, 4)` donne 7 departs et 4 collectes ;
      `(C, N) = (4, 1)` donne 4 departs pour 1 collecte. Apres la tache 4, les attentes
      deviennent 3, 4 et 1 : ces tests documentent le changement, ils ne sont pas jettes.
- [ ] Verifier que les parties restantes ne sont pas dans le vecteur de retour et qu'aucun
      exemple ne leur est associe. Le test doit echouer si la collecte des C - 1 parties
      restantes etait deja faite par erreur.
- [ ] Mesurer, sur une execution courte reelle (modele ONNX, 100/20 simulations, 32 places
      et 128 parties), la distribution de longueur et d'issue des parties collectees
      contre les parties encore actives a la coupure. But : quantifier le biais vers les
      parties courtes, ou constater qu'il est negligeable. Aucun modele n'est necessaire
      pour la logique ; la mesure de distribution, elle, exige le vrai reseau.
- [ ] Consigner les chiffres dans `out/multicore/` pour la tache 6.
- [ ] **Aucun commit** sauf si des fixtures de test sont ajoutees, dans le commit de la
      tache 1 ou de la tache 4.

## Tache 3 : rectifier les diagnostics

**Fichiers :** modifier `python_src/dev_tools/selfplay_refill_bench.py`,
`docs/devlog.md`, et tout rapport qui reprend l'explication.

- [ ] Reecrire l'en-tete du banc : les deux regimes relancent les places, la difference
      comparee est le nombre de departs, l'horizon et le nombre de parties abandonnees.
      Nommer les configurations `(concurrent, total)` partout, y compris dans les lignes
      affichees, et non « pool vide » contre « pool renouvele ».
- [ ] Ajouter la reserve aux usages du chiffre : le gain du passage a 256 places n'est pas
      invalide, mais l'explication causale par un renouvellement nouvellement active est
      fausse. Ajouter une entree datée dans `docs/devlog.md`, sans reecrire l'entree du
      2026-09-20 15:25 qui conserve la valeur historique.
- [ ] Si `2026-09-19-cout-calcul-resultats.md` ou un rapport d'ingenierie reprend le
      chiffre, y ajouter un renvoi cible. Ne pas reecrire les mesures.
- [ ] **Commit :** `Rectifie l interpretation du banc de self-play`.

## Tache 4 : generation finie (R9-B)

**Fichiers :** modifier `src/selfplay_manager.hpp`, `src/selfplay_manager.cpp`,
`tests/cpp/test_selfplay_shared_core.cpp`.

- [ ] **Ecrire les tests en echec** : matrice `C = 4` avec `N = 0, 1, 3, 4, 7`, attendre
      `started == completed == N` et `active_slots == 0` a la fin, avec au plus
      `min(C, N)` places initialisees. Sans changement du scheduler, `N = 4` echoue deja
      (7 departs), `N = 1` demarre 4 parties inutiles.
- [ ] **Implementer** :
      separation de `games_started` et `games_completed` ;
      etat actif/inactif par place : preparer les conteneurs et le MCTS partage dans le
      constructeur, sans demarrer de partie avant que N soit connu ;
      apres chaque fin, enregistrer une fois, incrementer les fins, puis relancer la place
      seulement si `games_started < N`, sinon la desactiver ;
      arreter la boucle apres `completed == N` ;
      exclure les places inactives des descentes, de la condition « tout est bloque » et de
      la taille utile du lot ; accepter les lots partiels en fin de generation ;
      definir `N = 0` : aucun demarrage, aucune evaluation, resultat vide ;
      refuser `N < 0` et `C <= 0` avant allocation ;
      definir un second appel sur le meme gestionnaire comme une generation independante,
      compteurs remis a zero, aucune partie active heritee.
- [ ] **Verifier** les autres contrats du noyau partage : puzzles tactiques, amnesie,
      signe de l'issue finale, format des exemples, `inspect_tree` au repos apres
      interruption.
- [ ] **Commit :** `Termine les parties engagees du self-play`.

## Tache 5 : mesure avant/apres

**Fichiers :** modifier `python_src/dev_tools/selfplay_refill_bench.py` si necessaire ;
resultats dans `out/multicore/`.

- [ ] Comparer l'ancien et le nouveau scheduler, interleaved, meme session, ordre alterne,
      sur la matrice `(128, 128)`, `(256, 512)`, `(512, 512)` et `(128, 256)`, a 100/20
      simulations. Puis un passage a 700/100 sur la configuration de production
      `(256, 512)`, sur decision explicite seulement.
- [ ] Rapporter : temps mural, parties par seconde, nouveaux coups par seconde, positions
      conservees par seconde, profil de remplissage des lots par tranche de generation,
      duree de la queue de fin (temps pour collecter les dernieres parties), departs
      abandonnes, distribution de longueur et d'issue des parties conservees.
- [ ] Utiliser `new_plies` et non `GameResult.total_real_moves` comme denominateur du
      travail de jeu nouveau : l'historique rejoue d'un puzzle ne doit pas gonfler le
      chiffre.
- [ ] Critere : zero depart abandonne apres correction, `started == completed == N`, et
      pas de degradation du temps mural au-dela d'une marge de 10 pour cent a fixer avant
      la mesure. Si la queue de fin coute plus que le travail jeté, le documenter sans le
      masquer : le critere de ce lot est la production de N parties entieres et une
      comptabilite honnete, pas le gain de temps.
- [ ] **Aucun commit de code** ; les chiffres vont a la tache 6.

## Tache 6 : decision, rapport et cloture

**Fichiers :** creer
`docs/superpowers/specs/2026-09-21-fin-de-lot-self-play-resultats.md` ; modifier
`docs/backlog.md`, `CHANGELOG.md`, `docs/devlog.md`, `python_src/train_self_play.py` si la
decision le demande.

- [ ] Rediger le rapport : comportement avant/apres, matrice de quotas, caracterisation du
      biais, mesures de debit et de composition, limites, tests executes.
- [ ] Decider de la taille du pool dans un lot separe, avec les mesures comparables de la
      tache 5. Ne pas modifier les parametres de production dans le meme commit que le
      scheduler.
- [ ] Mettre a jour le backlog (cocher R9) et le devlog. Ajouter la ligne CHANGELOG.
- [ ] Suite CTest et pytest complete, plus une generation self-play minuscule avec faux
      evaluateur contenant une partie normale et une partie puzzle controlee.
- [ ] **Commit :** `Documente la fin de lot du self-play`.

## Risques et faux positifs

- **Queue de fin.** Terminer les parties restantes peut couter plus de temps que les
  jeter. Le nouveau contrat privilegie des parties entieres et des comptes justes ; si le
  temps mural se degrade, c'est une decision a documenter, pas un bug.
- **Changement de distribution des donnees.** Supprimer l'abandon des longues modifie la
  distribution des longueurs et des issues dans le replay buffer. C'est attendu, a
  mesurer et a mentionner, pas a corriger dans ce lot.
- **Denominateur trompeur.** `total_real_moves` inclut l'historique des puzzles. Toute
  mesure de debit qui l'ignore surestime le travail produit.
- **Anciens chiffres non comparables.** Les mesures du banc de remplissage d'avant ce lot
  ne sont plus une reference, ni pour le scheduler ni pour l'explication du gain de
  septembre.
- **Ne pas conclure sur la taille du pool.** Le lot corrige la fin de generation ; le choix
  256 contre 512 places se decide apres, sur des mesures propres.

## Ordre

```text
1 compteurs -> 2 caracterisation -> 3 diagnostics -> 4 generation finie
-> 5 mesure avant/apres -> 6 decision et rapport
```

Les taches 1 a 4 tournent sans modele et sans GPU, en tests C++ et en quelques minutes de
banc. La tache 5 represente une a deux heures selon la matrice retenue. Le chemin critique
est la tache 4 : sans la generation finie, la tache 5 n'a rien a comparer.
