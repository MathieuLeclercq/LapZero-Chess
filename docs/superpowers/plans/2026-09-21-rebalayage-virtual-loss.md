# Rebalayage du virtual loss apres R1 : plan d'implementation

**Statut (2026-09-22) : execute.** A/B interleaved, prefiltre 500, campagne 2500,
decision et propagation faits ; `virtual_loss = 2` est active. Resultats et
reserves : `../specs/2026-09-22-rebalayage-virtual-loss-resultats.md`. La
comparaison ancien contre nouveau scheduler de R9 reste hors perimetre, elle
demanderait de reconstruire l'ancien commit.

> **Pour les agents charges de l'execution :** executer les taches en ligne, une par
> une, avec le skill `superpowers:executing-plans` si disponible. Aucun sous-agent.
> Les cases a cocher servent au suivi. Aucune campagne qualite longue ni aucun commit
> automatique sans decision explicite du proprietaire.

**Objectif :** re-mesurer l'effet du virtual loss et de son couplage avec la FPU apres la
correction R1, remplacer un balayage non interpretable par une campagne propre, et decider
si les defauts de divergence restent 1, 0.30 et 4 ou changent.

**Architecture :** aucun changement de production dans les taches 1 a 4. Le correctif R1 du
plan `2026-09-21-audit-correctifs-mcts.md` est un prerequis dur. Les taches 5 et 6
n'activent un candidat que s'il franchit la mediane, le p95 et la barriere qualite, et
propagent alors le reglage au bot et au self-play dans le meme lot.

**Technologies :** C++17, CMake/CTest, Python 3.13, pytest, uv, ONNX Runtime 1.24.3 CUDA,
bancs `search_bench.py` et `puzzle_bench.py`.

**References :**
`docs/superpowers/plans/2026-09-21-audit-correctifs-mcts.md` (R1 section 3, R2 section 4,
contraintes du niveau D section 13),
`docs/superpowers/specs/2026-09-19-cout-calcul-cpu-gpu.md` (T3) et son rapport
`2026-09-19-cout-calcul-resultats.md` (section 4, balayage pollue),
`docs/superpowers/specs/2026-09-16-multicore-waves-results.md` (protocole de reference).

**Base :** HEAD apres le commit des documents (R1 pas encore implemente). Enregistrer le
commit de depart de la branche de travail au debut de l'execution.

**Statut du document :** plan prepare le 2026-09-21. Aucune mesure, compilation ni campagne
n'a ete executee pour le rediger.

## Contexte

Le balayage de divergence du 2026-09-19 a rejete `virtual_loss = 2` et
`virtual_loss = 2 + FPU 0.45` pour perte de qualite sur 500 puzzles, malgre un gain de
debit de 13 a 25 pour cent. Ce balayage a tourne avec le defaut R1 : `reserve(node, units)`
incrementait `n_in_flight` de `units`, mais `release()` retirait 1 par entree de chemin.
Des que l'amplitude depassait 1, chaque descente laissait un residu permanent de +1 sur
chaque noeud traverse. Les noeuds chauds accumulaient donc une penalite croissante, ce qui
ecarte mecaniquement la recherche des chemins deja explores.

Deux consequences se suivent. Le gain de debit mesure peut venir de cette fuite et non du
virtual loss lui-meme : apres R1, il peut disparaitre. Et la perte de qualite peut etre
causee par la penalite cumulative, qui n'existe plus apres R1. Le rejet de septembre n'est
donc ni confirme ni infirme, il est non interpretable dans les deux sens.

Le candidat `FPU 0.45` seul n'a pas ete passe a la barriere qualite en septembre, son gain
de debit etant plus faible. Il reste a mesurer proprement, seul puis couple.

Hypothese principale a tester : sans la fuite, `virtual_loss = 2` ne gagne plus de debit
notable et ne degrade plus la qualite. Hypothese alternative : il gagne encore, par
divergence reelle des descentes, et la qualite se maintient. La campagne doit distinguer
ces deux cas, pas chercher a confirmer un candidat.

## Contraintes globales

- Le correctif R1 doit etre merge et ses tests verts avant toute mesure. R2 est recommande
  avant la tache 1, car sans lui `en_vol` n'incremente pas `violations` et une fuite
  residuelle passerait inapercue.
- Ne pas modifier les defauts de `virtual_loss`, `fpu_reduction` ou
  `collision_attempt_factor` avant la tache 5.
- Ne pas melanger dans une meme comparaison la correction R1 et un autre changement de
  recherche. c_puct, la politique TT, la taille de batch et la taille de table restent
  celles du regime mesure.
- Valeurs absolues variables de 15 a 30 pour cent entre sessions : toute comparaison est
  intra-session, interleaved, avec ordre alterne. Ne jamais comparer les chiffres de cette
  campagne aux tableaux de septembre.
- Meme modele pour tous les bras : `2026_04_23_23h25_iter316_unsupervised.onnx`,
  sha256 `20a19f82e170`. Meme banc : `data/puzzles_bench.txt`, sha256 `a60b584abc25`.
- Un seul processus GPU a la fois. Les resultats bruts restent dans `out/multicore/`,
  ignore par git. Pas de binaire, de modele ni de CSV volumineux committe.
- Pas de tiret cadratin dans les textes ajoutes. Pas de ligne `Co-Authored-By`.
- Tests C++ en Release : controle par exception, jamais `assert()` supprime par `NDEBUG`.
- Toute campagne qualite longue, tout tournoi de niveau de jeu et tout commit sont
  executes sur decision explicite du proprietaire.

## Fichiers

| Fichiers | Responsabilite |
|---|---|
| `docs/superpowers/specs/2026-09-19-cout-calcul-resultats.md` | avertissement cible sur la section 4, sans reecrire les mesures |
| `out/multicore/rebalayage-*.json` | mesures de debit et compteurs, non committes |
| `docs/superpowers/specs/2026-09-21-rebalayage-virtual-loss-resultats.md` | rapport final et decision, cree en tache 6 |
| `python_src/search_bench.py`, `python_src/tests/test_search_bench.py` | banc de debit, deja pourvu des options de tuning |
| `python_src/puzzle_bench.py` | barriere qualite, deja pourvue des options de tuning |
| `python_src/uci.py`, `python_src/tests/test_uci_race.py` | activation conditionnelle du bot |
| `src/selfplay_manager.hpp`, `src/selfplay_manager.cpp`, `src/bindings.cpp`, `python_src/train_self_play.py` | propagation conditionnelle au self-play |
| `docs/backlog.md`, `CHANGELOG.md` | cloture |

## Commandes communes

Executer depuis la racine du depot. Adapter les chemins de CMake et de ctest a la machine
si necessaire.

```powershell
$cmakeExe = 'C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
$ctestExe = 'C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe'
$model = 'python_src/checkpoints_onnx/2026_04_23_23h25_iter316_unsupervised.onnx'
New-Item -ItemType Directory -Force out/multicore | Out-Null
& $cmakeExe -S . -B build
& $cmakeExe --build build --config Release
uv run pytest python_src/tests -q
& $ctestExe --test-dir build -C Release --output-on-failure
```

Debit, une configuration par invocation, cinq passages de six repetitions, ordre alterne
entre passages :

```powershell
uv run python python_src/search_bench.py --model $model --gpu --fixed-batch `
    --batch-sizes 8 --worker-counts 8 --cache-history-depths 0 --slices 64 `
    --simulations 700 --passages 5 --repetitions 6 --warmup-pool `
    --virtual-loss 2 --fpu 0.30 --collision-attempts 4 `
    --tt-size 8192 --timings --out-json out/multicore/rebalayage-vl2-s64.json
```

Qualite, 500 puis 2500 puzzles, meme disposition que les campagnes du 2026-09-15 :
historique reel, sans Dirichlet, lot 8, 700 simulations, 16 travailleurs, TT neuve par
puzzle, sous-echantillonnage deterministe.

```powershell
uv run python python_src/puzzle_bench.py --model $model --limite 500 `
    --simulations 700 --batch-size 8 --travailleurs 16 `
    --virtual-loss 2 --fpu 0.30 --collision-attempts 4 `
    --out-csv out/multicore/rebalayage-vl2-prefiltre500.csv `
    --out-rapport out/multicore/rebalayage-vl2-prefiltre500.txt
```

## Tache 0 : prerequis et avertissement historique

**Fichiers :** modifier `docs/superpowers/specs/2026-09-19-cout-calcul-resultats.md`.

- [ ] Verifier que le correctif R1 est merge et que R1-T1 a R1-T6 passent, en particulier
      R1-T5 sur les amplitudes 1, 2, 3 et 8 dans les vrais chemins de recherche.
- [ ] Verifier que R2 est merge ou, a defaut, controler `en_vol` a la main apres chaque
      recherche de la tache 1, l'inspection ne le comptant pas encore comme violation.
- [ ] Ajouter un avertissement cible a la section 4 du rapport du cout de calcul : les deux
      candidats a amplitude 2 ont ete mesures avec une reservation qui laissait +1 permanent
      par descente et par noeud ; le rejet n'est ni confirme ni infirme et le rebalayage
      `2026-09-21-rebalayage-virtual-loss.md` le remplace. Ne pas reecrire les chiffres.
- [ ] Ajouter le meme renvoi dans le plan d'audit, section R1, acceptation de l'historique
      des mesures si ce n'est pas deja fait.
- [ ] **Commit :** `Avertit que le balayage de divergence de septembre est non interpretable`.

## Tache 1 : neutralite du correctif R1 a amplitude 1

**Fichiers :** aucun fichier de production. Mesures seulement.

Le correctif R1 doit etre neutre au defaut : a `virtual_loss = 1`, `reserve` et `release`
s'equilibraient deja avant. Cette tache prouve que le correctif ne change rien au
comportement courant, avant de mesurer les amplitudes superieures.

- [ ] Recherche de reference sur le code d'avant R1 (commit de depart) et sur le code
      corrige, meme session, interleaved, a amplitude 1, tranches 64, workers 8, lot fixe.
      Attendre : memes compteurs, memes nombres de simulations, memes debits a la
      dispersion pres. Sur le chemin sequentiel a 1 worker, visites et priors identiques au
      bit pres.
- [ ] Apres chaque recherche corrigee, controler `inspect_tree` : `en_vol == 0`,
      `pending == 0`, simulations completees egales au budget, a amplitude 1, 2 et 3.
      Ce controle remplace le test unitaire de fuite au niveau du banc.
- [ ] Si un ecart apparait a amplitude 1, arreter le plan et ouvrir un correctif R1 : la
      neutralite est un prerequis, pas un resultat a commenter.

## Tache 2 : debit, configurations du balayage

**Fichiers :** aucun fichier de production. JSON dans `out/multicore/`.

Configurations, toutes avec `collision_attempt_factor = 4` :

| Nom | `virtual_loss` | `fpu_reduction` | Motif |
|---|---|---|---|
| A | 1 | 0.30 | defaut, reference des comparaisons appariees |
| B | 2 | 0.30 | l'ancien candidat, a re-mesurer seul |
| C | 3 | 0.30 | verifier la monotonie de la fuite supposee |
| D | 2 | 0.45 | ancien couple, FPU jamais mesuree seule en qualite |
| E | 1 | 0.45 | FPU seule, si D montre un effet propre |

- [ ] Mesurer A a D (puis E si necessaire) en tranches de 64, 700 simulations, lot 8,
      workers 8, lot fixe, h0, pool chaud, 5 passages de 6 repetitions, ordre alterne.
- [ ] Mesurer la gagnante eventuelle en tranches de 20 en secondaire, puis a 700
      simulations en un appel unique, pour couvrir le regime de temps court du bot.
- [ ] Rapporter par configuration : mediane et dispersion des simulations par seconde par
      position, debit renormalise par le remplissage, collisions, remplissage, temps
      evaluateur, `en_vol` et `pending` residuels.
- [ ] Critere de passage en tache 3 : mediane en hausse d'au moins 5 pour cent sur au moins
      deux positions sur trois a tranches 64, p95 de latence sous plus 5 pour cent, aucune
      fuite residuelle, aucun changement de compteurs de simulation. Une hausse dans la
      marge de bruit de session ne qualifie pas un candidat.
- [ ] Si aucun candidat ne franchit le critere, passer directement a la tache 6 : le rejet
      de septembre est confirme proprement et les defauts restent.
- [ ] Aucun commit intermediaire : les JSON restent dans `out/multicore/` et les chiffres
      vont au rapport.

## Tache 3 : prefiltre qualite 500 puzzles

**Fichiers :** aucun fichier de production. CSV et rapports dans `out/multicore/`.

- [ ] Passer chaque candidat retenu en tache 2 et la reference A sur les memes 500 puzzles,
      meme ordre, parametres de recherche identiques par ailleurs, vecteurs de reussite par
      puzzle conserves.
- [ ] Comparaison appariee contre A : table de discordance, McNemar exact, delta de
      reussite avec intervalle de confiance. La colonne reseau seul doit etre identique au
      bit pres entre les bras, sinon la campagne est invalide.
- [ ] Marge de non-inferiorite : moins 1 point, comme la campagne multicoeur. Un candidat
      passe si la borne basse de l'intervalle depasse moins 1 point. Si la borne est proche
      du seuil, repeter le prefiltre avant de conclure.
- [ ] Ne pas lire ces 500 puzzles comme une mesure de force : c'est un filtre, pas une
      estimation d'Elo. Un ecart plus petit que la resolution du banc ne tranche rien.
- [ ] **Aucun commit.**

## Tache 4 : campagne complete 2500 puzzles

**Fichiers :** aucun fichier de production. CSV et rapports dans `out/multicore/`.

- [ ] Pour chaque candidat ayant franchi le prefiltre, campagne complete sur les 2500
      puzzles, meme protocole apparie contre A, une session, un candidat a la fois.
- [ ] Mesure a temps egal en option pour le meilleur candidat : budget egal a la duree
      mediane de A sur 500 puzzles, et rapporter separement le verdict a simulations egales
      et le verdict a temps egal, comme au multicore.
- [ ] Verdict explicite par candidat : non-inferiorite, superiorite ou rejet, avec delta et
      intervalle, nombre de paires gagnees et perdues, et la dispersion de la session.
- [ ] Si aucun candidat ne passe, conclure au maintien des defauts.

## Tache 5 : decision et activation conditionnelle

**Fichiers :** modifier `python_src/uci.py`, `python_src/tests/test_uci_race.py`,
`src/selfplay_manager.hpp`, `src/selfplay_manager.cpp`, `src/bindings.cpp`,
`python_src/train_self_play.py`, et les scripts de jeu si la coherence l'exige.

- [ ] Ecrire d'abord le test qui fixe le nouveau reglage : la valeur de tuning du moteur
      UCI est celle decidee, et elle est bien transmise a `set_tuning` avant la recherche.
- [ ] Si un candidat gagne : declarer la constante de divergence a cote de
      `MCTS_BATCH_SIZE` et `MCTS_WORKER_COUNT` dans `uci.py`, avec les chiffres mesures en
      commentaire. Conserver `c_puct` inchange et ne pas modifier la taille de table.
- [ ] Propager au self-play dans le meme lot. `SelfPlayManager` doit exposer le tuning et
      `generate_self_play_games` le recevoir, sinon le bot et les donnees d'entrainement
      utiliseraient deux recherches differentes. Ajouter un test de propagation.
- [ ] Faire de meme dans `play_against_bot.py` et `tournament_elo.py` si leur recherche
      doit rester identique au bot.
- [ ] Si aucun candidat ne gagne : ne rien changer en production, documenter le rejet
      propre, et retirer la mention « non interpretable » au profit de « rejet confirme ».
- [ ] **Commit :** `Active un virtual loss mesure` si un reglage change, sinon
      `Documente le rejet du rebalayage de virtual loss`.
- [ ] Tournoi de niveau de jeu (WHR ou Stockfish ancre) : uniquement sur decision explicite
      du proprietaire, dans un lot separe. Aucune campagne longue automatique. Le banc de
      puzzles ne mesure pas un Elo.

## Tache 6 : rapport, backlog et cloture

**Fichiers :** creer
`docs/superpowers/specs/2026-09-21-rebalayage-virtual-loss-resultats.md` ; modifier
`docs/backlog.md`, `CHANGELOG.md`, `docs/superpowers/specs/2026-09-19-cout-calcul-resultats.md`.

- [ ] Rediger le rapport : modele et hash, banc et hash, commits, protocole exact, tableaux
      de debit avec dispersion, tableaux de puzzles apparies, decision et limites. Y noter
      que les chiffres de septembre ne sont pas comparables et pourquoi.
- [ ] Y consigner les tests executes, CTest et pytest, et le smoke UCI si la production a
      change.
- [ ] Mettre a jour le backlog : cocher le rebalayage, laisser ouvertes les autres taches
      de l'audit, et consigner la suite (R9-A et R9-B en priorite).
- [ ] Ajouter la ligne `CHANGELOG.md` si un defaut de production change.
- [ ] **Commit :** `Documente le rebalayage du virtual loss`.

## Risques et faux positifs

- **Le gain de debit peut disparaitre.** Si le +13 pour cent de septembre venait de la
  penalite cumulative, il ne se reproduira pas sur le code corrige. C'est un resultat
  attendu, pas une regression : la recherche revient a sa definition correcte.
- **500 puzzles ne tranchent pas un point.** La resolution du prefiltre est de l'ordre de
  plusieurs points ; toute decision se prend sur la campagne 2500, le prefiltre ne sert
  qu'a ecarter l'evident.
- **Le bruit inter-session interdit les comparaisons historiques.** Comparer au balayage de
  septembre, meme a configuration egale, n'a pas de sens. Toute la lecture est
  intra-session.
- **Les grandes amplitudes n'ont jamais ete mesurees proprement.** 4 et au-dela ne se
  testent que si la tendance entre 1, 2 et 3 est monotone et documentee, et pas dans le
  meme lot.
- **Le self-play est le premier consommateur.** Une adoption non propagee au self-play
  ferait diverger le bot et les donnees d'entrainement. C'est une condition d'acceptation,
  pas une finition.

## Ordre

```text
0 prerequis et avertissement -> 1 neutralite R1 -> 2 debit -> 3 prefiltre 500
-> 4 campagne 2500 -> 5 decision et activation -> 6 rapport
```

Ordre de grandeur : la tache 1 et la tache 2 representent moins d'une heure de mesure GPU,
les prefiltres une demi-heure, les campagnes completes de 15 a 45 minutes chacune selon
l'occupation de la machine. Le chemin critique reste R1 : sans lui, rien de ce plan n'est
interpretable.
