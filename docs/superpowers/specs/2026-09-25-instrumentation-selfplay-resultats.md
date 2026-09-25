# Instrumentation du self-play : resultats

Date : 25 septembre 2026. Chantier A0 de
`superpowers/specs/2026-09-25-selfplay-throughput-audit.md`.

Objectif : rendre le vrai chemin de production observable avant de choisir une
optimisation. Le livrable est l'instrumentation elle-meme, le banc qui
l'exploite, et les premiers profils obtenus sur la machine de developpement.
**Ces profils ne remplacent pas une mesure sur le PC fixe** et ne doivent pas
etre compares a ses 9,84 positions/s : configuration, pool et machine
differents.

## 1. Ce qui est en place

- `src/selfplay_timing.hpp` : phases de premier niveau disjointes (departs
  initiaux, gestion des coups, collecte, assemblage et decision d'envoi, appel
  evaluateur, validation, consommation, finalisation), residu explicite signe,
  sous-durees ONNX cumulees et compteurs de debit.
- `SelfPlayManager::set_diagnostics_mode` : 0 desactive (aucune horloge lue),
  1 phases, 2 phases plus travail mural par worker de la collecte. Le mode 2
  n'ajoute pas de chronometre par noeud.
- `Evaluator::diagnostic_totals` / `set_timing_enabled` : totaux cumules
  d'appels et de lignes, horloges internes de `session->Run` et du softmax.
  L'implementation ONNX compte les appels en permanence ; les horloges ne
  tournent qu'a la demande.
- Binding `generate_self_play_games_with_diagnostics` (parties, compteurs,
  phases) ; les deux fonctions de production restent inchangees.
- `python_src/selfplay_diag.py` : rapport pur, testable sans modele, avec les
  trois debits distincts (coups nouveaux/s, exemples/s, evaluations reseau/s),
  les pourcentages avec leur denominateur et l'histogramme des lots.
- `python_src/dev_tools/selfplay_diagnostics.py` : pilote sur le vrai chemin,
  horloges Python separees (creation, appel C++, conversion, nettoyage),
  empreintes du binaire, du modele et revision git, ecriture d'un JSON dans
  `out/selfplay_diagnostics/`.
- `run_selftrain.py --selfplay-diagnostics N` : active le rapport en
  production et journalise les pourcentages de phase dans wandb. Defaut 0,
  generation inchangee.

## 2. Protocole sur le PC fixe

Depuis la racine, avec l'environnement du fixe et une machine au repos :

```
python python_src/dev_tools/selfplay_diagnostics.py \
    --model <checkpoint ONNX> \
    --games 512 --concurrent 256 \
    --slow-sims 700 --fast-sims 100 \
    --passes 1 --mode 1
```

Verifier avant de conclure : departs = fins = quota, places actives finales
nulles, residu faible devant la generation, sous-durees ONNX non additionnees
aux phases. Pour choisir une optimisation, lire le rapport dans cet ordre :
attente et formes d'abord (A1), post-traitement ensuite (A2), softmax (A3),
puis seulement le calcul et le transfert (taille des lots, A4, A12).

Le cout du profilage doit etre connu : comparer des passages alternes
mode 0 / mode 1, repetes, car la variance entre passages est forte (voir §4).

## 3. Validation de l'instrumentation

- Inertie : meme graine et meme bruit, diagnostic actif ou non, parties
  identiques au bit pres (`test_diagnostics_are_inert_and_match_the_evaluator`).
- Coherence : appels et lignes reseau egaux a ce que l'evaluateur a recu,
  histogramme somme aux appels physiques, requetes de feuilles egales aux
  lignes lancees, simulations terminees = lignes consommees + simulations sans
  reseau, exemples sauves egaux a la somme des coups enregistres, residu qui
  ferme le budget de phases, mode 2 sain (`worker_busy_sum >= worker_busy_max`).
- Compteurs remis a zero entre generations, retour au silence apres passage en
  mode 0.
- Course corrigee : les compteurs de la collecte OpenMP sont desormais ecrits
  par worker puis agreges apres la barriere. Avant le correctif, les lignes
  lancees depassaient les requetes comptees (increments perdus).
- `pytest` : 8 tests du rapport pur, 2 tests de la conversion FP16 par partie.
- `softmax_rows` : 4 tests prouvant l'egalite bit a bit avec le chemin
  sequentiel, y compris sur logits extremes et formes particulieres.

## 4. Premiers profils (machine de developpement, 32 places)

Configuration : iter436, 32 parties, 32 places, 700/100, puzzles actifs, mode 1.
La queue de generation (peu de places actives) y est surrepresentee par
rapport a la production, donc les petits lots y sont surrepresentes.

| Passage | Generation | Evaluateur | Softmax | Collecte | Consommation | Lots | Lignes | Moyenne |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 251,9 s | 90,8 % | 12,2 s | 2,5 % | 3,8 % | 65 020 | 1 009 697 | 15,5 |
| 2 | 236,4 s | 90,2 % | 12,1 s | 2,8 % | 3,8 % | 61 471 | 927 366 | 15,1 |

Un troisieme profil avec un pool de 128 places et 128 parties (mode 1,
iter436, 700/100) donne une image plus proche de la production :

| Mesure | Valeur | Part |
|---|---:|---:|
| Generation | 891,7 s | 100 % |
| Appel a l'evaluateur | 662,1 s | 74,3 % |
| dont ONNX Run | 497,9 s | 55,8 % |
| dont softmax | 162,8 s | 18,2 % |
| Collecte (descentes et tenseurs) | 117,9 s | 13,2 % |
| Consommation des feuilles | 54,7 s | 6,1 % |
| Assemblage et envoi | 28,3 s | 3,2 % |
| Gestion des coups | 17,9 s | 2,0 % |
| Validation et finalisation | 10,0 s | 1,1 % |
| Lots | 81 833 appels, 4 321 164 lignes, moyenne 52,8, max 128 | |
| Changements de taille | 796 | |
| Attente | 180 096 tours sur 265 204 (68 %), 125,0 s cumulees (14 %), age max 146,1 ms | |
| Simulations sans reseau | 247 636 sur 4 568 800 (5,4 %) | |

Enseignements : la taille moyenne des lots grandit avec le pool (52,8 a 128
places contre 15,1 a 32), l'evaluateur reste dominant mais les postes CPU
deviennent visibles (collecte 13,2 %, consommation 6,1 %, assemblage 3,2 %), et
le report d'envoi n'est plus negligeable (14 % du temps mural, age maximal
146 ms). C'est la configuration ou A1 et A2 meritent d'etre etudies, avec la
contrainte de formes stables mesuree au §5.

- L'evaluateur domine, comme attendu. Le softmax represente environ 5 % de la
  generation aux lots reellement utilises.
- Le report d'envoi existe (38 a 48 % des tours avec un lot pret non envoye)
  mais coute moins de 2 % du temps mural ; l'age maximal d'un lot en attente
  reste inferieur a 50 ms.
- Les simulations sans reseau (hits de table surtout) representent 4 a 7 % des
  simulations terminees. Aucune simulation terminale sur ces passages.
- La variance entre passages de meme configuration est forte : 236 s a 414 s
  selon le tirage des parties et l'etat de la machine. Toute comparaison A/B
  doit etre alternee et repetee ; un passage unique ne prouve rien.

Le cout du profilage a ete cherche par une campagne alternee A/B/B/A sur la
meme configuration : mode 1, mode 0, mode 0, mode 1.

| Passage | Mode | Appel C++ | Generation | Coups nouveaux/s |
|---|---:|---:|---:|---:|
| 1 | 1 | 444,4 s | 442,1 s | 8,80 |
| 2 | 0 | 530,1 s | 0 (silencieux) | 8,50 |
| 3 | 0 | 442,3 s | 0 (silencieux) | 8,44 |
| 4 | 1 | 415,0 s | 412,8 s | 9,20 |

Deux passages identiques varient de 29 a 88 s (jusqu'a 20 %), et l'ordre
observe est inverse de celui attendu si le profilage coutait : la derive de la
machine domine, aucune conclusion de surcout n'est possible a cette echelle.
D'autres processus utilisateur tournaient pendant la campagne (enregistreur
vocal, application de traitement B-scan), ce que l'audit demande de
documenter. Le cout reste borne par construction : une horloge par phase et
par tour, aucun chronometre par noeud, deux instantanes de totaux par appel de
lot, et le residu mesure reste sous 0,1 % de la generation.

## 5. Piste A1 rejetee en l'etat : lancer a chaque tour

Hypothese testee : ne plus attendre qu'une partie qui enchaine les
simulations sans reseau fournisse une requete, et envoyer le lot des qu'une
passe de collecte est terminee.

Mesure : 615,1 s de generation contre 236,4 s pour la meme configuration, avec
79 905 appels (moyenne 13,5) et surtout **30 474 changements de forme** contre
238. Le banc hors arbre explique le mecanisme : une ligne dont la forme vient
de changer coute environ 16 ms, contre 3,6 ms pour un huit lignes a forme
fixe et environ 13 ms pour un lot fixe de 256. ONNX Runtime repaye son plan a
chaque changement de forme, meme deja vu.

Consequence : le report d'envoi, qui laisse les feuilles s'accumuler et garde
des formes stables, est conserve tel quel. Un padding vers des tailles fixes
n'est pas retenu a ce stade : avec 238 changements mesures, le plan coute
environ 2,9 s, alors que padder les dizaines de milliers de petits lots
ajouterait davantage de lignes GPU que cela n'en economiserait. Toute variante
devra d'abord demontrer qu'elle reduit les changements de forme sans ajouter
plus de travail GPU.

## 6. A3 : softmax par lignes, code mais desactive en production

`src/softmax.hpp` repartit les lignes entre threads au-dela de 16 lignes, sans
jamais parallelliser la reduction d'une ligne : le resultat est bit a bit
identique au chemin sequentiel, avec ou sans parallelisation (tests dedies).

Au banc isole, machine au repos et lot fixe, le gain est net :

| Lot | Softmax sequentiel | Softmax parallele |
|---|---:|---:|
| 8 | 0,09 a 0,12 ms | 0,09 a 0,12 ms (seuil) |
| 64 | 0,66 a 0,81 ms | 0,11 a 0,13 ms |
| 256 | 2,46 a 2,79 ms | 0,40 a 0,44 ms |

Mais le profil a 128 places contredit la transposition : 162,8 s de softmax en
mode parallele pour 4,32 M de lignes, alors que le sequentiel en coute environ
50 s au rythme mesure. Une region OpenMP par appel, sur une machine chargee et
avec des tailles qui varient, coute plus cher que les exponentielles
economisees. Le mode parallele est donc **desactive par defaut dans
l'evaluateur** (`allow_parallel = false`), tout en restant teste et disponible.
Le second palier de l'audit, un softmax calcule sur les seuls coups legaux
(environ trente fois moins de travail), est la piste qui reste.

Lecon generale : un gain de banc isole, lot fixe et machine au repos, ne se
transpose pas forcement a une boucle self-play chargee ; la mesure de contexte
prime.

## 7. A9 applique : copies en moins

- Cote C++ : les `GameResult` sont deplaces dans le vecteur de sortie, les
  donnees par coup de la place sont liberees des le transfert, et le retour du
  vecteur est un deplacement.
- Cote Python : conversion FP16 par partie au lieu d'une conversion par
  position, precision et signe de value inchanges (tests dedies).

## 8. Suite

1. Executer le protocole §2 sur le PC fixe, avec sa vraie configuration.
2. Choisir le premier levier d'apres ce rapport : A1 avec une variante qui
   stabilise les formes (le report coute 14 % a 128 places, mais lancer a
   chaque tour coute 2,6 fois le total), A2 si le post-traitement domine, A4 si
   les doublons exacts sont frequents, puis le softmax legal.
3. Repeter les passages pour la variabilite avant toute conclusion de debit.
