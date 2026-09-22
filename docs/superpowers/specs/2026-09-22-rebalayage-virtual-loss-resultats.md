# Rebalayage du virtual loss apres R1 : resultats et decision

Date : 2026-09-22

Statut : campagnes terminees, `virtual_loss = 2` active dans le bot, le tournoi,
la GUI et l'ancrage Stockfish.

Plan : `2026-09-21-rebalayage-virtual-loss.md`. Modele : `2026_04_23_23h25_iter316_unsupervised.onnx`
pour le banc de recherche, `2026_04_30_09h53_iter436_unsupervised.onnx` pour le
banc de puzzles du confondant. Materiel : RTX PRO 2000 Blackwell Laptop 8 Gio,
16 coeurs logiques.

## 1. Resume

Le rejet de `virtual_loss = 2` du 2026-09-19 etait pollue par la fuite de
reservation corrigee par R1 : au-dela de l'amplitude 1, chaque descente laissait
+1 permanent sur les noeuds chauds. Le rebalayage propre renverse le verdict :
sur quatre invocations alternees A, B, B, A, l'amplitude 2 gagne 9 a 20 % de
debit sur les trois positions, abaisse le p95 de latence de 13 a 20 %, et passe
la barriere qualite sur 2500 puzzles avec un intervalle de confiance
[-0,4 ; +0,56] point. Le defaut de production est donc passe a 2 dans tous les
lanceurs du bot, du tournoi, de la GUI et de l'ancrage Stockfish. La campagne de
2500 a tourne avec un seul worker de recherche par processus, donc sur le chemin
mono batche ; une passe reduite sur les vagues (500 puzzles, 8 workers, meme
echantillon que le prefiltre) donne 367 contre 369 reussites, non-inferiorite,
ce qui leve l'essentiel du doute sur le chemin deploye.

## 2. Debit, A/B interleaved

Quatre invocations du banc de recherche en ordre A, B, B, A, 700 simulations,
lot fixe 8, 8 workers, tranches 64, pool chaud, 5 passages x 6 repetitions.
Medianes en simulations par seconde, dispersion entre crochets :

| chemin | position | A, vloss 1 | B, vloss 2 | gain |
|---|---|---|---|---|
| step_analysis | ouverture | 1919 [1625-2246] | 2087 [1741-2371] | x1.09 |
| step_analysis | milieu | 1436 [1160-1675] | 1684 [1460-1851] | x1.17 |
| step_analysis | finale | 1827 [1449-2173] | 2006 [1689-2194] | x1.10 |
| mcts_search | ouverture | 1272 [1064-1502] | 1482 [1281-1631] | x1.17 |
| mcts_search | milieu | 974 [836-1146] | 1169 [1032-1288] | x1.20 |
| mcts_search | finale | 1126 [933-1324] | 1325 [1161-1502] | x1.18 |

Le p95 de latence du candidat vaut 0,80 a 0,87 fois celui des defauts, donc il
s'ameliore. Les collisions en milieu passent de 3284 a 2054, et le remplissage
de 5,2 a 6,0. Les trois positions depassent le critere de +5 %, la queue est
meilleure et aucune unite ne reste en vol (verifie par les tests R1 et par un
passage reel a amplitude 3 avec 8 workers).

## 3. Barriere qualite

Prefiltre 500 puzzles, puis campagne 2500, appariess sur les memes lignes,
meme modele et meme budget, seule l'amplitude change. Verdict par
`multicore_comparison.py`, marge de non-inferiorite de moins 1 point.

| campagne | reference vloss 1 | candidat vloss 2 | delta | IC95 | McNemar p | verdict |
|---|---|---|---|---|---|---|
| prefiltre 500 | 366 | 368 | +0,4 pt | [-0,8 ; +1,6] | 0,75 | non-inferiorite |
| campagne 2500 | 1926 | 1928 | +0,08 pt | [-0,4 ; +0,56] | 0,87 | non-inferiorite |

Ces deux lignes ont tourne avec `search_workers = 1` : les seize
« travailleurs » sont des processus Python independants, donc la non-inferiorite
porte sur le chemin mono batche. Le bot deploye collecte ses feuilles par vagues
de huit workers C++, ou les reservations sont visibles pendant les descentes
concurrentes et ou les collisions different. Le meme sous-echantillon de 500
lignes a donc ete rejoue sur ce chemin, deux processus de huit workers :

| campagne | reference vloss 1 | candidat vloss 2 | delta | IC95 | McNemar p | verdict |
|---|---|---|---|---|---|---|
| vagues 500, 8 workers | 367 | 369 | +0,4 pt | [-0,4 ; +1,2] | 0,62 | non-inferiorite |

Le prefiltre mono et cette passe utilisent les memes 500 lignes, verifie sur la
colonne `ligne`, donc les chiffres sont comparables ligne a ligne. La passe de
2500 en vagues n'a pas ete refaite : elle reste la reserve explicite de la
section 5.

La colonne reseau seul est identique au bit pres entre les deux bras, sur les
500 puis sur les 2500 lignes : la comparaison ne porte que sur la recherche.

## 4. Decision et propagation

`virtual_loss = 2` est active dans `uci.py` (`MCTS_VIRTUAL_LOSS = 2`, applique
au MCTS de production), `play_against_bot.py`, `tournament_elo.py` et
`stockfish_player.py`, pour que le bot, la GUI, le tournoi et l'ancrage
cherchent de la meme facon.

Le self-play n'est pas concerne, et c'est un resultat du code, pas un oubli :
chaque arbre y est descendu une seule fois par vague, donc `n_in_flight` n'a
aucun lecteur entre la reservation et sa liberation, et l'amplitude y est
inerte. Avant R1, la fuite faisait s'accumuler ces unites d'une vague a l'autre,
ce qui est une des raisons pour lesquelles le balayage de septembre etait non
interpretable. Le gestionnaire garde donc le defaut 1, avec un commentaire qui
documente l'inerte.

Les deux affirmations sont verrouillees par des tests C++. Un test avec
l'evaluateur discriminant montre que le chemin mono batche, celui du banc de
puzzles, change bien de distribution de visites entre amplitude 1 et 2, donc la
barriere qualite n'est pas vide. Un autre test genere deux fois les memes
parties avec la meme graine et verifie qu'elles sont identiques au bit pres a
amplitude 1 et 2, ce qui etablit l'inerte du self-play.

Le defaut de la classe `MCTS` et des bancs reste 1 : les campagnes historiques
gardent leur sens, et un banc qui n'explicite pas le reglage mesure toujours le
defaut d'origine. Un test Python verrouille l'accord des trois lanceurs, un test
C++ verrouille la propagation au gestionnaire.

## 5. Reserves

- La campagne qualite de 2500 a tourne en mono batche. Le chemin des vagues est
  couvert par une passe reduite de 500 puzzles (non-inferiorite, IC95
  [-0,4 ; +1,2]) ; refaire les 2500 en vagues reste la seule facon d'atteindre
  la precision de la campagne.
- La comparaison ancien contre nouveau scheduler de R9 n'a pas ete faite : elle
  demanderait de reconstruire l'ancien commit. La matrice de pools a la place
  mesure (128, 256), (256, 512) et (512, 512) places a 100/20 simulations :
  192,9, 231,2 et 254,4 plies nouveaux par seconde, avec departs = fins = total
  partout. Une confirmation a 700/100 est necessaire avant de changer la taille
  de pool de production, qui reste 256.
- Les chiffres sont mesures sur iter316. Le serveur de self-play n'a pas besoin
  du reglage : l'amplitude y est inerte (section 4), seuls le bot et ses chemins
  batches en beneficient.
- Aucun tournoi de niveau de jeu n'a ete lance : le banc de puzzles mesure une
  non-inferiorite, pas un gain Elo. Un tournoi reste la seule mesure de force.
- Les mesures de septembre 2026 sur la divergence restent non interpretables et
  ne sont pas reecrites ; ce document les remplace.

## 6. Tests

- 18/18 CTest, dont le perft roundtrip et le palier calibre.
- 285 pytest, dont la propagation du tuning au gestionnaire, l'accord des trois
  lanceurs et le test de regression du binding self-play avec compteurs.
- Depuis la revue : restauration du plateau apres une erreur d'evaluation en
  cours de descente, invariant de la racine reutilisee pour le bruit differe,
  test du puzzle lent rendu non vacuous, et test d'inerte du self-play passe a
  l'evaluateur discriminant.
- A/B interleaved pour toutes les comparaisons de debit, meme session, ordre
  alterne.
