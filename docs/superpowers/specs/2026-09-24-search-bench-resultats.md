# Banc de recherche : resultats

Modele : `2026_04_23_23h25_iter316_unsupervised.onnx`, iteration None, global_step None
Protocole : 30 passages, 700 simulations, c_puct 1.4, GPU, un seul processus, batches demandes [8], workers [8], pool reutilise entre mesures, politiques TT [0]

Trois grandeurs distinctes. Les **simulations par seconde** mesurent le
debit de la recherche. Les **positions reseau par seconde** comptent les
positions effectivement evaluees, les **appels batch par seconde** les
lancements physiques et leur ratio est le **remplissage**. Le **taux de
table** est la part des consultations reussies.

## Debit

| Position | Chemin | TT | batch | workers | pool | passages | sims/s (med) | sims/s (min a max) | positions reseau/s | appels batch/s | remplissage | taux table | match position | 50 coups | contexte | historique |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| finale | mcts_search | h0 | 8 | 8 | chaud | 30 | 1297.9 | 1064.7 a 2080.0 | 1299.8 | 193.8 | 6.7 | 9.4 % | 99.5 (12.8 %) | 24 (3.1 %) | 2 (0.3 %) | 0 (0.0 %) |
| finale | step_analysis | h0 | 8 | 8 | chaud | 30 | 1358.9 | 1010.0 a 2010.6 | 1360.9 | 203.8 | 6.6 | 9.2 % | 98 (12.7 %) | 24 (3.1 %) | 2 (0.3 %) | 0 (0.0 %) |
| milieu | mcts_search | h0 | 8 | 8 | chaud | 30 | 1069.9 | 821.9 a 1554.3 | 1071.4 | 211.6 | 5.1 | 4.1 % | 32.5 (4.4 %) | 3 (0.4 %) | 0 (0.0 %) | 0 (0.0 %) |
| milieu | step_analysis | h0 | 8 | 8 | chaud | 30 | 1052.9 | 815.7 a 1666.5 | 1054.5 | 209.7 | 5.1 | 4.2 % | 34.5 (4.7 %) | 3 (0.4 %) | 0 (0.0 %) | 0 (0.0 %) |
| ouverture | mcts_search | h0 | 8 | 8 | chaud | 29 | 1433.5 | 1132.2 a 2185.7 | 1435.5 | 202.9 | 7.2 | 2.9 % | 33 (4.6 %) | 12 (1.7 %) | 0 (0.0 %) | 0 (0.0 %) |
| ouverture | mcts_search | h0 | 8 | 8 | froid | 1 | 1734.6 | 1734.6 a 1734.6 | 1737.1 | 235.4 | 7.4 | 3.2 % | 35 (4.8 %) | 12 (1.7 %) | 0 (0.0 %) | 0 (0.0 %) |
| ouverture | step_analysis | h0 | 8 | 8 | chaud | 29 | 1511.0 | 1229.2 a 2139.1 | 1513.2 | 207.2 | 7.2 | 2.8 % | 33 (4.6 %) | 12 (1.7 %) | 0 (0.0 %) | 0 (0.0 %) |
| ouverture | step_analysis | h0 | 8 | 8 | froid | 1 | 1226.8 | 1226.8 a 1226.8 | 1228.6 | 170.0 | 7.2 | 2.8 % | 32 (4.4 %) | 12 (1.7 %) | 0 (0.0 %) | 0 (0.0 %) |

## Invariants d'arbre

Non verifies lors de ce passage.
