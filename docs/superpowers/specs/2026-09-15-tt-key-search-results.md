# Banc de recherche : resultats

Modele : `2026_04_23_23h25_iter316_unsupervised.onnx`, iteration 316, global_step 19415
Protocole : 7 passages, 700 simulations, c_puct 1.4, GPU, un seul processus, batches demandes [8], politiques TT [-1, 0, 1, 3, 7]

Trois grandeurs distinctes. Les **simulations par seconde** mesurent le
debit de la recherche. Les **positions reseau par seconde** comptent les
positions effectivement evaluees, les **appels batch par seconde** les
lancements physiques et leur ratio est le **remplissage**. Le **taux de
table** est la part des consultations reussies.

## Debit

| Position | Chemin | TT | batch | passages | sims/s (med) | sims/s (min a max) | positions reseau/s | appels batch/s | remplissage | taux table | match position | 50 coups | contexte | historique |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| finale | mcts_search | legacy | 8 | 7 | 799.7 | 750.1 a 812.7 | 800.9 | 129.1 | 6.2 | 14.2 % | 116 (14.2 %) | 0 (0.0 %) | 0 (0.0 %) | 0 (0.0 %) |
| finale | mcts_search | h0 | 8 | 7 | 715.3 | 693.4 a 765.7 | 716.3 | 117.5 | 6.1 | 9.2 % | 100 (13.0 %) | 27 (3.5 %) | 2 (0.3 %) | 0 (0.0 %) |
| finale | mcts_search | h1 | 8 | 7 | 706.6 | 665.1 a 728.4 | 707.6 | 120.1 | 5.9 | 5.7 % | 93 (12.5 %) | 25 (3.4 %) | 1 (0.1 %) | 25 (3.4 %) |
| finale | mcts_search | h3 | 8 | 7 | 707.3 | 694.5 a 759.0 | 708.3 | 118.2 | 6.0 | 1.4 % | 84 (11.8 %) | 25 (3.5 %) | 2 (0.3 %) | 47 (6.6 %) |
| finale | mcts_search | h7 | 8 | 7 | 703.4 | 677.6 a 744.7 | 704.4 | 117.6 | 6.0 | 0.0 % | 84 (12.0 %) | 26 (3.7 %) | 1 (0.1 %) | 57 (8.1 %) |
| finale | step_analysis | legacy | 8 | 7 | 783.7 | 748.0 a 796.8 | 784.8 | 126.5 | 6.2 | 14.2 % | 116 (14.2 %) | 0 (0.0 %) | 0 (0.0 %) | 0 (0.0 %) |
| finale | step_analysis | h0 | 8 | 7 | 692.2 | 680.2 a 710.1 | 693.2 | 113.7 | 6.1 | 9.2 % | 100 (13.0 %) | 27 (3.5 %) | 2 (0.3 %) | 0 (0.0 %) |
| finale | step_analysis | h1 | 8 | 7 | 676.5 | 646.7 a 709.5 | 677.5 | 115.0 | 5.9 | 5.7 % | 93 (12.5 %) | 25 (3.4 %) | 1 (0.1 %) | 25 (3.4 %) |
| finale | step_analysis | h3 | 8 | 7 | 712.0 | 663.4 a 733.3 | 713.0 | 119.0 | 6.0 | 1.4 % | 84 (11.8 %) | 25 (3.5 %) | 2 (0.3 %) | 47 (6.6 %) |
| finale | step_analysis | h7 | 8 | 7 | 699.0 | 654.2 a 738.2 | 700.0 | 116.8 | 6.0 | 0.0 % | 84 (12.0 %) | 26 (3.7 %) | 1 (0.1 %) | 57 (8.1 %) |
| milieu | mcts_search | legacy | 8 | 7 | 454.7 | 436.0 a 487.3 | 455.4 | 94.8 | 4.8 | 4.9 % | 36 (4.9 %) | 0 (0.0 %) | 0 (0.0 %) | 0 (0.0 %) |
| milieu | mcts_search | h0 | 8 | 7 | 500.4 | 465.8 a 522.7 | 501.1 | 102.9 | 4.9 | 4.2 % | 33 (4.5 %) | 2 (0.3 %) | 0 (0.0 %) | 0 (0.0 %) |
| milieu | mcts_search | h1 | 8 | 7 | 402.3 | 387.0 a 421.6 | 402.9 | 97.7 | 4.1 | 5.7 % | 49 (6.6 %) | 2 (0.3 %) | 0 (0.0 %) | 5 (0.7 %) |
| milieu | mcts_search | h3 | 8 | 7 | 479.6 | 458.1 a 512.0 | 480.3 | 99.4 | 4.8 | 5.5 % | 88 (11.9 %) | 11 (1.5 %) | 0 (0.0 %) | 36 (4.9 %) |
| milieu | mcts_search | h7 | 8 | 7 | 494.2 | 460.2 a 510.1 | 494.9 | 102.4 | 4.8 | 0.8 % | 93 (13.2 %) | 12 (1.7 %) | 0 (0.0 %) | 75 (10.6 %) |
| milieu | step_analysis | legacy | 8 | 7 | 470.1 | 443.3 a 504.9 | 470.7 | 98.0 | 4.8 | 4.9 % | 36 (4.9 %) | 0 (0.0 %) | 0 (0.0 %) | 0 (0.0 %) |
| milieu | step_analysis | h0 | 8 | 7 | 497.4 | 487.7 a 517.7 | 498.2 | 102.3 | 4.9 | 4.2 % | 33 (4.5 %) | 2 (0.3 %) | 0 (0.0 %) | 0 (0.0 %) |
| milieu | step_analysis | h1 | 8 | 7 | 421.6 | 405.1 a 437.4 | 422.2 | 102.4 | 4.1 | 5.7 % | 49 (6.6 %) | 2 (0.3 %) | 0 (0.0 %) | 5 (0.7 %) |
| milieu | step_analysis | h3 | 8 | 7 | 490.8 | 473.6 a 522.1 | 491.5 | 101.7 | 4.8 | 5.5 % | 88 (11.9 %) | 11 (1.5 %) | 0 (0.0 %) | 36 (4.9 %) |
| milieu | step_analysis | h7 | 8 | 7 | 500.4 | 466.7 a 529.6 | 501.2 | 103.7 | 4.8 | 0.8 % | 93 (13.2 %) | 12 (1.7 %) | 0 (0.0 %) | 75 (10.6 %) |
| ouverture | mcts_search | legacy | 8 | 7 | 975.3 | 912.5 a 1023.3 | 976.7 | 144.9 | 6.7 | 4.1 % | 30 (4.1 %) | 0 (0.0 %) | 0 (0.0 %) | 0 (0.0 %) |
| ouverture | mcts_search | h0 | 8 | 7 | 1051.4 | 994.1 a 1084.0 | 1052.9 | 151.7 | 6.9 | 2.9 % | 34 (4.7 %) | 13 (1.8 %) | 0 (0.0 %) | 0 (0.0 %) |
| ouverture | mcts_search | h1 | 8 | 7 | 1004.3 | 951.5 a 1059.8 | 1005.8 | 146.3 | 6.9 | 2.9 % | 37 (5.1 %) | 12 (1.7 %) | 0 (0.0 %) | 4 (0.6 %) |
| ouverture | mcts_search | h3 | 8 | 7 | 1018.2 | 999.4 a 1060.0 | 1019.7 | 146.9 | 6.9 | 1.3 % | 36 (5.1 %) | 13 (1.8 %) | 0 (0.0 %) | 14 (2.0 %) |
| ouverture | mcts_search | h7 | 8 | 7 | 981.7 | 925.3 a 1015.3 | 983.1 | 143.0 | 6.9 | 0.0 % | 38 (5.4 %) | 13 (1.9 %) | 0 (0.0 %) | 25 (3.6 %) |
| ouverture | step_analysis | legacy | 8 | 7 | 990.0 | 935.5 a 1028.2 | 991.5 | 147.1 | 6.7 | 4.1 % | 30 (4.1 %) | 0 (0.0 %) | 0 (0.0 %) | 0 (0.0 %) |
| ouverture | step_analysis | h0 | 8 | 7 | 1020.5 | 994.9 a 1065.8 | 1022.0 | 147.2 | 6.9 | 2.9 % | 34 (4.7 %) | 13 (1.8 %) | 0 (0.0 %) | 0 (0.0 %) |
| ouverture | step_analysis | h1 | 8 | 7 | 1011.0 | 978.1 a 1050.4 | 1012.5 | 147.3 | 6.9 | 2.9 % | 37 (5.1 %) | 12 (1.7 %) | 0 (0.0 %) | 4 (0.6 %) |
| ouverture | step_analysis | h3 | 8 | 7 | 1030.5 | 966.3 a 1059.0 | 1032.0 | 148.7 | 6.9 | 1.3 % | 36 (5.1 %) | 13 (1.8 %) | 0 (0.0 %) | 14 (2.0 %) |
| ouverture | step_analysis | h7 | 8 | 7 | 968.6 | 912.5 a 1012.4 | 970.0 | 141.1 | 6.9 | 0.0 % | 38 (5.4 %) | 13 (1.9 %) | 0 (0.0 %) | 25 (3.6 %) |

## Invariants d'arbre

Non verifies lors de ce passage.
