# Banc de recherche : resultats

Modele : `2026_04_23_23h25_iter316_unsupervised.onnx`, iteration 316, global_step 19415
Protocole : 5 passages, 400 simulations, c_puct 1.4, GPU, un seul processus, batches demandes [0, 1, 2, 4, 8, 16, 32, 64]

Trois grandeurs distinctes. Les **simulations par seconde** mesurent le
debit de la recherche. Les **positions reseau par seconde** comptent les
positions effectivement evaluees, les **appels batch par seconde** les
lancements physiques et leur ratio est le **remplissage**. Le **taux de
table** est la part des consultations reussies.

## Debit

| Position | Chemin | batch | passages | sims/s (med) | sims/s (min a max) | positions reseau/s | appels batch/s | remplissage | taux table |
|---|---|---|---|---|---|---|---|---|---|
| finale | mcts_search | 0 | 5 | 323.7 | 294.1 a 358.0 | 324.5 | 324.5 | 1.0 | 10.7 % |
| finale | mcts_search | 1 | 5 | 316.3 | 306.1 a 340.7 | 317.1 | 317.1 | 1.0 | 10.7 % |
| finale | mcts_search | 2 | 5 | 453.1 | 405.5 a 504.4 | 454.3 | 233.4 | 1.9 | 11.3 % |
| finale | mcts_search | 4 | 5 | 734.8 | 714.8 a 752.3 | 736.6 | 198.4 | 3.7 | 11.1 % |
| finale | mcts_search | 8 | 5 | 708.2 | 680.4 a 725.8 | 710.0 | 123.9 | 5.7 | 11.7 % |
| finale | mcts_search | 16 | 5 | 611.8 | 549.3 a 651.0 | 613.3 | 88.7 | 6.9 | 11.7 % |
| finale | mcts_search | 32 | 5 | 640.3 | 619.0 a 696.6 | 641.9 | 89.6 | 7.2 | 11.3 % |
| finale | mcts_search | 64 | 5 | 646.6 | 597.3 a 690.4 | 648.2 | 90.5 | 7.2 | 11.3 % |
| finale | step_analysis | 0 | 5 | 318.4 | 294.7 a 353.5 | 319.2 | 319.2 | 1.0 | 10.7 % |
| finale | step_analysis | 1 | 5 | 329.4 | 308.3 a 348.0 | 330.2 | 330.2 | 1.0 | 10.7 % |
| finale | step_analysis | 2 | 5 | 459.0 | 355.4 a 495.4 | 460.1 | 236.4 | 1.9 | 11.3 % |
| finale | step_analysis | 4 | 5 | 743.0 | 680.7 a 761.5 | 744.9 | 200.6 | 3.7 | 11.1 % |
| finale | step_analysis | 8 | 5 | 686.0 | 627.1 a 713.5 | 687.7 | 120.1 | 5.7 | 11.7 % |
| finale | step_analysis | 16 | 5 | 602.4 | 578.9 a 639.4 | 603.9 | 87.4 | 6.9 | 11.7 % |
| finale | step_analysis | 32 | 5 | 639.8 | 605.1 a 700.8 | 641.4 | 89.6 | 7.2 | 11.3 % |
| finale | step_analysis | 64 | 5 | 661.3 | 592.4 a 690.9 | 663.0 | 92.6 | 7.2 | 11.3 % |
| milieu | mcts_search | 0 | 5 | 328.3 | 315.5 a 345.7 | 329.1 | 329.1 | 1.0 | 3.8 % |
| milieu | mcts_search | 1 | 5 | 319.6 | 306.6 a 347.6 | 320.4 | 320.4 | 1.0 | 3.8 % |
| milieu | mcts_search | 2 | 5 | 445.9 | 416.3 a 475.1 | 447.0 | 233.0 | 1.9 | 4.3 % |
| milieu | mcts_search | 4 | 5 | 446.7 | 427.6 a 455.7 | 447.8 | 130.7 | 3.4 | 4.5 % |
| milieu | mcts_search | 8 | 5 | 467.9 | 452.0 a 482.3 | 469.1 | 94.7 | 5.0 | 5.0 % |
| milieu | mcts_search | 16 | 5 | 467.2 | 435.3 a 491.8 | 468.4 | 85.3 | 5.5 | 4.3 % |
| milieu | mcts_search | 32 | 5 | 471.8 | 459.2 a 506.0 | 473.0 | 88.5 | 5.3 | 4.1 % |
| milieu | mcts_search | 64 | 5 | 477.3 | 456.6 a 504.0 | 478.5 | 89.5 | 5.3 | 4.1 % |
| milieu | step_analysis | 0 | 5 | 322.9 | 296.5 a 332.6 | 323.7 | 323.7 | 1.0 | 3.8 % |
| milieu | step_analysis | 1 | 5 | 319.7 | 304.0 a 337.4 | 320.5 | 320.5 | 1.0 | 3.8 % |
| milieu | step_analysis | 2 | 5 | 445.0 | 418.6 a 469.4 | 446.1 | 232.5 | 1.9 | 4.3 % |
| milieu | step_analysis | 4 | 5 | 447.6 | 425.8 a 459.2 | 448.8 | 130.9 | 3.4 | 4.5 % |
| milieu | step_analysis | 8 | 5 | 483.8 | 461.8 a 488.5 | 485.0 | 98.0 | 5.0 | 5.0 % |
| milieu | step_analysis | 16 | 5 | 472.9 | 431.5 a 497.5 | 474.1 | 86.3 | 5.5 | 4.3 % |
| milieu | step_analysis | 32 | 5 | 467.5 | 452.6 a 505.9 | 468.7 | 87.7 | 5.3 | 4.1 % |
| milieu | step_analysis | 64 | 5 | 468.1 | 452.7 a 492.5 | 469.3 | 87.8 | 5.3 | 4.1 % |
| ouverture | mcts_search | 0 | 5 | 316.8 | 296.1 a 329.2 | 317.6 | 317.6 | 1.0 | 2.7 % |
| ouverture | mcts_search | 1 | 5 | 325.9 | 307.2 a 343.9 | 326.7 | 326.7 | 1.0 | 2.7 % |
| ouverture | mcts_search | 2 | 5 | 569.2 | 545.1 a 608.0 | 570.6 | 290.3 | 2.0 | 2.7 % |
| ouverture | mcts_search | 4 | 5 | 708.5 | 631.9 a 763.7 | 710.3 | 189.5 | 3.7 | 2.7 % |
| ouverture | mcts_search | 8 | 5 | 785.5 | 761.4 a 799.5 | 787.5 | 123.7 | 6.4 | 2.7 % |
| ouverture | mcts_search | 16 | 5 | 775.3 | 716.5 a 793.0 | 777.2 | 89.2 | 8.7 | 2.2 % |
| ouverture | mcts_search | 32 | 5 | 783.1 | 765.3 a 879.6 | 785.0 | 80.3 | 9.8 | 2.0 % |
| ouverture | mcts_search | 64 | 5 | 813.7 | 802.5 a 896.8 | 815.7 | 83.4 | 9.8 | 2.0 % |
| ouverture | step_analysis | 0 | 5 | 313.1 | 307.7 a 353.2 | 313.9 | 313.9 | 1.0 | 2.7 % |
| ouverture | step_analysis | 1 | 5 | 313.4 | 304.0 a 335.7 | 314.2 | 314.2 | 1.0 | 2.7 % |
| ouverture | step_analysis | 2 | 5 | 546.3 | 540.3 a 590.2 | 547.7 | 278.6 | 2.0 | 2.7 % |
| ouverture | step_analysis | 4 | 5 | 686.7 | 676.5 a 731.5 | 688.4 | 183.7 | 3.7 | 2.7 % |
| ouverture | step_analysis | 8 | 5 | 763.8 | 737.2 a 783.4 | 765.7 | 120.3 | 6.4 | 2.7 % |
| ouverture | step_analysis | 16 | 5 | 762.1 | 751.5 a 799.1 | 764.0 | 87.6 | 8.7 | 2.2 % |
| ouverture | step_analysis | 32 | 5 | 823.9 | 754.8 a 838.8 | 825.9 | 84.4 | 9.8 | 2.0 % |
| ouverture | step_analysis | 64 | 5 | 852.0 | 821.0 a 927.6 | 854.1 | 87.3 | 9.8 | 2.0 % |

## Invariants d'arbre

Non verifies lors de ce passage.
