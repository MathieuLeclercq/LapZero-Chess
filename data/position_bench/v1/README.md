# Banc externe de positions, v1

Positions extraites des archives de broadcasts Lichess (CC BY-SA 4.0, attribution Lichess), annotees par Stockfish hors ligne. Ce banc est interdit a l'entrainement : `training_forbidden` vaut true dans le manifeste, et aucun chargeur de donnees ne le decouvre automatiquement.

- Sources : lichess-broadcasts-2026-08, lichess-broadcasts-2026-07, lichess-broadcasts-2026-06, lichess-broadcasts-2026-05, lichess-broadcasts-2026-04
- Positions : 10000
- Identifiants de recherche : 256
- SHA-256 du JSONL compresse : c93c4b94ce6c63817139c74e87b1d85f2ff0d611eb94505d734a7c48a23e5155
- Moteur d'annotation : Stockfish 19

Les identifiants de recherche sont la liste explicite du manifeste ; l'ordre des positions du fichier est celui des `position_id` croissants.

## Reparation depuis l'audit en cache

Le rapport d'audit conserve les mesures sur les etiquettes d'origine, avant correction. Son champ `passed` n'est pas falsifie : un echec est accepte explicitement avec avertissement. Aucun nouvel audit independant apres correction n'a ete execute.

- Positions corrigees dans le banc : 497
- Budget des references MCTS : 400000 noeuds par coup legal
- Budget demande par position : `annotation_budget_nodes` ; les `nodes` des coups sont les nombres de noeuds effectivement examines.
