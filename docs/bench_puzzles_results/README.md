# Résultats bruts du banc de puzzles

Passages du banc `python_src/puzzle_bench.py` conservés comme actifs de
référence, pour pouvoir refaire des comparaisons appariées sans relancer le
banc. Un dossier par modèle, nommé d'après l'itération du checkpoint. Les
fichiers gardent les noms produits par le banc :

- `<nom>.csv` : mesures par puzzle, bras avec historique réel.
- `<nom>_sans.csv` : les mêmes puzzles, historique vidé.
- `<nom>.meta.json` : modèle et son SHA256, banc et son SHA256, paramètres de
  recherche, durée.
- `<nom>.md` : rapport lisible, avec la section de comparaison appariée entre
  les deux bras.

## Modèles conservés

| Dossier | Modèle | Passages |
|---|---|---|
| `iter436` | `2026_04_30_09h53_iter436_unsupervised.onnx` | 500 et 2500 puzzles |
| `iter506` | `2026_09_22_17h14_iter506_unsupervised.onnx` | 500 puzzles |

## Protocole de référence

Commande lancée depuis `python_src`, protocole identique pour tous les passages
comparables : 700 simulations, c_puct 1.4, lot 8, TT h0, 16 travailleurs Python,
1 worker de recherche, CPU.

```powershell
uv run python puzzle_bench.py --model <chemin.onnx> --limite 2500 `
    --batch-size 8 --cache-history-depth 0 --travailleurs 16 `
    --search-workers 1 --comparer-historique `
    --out-csv ..\out\multicore\<nom>.csv --out-rapport ..\out\multicore\<nom>.md
```

## Comparer deux passages

```powershell
uv run python python_src/dev_tools/compare_bench_models.py `
    docs/bench_puzzles_results/iter436/hist-500.csv `
    docs/bench_puzzles_results/iter506/puzzle506-500.csv
```

L'outil apparie les puzzles par numéro de ligne, calcule le McNemar exact et
les écarts médians de prior, de value et de part de visites.

## Règle de conservation

On ne committe que les passages de référence et les jalons utiles à une
décision, un dossier par modèle concerné. Les vérifications intermédiaires
restent dans `out/`, ignoré par git, et les synthèses vont dans
`docs/superpowers/specs/` ou dans le devlog.
