# Comparaison des politiques TT

Puzzles communs valides : **2500**

## Resultats par politique

| Politique | n | Resolus | Reussite | Duree cumulee |
|---|---:|---:|---:|---:|
| legacy | 2500 | 1920 | 76.80 % | 14171.0 s |
| h0 | 2500 | 1926 | 77.04 % | 19494.8 s |
| h1 | 2500 | 1930 | 77.20 % | 19242.8 s |
| h7 | 2500 | 1928 | 77.12 % | 14415.8 s |

## Comparaisons appariees

Le delta est le taux de droite moins le taux de gauche.

| Gauche | Droite | Gauche seule | Droite seule | Accord coups | Delta reussite | McNemar p |
|---|---|---:|---:|---:|---:|---:|
| legacy | h0 | 9 | 15 | 97.88 % | +0.24 points | 0.307 |
| legacy | h1 | 11 | 21 | 97.56 % | +0.40 points | 0.112 |
| legacy | h7 | 13 | 21 | 97.32 % | +0.32 points | 0.23 |
| h0 | h1 | 7 | 11 | 98.12 % | +0.16 points | 0.48 |
| h0 | h7 | 10 | 12 | 97.96 % | +0.08 points | 0.831 |
| h1 | h7 | 11 | 9 | 98.36 % | -0.08 points | 0.823 |
