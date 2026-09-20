"""Genere les figures du rapport cout de calcul a partir des mesures.

Les JSON sources vivent dans out/multicore/ (hors depot, volumineux). Les
figures PDF produites sont commitees avec le rapport.

Usage : uv run python figures/generer_figures.py  (depuis ce dossier)
"""
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RACINE_DEPOT = Path(__file__).resolve().parents[4]
MESURES = RACINE_DEPOT / "out" / "multicore"
SORTIE = Path(__file__).resolve().parent

BLEU = "#0B3C5D"
VERT = "#2E7D32"
AMBRE = "#8A5A00"
ROUGE = "#B3261E"
GRIS = "#666666"
COULEURS_POSITION = {"ouverture": "#5E93B5", "milieu": AMBRE,
                     "finale": VERT}

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def charger(nom):
    return json.loads((MESURES / nom).read_text(encoding="utf-8"))


def mediane(valeurs):
    return statistics.median(valeurs)


def par_lot(nom):
    d = charger(nom)
    g = {}
    for m in d["measurements"]:
        g.setdefault(m["batch"], []).append(m)
    return {
        lot: {
            "appel": mediane([m["ms_appel"] for m in mesures]),
            "position": mediane([m["ms_position"] for m in mesures]),
            "run": mediane([m["run_ns"] for m in mesures]) / 1e6,
        }
        for lot, mesures in g.items()
    }


def par_groupe(chemins, chemin_recherche):
    groupes = {}
    for chemin in chemins:
        d = charger(chemin)
        for m in d["mesures"]:
            if m["pool_etat"] != "chaud":
                continue
            if m["chemin"] != chemin_recherche:
                continue
            groupes.setdefault((m["position"], m["chemin"]), []).append(
                m["simulations"] / m["duree_s"])
    return groupes


def medianes(groupes):
    return {cle: mediane(valeurs) for cle, valeurs in groupes.items()}


def figure_courbe_evaluateur():
    fixe = par_lot("evaluator-batch-gpu.json")
    variable = par_lot("evaluator-variable-gpu.json")
    lots = sorted(fixe)
    variable_lots = sorted(variable)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    ax = axes[0]
    ax.plot(lots, [fixe[l]["position"] for l in lots], "o-", color=BLEU,
            ms=3.5, lw=1.4, label="lots fixes")
    ax.plot(variable_lots, [variable[l]["position"] for l in variable_lots],
            "s--", color=ROUGE, ms=3.5, lw=1.4, label="lots alternes 1 a 8")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64", "128", "256"])
    ax.set_xlabel("taille de lot")
    ax.set_ylabel("ms par position")
    ax.set_title("Cout par position")
    ax.grid(True, which="both", color="#DDDDDD", lw=0.5)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    ax = axes[1]
    ax.plot(lots, [fixe[l]["appel"] for l in lots], "o-", color=BLEU,
            ms=3.5, lw=1.4, label="appel total")
    ax.plot(lots, [fixe[l]["run"] for l in lots], "^:", color=VERT,
            ms=3.5, lw=1.2, label="session Run")
    ax.plot(variable_lots, [variable[l]["appel"] for l in variable_lots],
            "s--", color=ROUGE, ms=3.5, lw=1.4, label="appel, lots alternes")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64", "128", "256"])
    ax.set_xlabel("taille de lot")
    ax.set_ylabel("ms par appel")
    ax.set_title("Cout par appel")
    ax.grid(True, which="both", color="#DDDDDD", lw=0.5)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(SORTIE / "fig_evaluateur_courbe.pdf")
    plt.close(fig)


def figure_variable_contre_fixe():
    fixe = par_lot("evaluator-batch-gpu.json")
    variable = par_lot("evaluator-variable-gpu.json")
    ab = [median_of("ab-fix-a.json"), median_of("ab-fix-b.json"),
          median_of("ab-var-a.json"), median_of("ab-var-b.json")]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    lots = [1, 2, 4, 8]
    ax = axes[0]
    x = range(len(lots))
    ax.bar([i - 0.19 for i in x], [fixe[l]["appel"] for l in lots],
           width=0.36, color=BLEU, edgecolor="white", lw=0.4, label="lots fixes")
    ax.bar([i + 0.19 for i in x], [variable[l]["appel"] for l in lots],
           width=0.36, color=ROUGE, edgecolor="white", lw=0.4,
           label="lots alternes 1 a 8")
    for i, lot in enumerate(lots):
        ax.text(i, max(fixe[lot]["appel"], variable[lot]["appel"]) + 0.4,
                f"x{variable[lot]['appel'] / fixe[lot]['appel']:.1f}",
                ha="center", fontsize=8, color=GRIS)
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(l) for l in lots])
    ax.set_xlabel("taille de lot")
    ax.set_ylabel("ms par appel")
    ax.set_ylim(0, 16)
    ax.set_title("Alternance contre forme fixe")
    ax.grid(axis="y", color="#DDDDDD", lw=0.6)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.bar([0, 1], [ab[0], ab[2]], width=0.5, color=[BLEU, ROUGE],
           edgecolor="white", lw=0.4)
    ax.errorbar([0, 1], [ab[0], ab[2]], yerr=[abs(ab[1] - ab[0]),
                                              abs(ab[3] - ab[2])],
                fmt="none", ecolor=GRIS, elinewidth=1.0, capsize=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["fixe 8", "alterne 1 a 8"])
    ax.set_ylabel("ms par appel")
    ax.set_ylim(0, 16)
    ax.set_title("A/B interleaved, lot 8")
    ax.grid(axis="y", color="#DDDDDD", lw=0.6)
    for position, valeur in ((0, ab[0]), (1, ab[2])):
        ax.text(position, valeur + 0.5, f"{valeur:.2f}", ha="center",
                fontsize=8, color=GRIS)

    fig.tight_layout()
    fig.savefig(SORTIE / "fig_variable_contre_fixe.pdf")
    plt.close(fig)


def median_of(nom):
    d = charger(nom)
    return mediane([m["ms_appel"] for m in d["measurements"]])


def figure_gain_lot_fixe():
    off8 = medianes(par_groupe(["fixed-off-r1.json"], "step_analysis"))
    on8 = medianes(par_groupe(["fixed-on-r1.json"], "step_analysis"))
    off1 = medianes(par_groupe(["mono-off-r1.json"], "step_analysis"))
    on1 = medianes(par_groupe(["mono-on-r1.json"], "step_analysis"))

    cles = ["ouverture", "milieu", "finale"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
    for ax, (titre, avant, apres) in zip(
            axes, (("workers 8", off8, on8), ("mono, workers 1", off1, on1))):
        valeurs_avant = [avant[(p, "step_analysis")] for p in cles]
        valeurs_apres = [apres[(p, "step_analysis")] for p in cles]
        x = range(len(cles))
        ax.bar([i - 0.19 for i in x], valeurs_avant, width=0.36,
               color="#9DB8C9", edgecolor="white", lw=0.4)
        ax.bar([i + 0.19 for i in x], valeurs_apres, width=0.36, color=BLEU,
               edgecolor="white", lw=0.4)
        for i, (a, b) in enumerate(zip(valeurs_avant, valeurs_apres)):
            ax.text(i, b + 40, f"x{b / a:.2f}", ha="center", fontsize=8,
                    color=VERT)
        ax.set_xticks(list(x))
        ax.set_xticklabels(cles)
        ax.set_title(titre)
        ax.grid(axis="y", color="#DDDDDD", lw=0.6)
    axes[0].set_ylabel("simulations par seconde")
    axes[0].set_ylim(0, 2500)
    axes[0].text(-0.19, 120, "avant", ha="center", fontsize=8, color=GRIS)
    axes[0].text(0.19, 120, "apres", ha="center", fontsize=8, color=BLEU)
    fig.suptitle("Effet du lot de forme fixe, 700 simulations, "
                 "step_analysis, round 1", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(SORTIE / "fig_gain_lot_fixe.pdf")
    plt.close(fig)


def figure_p95():
    off = par_groupe(["fixed-off-r1.json", "fixed-off-r2.json"],
                     "step_analysis")
    on = par_groupe(["fixed-on-r1.json", "fixed-on-r2.json"],
                    "step_analysis")

    def percentile(v, q):
        v = sorted(v)
        k = (len(v) - 1) * q
        b = int(k)
        h = min(b + 1, len(v) - 1)
        f = k - b
        return v[b] * (1 - f) + v[h] * f

    cles = ["ouverture", "milieu", "finale"]
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    x = range(len(cles))
    lat_avant = [700.0 / percentile(off[(p, "step_analysis")], 0.05) * 1000
                 for p in cles]
    lat_apres = [700.0 / percentile(on[(p, "step_analysis")], 0.05) * 1000
                 for p in cles]
    ax.bar([i - 0.19 for i in x], lat_avant, width=0.36, color="#9DB8C9",
           edgecolor="white", lw=0.4, label="avant")
    ax.bar([i + 0.19 for i in x], lat_apres, width=0.36, color=BLEU,
           edgecolor="white", lw=0.4, label="apres")
    for i, (a, b) in enumerate(zip(lat_avant, lat_apres)):
        ax.text(i, b + 20, f"x{b / a:.2f}", ha="center", fontsize=8,
                color=VERT)
    ax.set_xticks(list(x))
    ax.set_xticklabels(cles)
    ax.set_ylabel("p95 de latence (ms)")
    ax.set_title("Queue de latence, workers 8")
    ax.grid(axis="y", color="#DDDDDD", lw=0.6)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(SORTIE / "fig_p95.pdf")
    plt.close(fig)


def figure_divergence():
    configs = ["base", "vl2", "vl3", "fpu45", "vl2fpu45", "att8"]
    libelles = ["defaut", "vloss 2", "vloss 3", "fpu 0.45",
                "vloss 2\nfpu 0.45", "tentatives\nx8"]
    positions = ["ouverture", "milieu", "finale"]

    debits = {}
    for nom in configs:
        d = charger(f"sweep-{nom}.json")
        for pos in positions:
            lot = [m for m in d["mesures"]
                   if m["pool_etat"] == "chaud"
                   and m["chemin"] == "step_analysis"
                   and m["position"] == pos]
            debits[(nom, pos)] = mediane(
                m["simulations"] / m["duree_s"] for m in lot)

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.1))
    ax = axes[0]
    x = range(len(configs))
    for i, pos in enumerate(positions):
        decalage = (i - 1) * 0.24
        ax.bar([j + decalage for j in x],
               [debits[(nom, pos)] for nom in configs],
               width=0.22, color=COULEURS_POSITION[pos], edgecolor="white",
               lw=0.4, label=pos)
    ax.set_xticks(list(x))
    ax.set_xticklabels(libelles, fontsize=8)
    ax.set_ylabel("simulations par seconde")
    ax.set_title("Debit du balayage de divergence")
    ax.grid(axis="y", color="#DDDDDD", lw=0.6)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    noms = ["vloss 2\nfpu 0.45", "vloss 2"]
    deltas = [-0.018, -0.024]
    bas = [-0.04005, -0.05]
    haut = [0.004, 0.0]
    couleurs = [ROUGE, ROUGE]
    ax.barh(range(2), [d * 100 for d in deltas], color=couleurs, height=0.45)
    ax.errorbar([d * 100 for d in deltas], range(2),
                xerr=[[(d - b) * 100 for d, b in zip(deltas, bas)],
                      [(h - d) * 100 for d, h in zip(deltas, haut)]],
                fmt="none", ecolor=GRIS, elinewidth=1.2, capsize=3)
    ax.axvline(-1.0, color=AMBRE, ls="--", lw=1.2)
    ax.text(-0.9, 1.55, "marge -1 point", color=AMBRE, fontsize=8)
    ax.axvline(0.0, color=GRIS, lw=0.8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(noms, fontsize=8)
    ax.set_xlabel("delta de resolution (point)")
    ax.set_xlim(-6, 2)
    ax.set_title("Qualite, prefiltre 500 puzzles")
    ax.grid(axis="x", color="#DDDDDD", lw=0.6)

    fig.tight_layout()
    fig.savefig(SORTIE / "fig_divergence.pdf")
    plt.close(fig)


def figure_cas_chaud():
    lignes = charger("hot-tree.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), sharex=True)
    for ax, (nom, titre) in zip(
            axes, (("ouverture", "Ouverture"), ("finale", "Finale"))):
        lot = [l for l in lignes if l["position"] == nom]
        x = [l["coup"] for l in lot]
        ax.plot(x, [l["sims_par_seconde"] for l in lot], "o-", color=BLEU,
                ms=3.5, lw=1.4, label="sims/s")
        ax.set_ylim(0, 2600)
        ax.set_xlabel("coup")
        ax.grid(True, color="#DDDDDD", lw=0.5)
        ax2 = ax.twinx()
        ax2.plot(x, [l["tt_hits"] for l in lot], "s--", color=AMBRE,
                 ms=3.0, lw=1.1, label="hits de table")
        ax2.set_ylim(0, 620)
        ax2.spines["right"].set_visible(True)
        if nom == "finale":
            ax2.set_ylabel("hits de table")
        else:
            ax.set_ylabel("simulations par seconde")
        ax.set_title(titre)
        lignes_legendes = ax.get_lines() + ax2.get_lines()
        ax.legend(lignes_legendes, [l.get_label() for l in lignes_legendes],
                  frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Cas chaud : le debit tient, les appels reseau restent "
                 "a 700 par coup", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(SORTIE / "fig_cas_chaud.pdf")
    plt.close(fig)


if __name__ == "__main__":
    figure_courbe_evaluateur()
    figure_variable_contre_fixe()
    figure_gain_lot_fixe()
    figure_p95()
    figure_divergence()
    figure_cas_chaud()
    print("figures generees dans", SORTIE)
