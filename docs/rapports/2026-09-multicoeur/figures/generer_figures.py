"""Genere les figures du rapport multicœur a partir des mesures conservees.

Les JSON sources vivent dans out/multicore/ (hors depot, volumineux). Les
figures PDF produites sont, elles, commitees avec le rapport.

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
COULEURS_WORKERS = {1: "#9DB8C9", 2: "#5E93B5", 4: "#2F6E99", 8: BLEU}
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


def percentile(v, q):
    v = sorted(v)
    k = (len(v) - 1) * q
    bas = int(k)
    haut = min(bas + 1, len(v) - 1)
    f = k - bas
    return v[bas] * (1 - f) + v[haut] * f


def figure_balayage():
    lots = [0, 1, 2, 4, 8, 16, 32, 64]
    series = {
        "ouverture": [316.8, 325.9, 569.2, 708.5, 785.5, 775.3, 783.1, 813.7],
        "milieu": [328.3, 319.6, 445.9, 446.7, 467.9, 467.2, 471.8, 477.3],
        "finale": [323.7, 316.3, 453.1, 734.8, 708.2, 611.8, 640.3, 646.6],
    }
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    positions = [1, 2, 3, 4, 5, 6, 7, 8]
    for nom, valeurs in series.items():
        ax.plot(positions, valeurs, marker="o", ms=3.5,
                color=COULEURS_POSITION[nom], label=nom)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(l) for l in lots])
    ax.set_xlabel("taille de lot (0 = boucle sequentielle)")
    ax.set_ylabel("simulations par seconde")
    ax.set_title("Balayage de la taille de lot, GPU, 400 simulations")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#DDDDDD", lw=0.6)
    fig.tight_layout()
    fig.savefig(SORTIE / "fig_balayage_batch.pdf")
    plt.close(fig)


def _groupes(nom, chaudes=True):
    d = charger(nom)
    g = {}
    for m in d["mesures"]:
        if chaudes and m.get("pool_etat") != "chaud":
            continue
        cle = (m["position"], m["chemin"], m["worker_count"])
        g.setdefault(cle, []).append(m["simulations"] / m["duree_s"])
    return g


def figure_debit_workers():
    g = _groupes("after.json")
    positions = ["ouverture", "milieu", "finale"]
    workers = [1, 2, 4, 8]
    largeur = 0.19
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for i, w in enumerate(workers):
        meds, bas, haut = [], [], []
        for pos in positions:
            v = g[(pos, "step_analysis", w)]
            med = statistics.median(v)
            meds.append(med)
            bas.append(med - min(v))
            haut.append(max(v) - med)
        x = [j + (i - 1.5) * largeur for j in range(len(positions))]
        ax.bar(x, meds, width=largeur, label=f"w{w}",
               color=COULEURS_WORKERS[w], edgecolor="white", lw=0.4)
        ax.errorbar(x, meds, yerr=[bas, haut], fmt="none", ecolor=GRIS,
                    elinewidth=0.7, capsize=2)
    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels(positions)
    ax.set_ylabel("simulations par seconde (mediane)")
    ax.set_title("Debit par nombre de workers, pool chaud, "
                 "700 simulations, 30 observations")
    ax.legend(frameon=False, ncol=4)
    ax.grid(axis="y", color="#DDDDDD", lw=0.6)
    fig.tight_layout()
    fig.savefig(SORTIE / "fig_debit_workers.pdf")
    plt.close(fig)


def figure_queue_latence():
    g = _groupes("queue.json")
    couples = [("finale", "step_analysis"), ("milieu", "step_analysis"),
               ("ouverture", "step_analysis"),
               ("finale", "mcts_search"), ("milieu", "mcts_search"),
               ("ouverture", "mcts_search")]
    etiquettes = [f"{p[:4]}-{c[:2]}" for p, c in couples]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, (titre, fonction) in zip(
            axes, [("mediane de debit", statistics.median),
                   ("p95 de latence",
                    lambda v: 1.0 / percentile(v, 0.05))]):
        for i, w in enumerate([2, 8]):
            ratios = []
            for pos, ch in couples:
                ref = fonction(g[(pos, ch, 1)])
                val = fonction(g[(pos, ch, w)])
                ratios.append(val / ref)
            y = [j + (i - 0.5) * 0.34 for j in range(len(couples))]
            ax.barh(y, ratios, height=0.32, color=COULEURS_WORKERS[w],
                    label=f"w{w}/w1", edgecolor="white", lw=0.4)
        ax.axvline(1.0, color=GRIS, lw=0.8)
        ax.axvline(1.05, color=ROUGE, lw=0.8, ls="--")
        ax.set_xlim(0, 1.35)
        ax.set_yticks(range(len(couples)))
        ax.set_yticklabels(etiquettes)
        ax.set_title(titre)
        ax.grid(axis="x", color="#DDDDDD", lw=0.6)
    axes[1].legend(frameon=False, loc="center left",
                   bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.suptitle("Rapports w2/w1 et w8/w1, campagne de queue, "
                 "50 observations par configuration", fontsize=10)
    fig.tight_layout(rect=(0, 0, 0.93, 0.94))
    fig.savefig(SORTIE / "fig_queue_latence.pdf")
    plt.close(fig)


def figure_phases():
    """Part du temps mur hors evaluateur, en proportions empilees.

    Les valeurs brutes sont trop petites pour etre lisibles cote a cote :
    c'est la composition de ce residu qui informe, pas sa duree absolue.
    """
    d = charger("phases.json")
    g = {}
    for m in d["mesures"]:
        if m.get("pool_etat") != "chaud" or not m.get("timing"):
            continue
        g.setdefault((m["position"], m["worker_count"]), []).append(m["timing"])
    positions = ["ouverture", "milieu", "finale"]
    composantes = [("selection_ns", "selection"),
                   ("tensor_key_ns", "tenseur et cle"),
                   ("tt_probe_store_ns", "table"),
                   ("expansion_ns", "expansion"),
                   ("backup_ns", "remontees"),
                   ("batch_assembly_ns", "assemblage"),
                   ("worker_wait_ns", "attente workers")]
    couleurs = ["#5E93B5", "#8FB8CE", "#C9DCE7", AMBRE, "#D4A94F",
                "#B9B9B9", ROUGE]
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    lignes, etiquettes = [], []
    for i, w in enumerate([1, 8]):
        for j, pos in enumerate(positions):
            y = i * 3.6 + j
            lignes.append(y)
            etiquettes.append(f"w{w} {pos}")
            total = sum(
                statistics.median(t[champ] for t in g[(pos, w)])
                for champ, _ in composantes)
            gauche = 0.0
            for (champ, _), couleur in zip(composantes, couleurs):
                valeur = statistics.median(
                    t[champ] for t in g[(pos, w)]) / total * 100.0
                ax.barh(y, valeur, left=gauche, height=0.62,
                        color=couleur, edgecolor="white", lw=0.3)
                gauche += valeur
            eval_ms = statistics.median(
                t["evaluator_ns"] for t in g[(pos, w)]) / 1e6
            ax.text(101, y, f"evaluateur {eval_ms:.0f} ms "
                    f"({100 * eval_ms / (total / 1e6 + eval_ms):.1f} % "
                    "du temps mur)", va="center", fontsize=7.5, color=BLEU)
    ax.set_yticks(lignes)
    ax.set_yticklabels(etiquettes, fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_xlabel("part du temps hors evaluateur (%, mediane de 3 mesures)")
    ax.set_title("Composition du temps hors evaluateur, 700 simulations")
    poignees = [plt.Rectangle((0, 0), 1, 1, color=c) for c in couleurs]
    ax.legend(poignees, [e for _, e in composantes], frameon=False,
              fontsize=7, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.28))
    ax.grid(axis="x", color="#DDDDDD", lw=0.6)
    fig.tight_layout()
    fig.savefig(SORTIE / "fig_phases.pdf")
    plt.close(fig)


def figure_tranches_uci():
    donnees = {
        (64, "ouverture"): (799.1, 912.5),
        (64, "milieu"): (373.1, 462.8),
        (64, "finale"): (582.5, 580.0),
        (20, "ouverture"): (503.9, 572.4),
        (20, "milieu"): (244.5, 256.1),
        (20, "finale"): (319.1, 405.5),
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    positions = ["ouverture", "milieu", "finale"]
    for ax, tranche in zip(axes, (64, 20)):
        for j, pos in enumerate(positions):
            w1, w8 = donnees[(tranche, pos)]
            ax.bar(j - 0.17, w1, width=0.32, color="#9DB8C9",
                   edgecolor="white", lw=0.4)
            ax.bar(j + 0.17, w8, width=0.32, color=BLEU,
                   edgecolor="white", lw=0.4)
            gain = 100 * (w8 / w1 - 1)
            ax.text(j, max(w1, w8) + 18, f"{gain:+.0f} %", ha="center",
                    fontsize=8, color=VERT if gain >= 5 else GRIS)
        ax.set_xticks(range(len(positions)))
        ax.set_xticklabels(positions)
        ax.set_title(f"tranches de {tranche} simulations")
        ax.grid(axis="y", color="#DDDDDD", lw=0.6)
    axes[0].set_ylabel("simulations par seconde")
    axes[0].set_ylim(0, 1080)
    poignees = [plt.Rectangle((0, 0), 1, 1, color="#9DB8C9"),
                plt.Rectangle((0, 0), 1, 1, color=BLEU)]
    axes[1].legend(poignees, ["w1", "w8"], frameon=False, fontsize=8,
                   loc="upper right")
    fig.suptitle("Regime reel de la boucle UCI : 704 simulations "
                 "par tranches", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(SORTIE / "fig_tranches_uci.pdf")
    plt.close(fig)


def figure_qualite():
    campagnes = [
        ("Prefiltre\n500", 369, 371, 0.004, -0.006, 0.016),
        ("Complete\n2500", 1927, 1920, -0.0028, -0.008, 0.0024),
        ("Temps egal\n500", 366, 364, -0.004, -0.016, 0.008),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    for i, (nom, w1, w8, delta, bas, haut) in enumerate(campagnes):
        ax.errorbar(i, 100 * delta,
                    yerr=[[100 * (delta - bas)], [100 * (haut - delta)]],
                    fmt="o", color=BLEU, capsize=4, ms=5)
        ax.text(i, 1.72, f"w1 {w1} / w8 {w8}",
                ha="center", va="top", fontsize=7.5, color=GRIS)
    ax.axhline(0, color=GRIS, lw=0.8)
    ax.axhline(-1.0, color=ROUGE, lw=0.9, ls="--")
    ax.text(2.42, -1.05, "marge", color=ROUGE, fontsize=7.5, va="top")
    ax.set_ylim(-2.2, 1.9)
    ax.set_xticks(range(len(campagnes)))
    ax.set_xticklabels([nom for nom, *_ in campagnes])
    ax.set_ylabel("delta apparie (points)")
    ax.set_title("Qualite : delta candidat moins reference et IC95")
    ax.grid(axis="y", color="#DDDDDD", lw=0.6)
    fig.tight_layout()
    fig.savefig(SORTIE / "fig_qualite.pdf")
    plt.close(fig)


if __name__ == "__main__":
    figure_balayage()
    figure_debit_workers()
    figure_queue_latence()
    figure_phases()
    figure_tranches_uci()
    figure_qualite()
    print(f"Figures ecrites dans {SORTIE}")
