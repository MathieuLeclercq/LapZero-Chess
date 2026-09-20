"""Genere les figures du rapport d'entrainement (modele iter436).

Sources : export wandb local dans out/wandb_export/ (hors depot). Sortie :
figures PDF commitees avec le rapport.

Usage : uv run python figures/generer_figures.py  (depuis ce dossier)
"""
import csv
import datetime as dt
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

RACINE = Path(__file__).resolve().parents[4]
EXPORT = RACINE / "out" / "wandb_export"
SORTIE = Path(__file__).resolve().parent

BLEU = "#0B3C5D"
BLEU_CLAIR = "#5E93B5"
VERT = "#2E7D32"
VERT_CLAIR = "#7CB342"
AMBRE = "#8A5A00"
ROUGE = "#B3261E"
GRIS = "#666666"

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "legend.frameon": False,
    "legend.fontsize": 8,
})


def lire_supervise(fragment):
    d = next(EXPORT.glob(f"*{fragment}*"))
    with (d / "history.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {
        "step": np.array([float(r["trainer/global_step"]) for r in rows]),
        "policy": np.array([float(r["train/policy_loss"]) for r in rows]),
        "value": np.array([float(r["train/value_loss"]) for r in rows]),
        "acc": np.array([float(r["train/policy_acc"]) for r in rows]),
    }


def lire_selfplay():
    with (EXPORT / "merged_selfplay.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with (EXPORT / "runs_augmented.json").open(encoding="utf-8") as f:
        runs = {r["slug"]: r for r in json.load(f)}

    def colonne(nom):
        return np.array([float(r[nom]) if r[nom] else np.nan for r in rows])

    return {
        "iteration": colonne("iteration"),
        "step": colonne("global_step"),
        "new_positions": colonne("new_positions"),
        "buffer": colonne("buffer_size"),
        "longueur": colonne("avg_game_length"),
        "games": colonne("games"),
        "nulles": colonne("draw_rate"),
        "nulles_rep": colonne("draws_repetition"),
        "nulles_50": colonne("draws_50_moves"),
        "nulles_pat": colonne("draws_stalemate"),
        "nulles_mat": colonne("draws_insuff_mat"),
        "nulles_max": colonne("draws_max_moves"),
        "perte": colonne("epoch_loss"),
        "perte_policy": colonne("epoch_policy_loss"),
        "perte_value": colonne("epoch_value_loss"),
        "lr": colonne("learning_rate"),
        "elo": colonne("elo"),
        "winrate": colonne("winrate"),
        "sf_elo": colonne("sf_elo"),
        "parties_par_s": colonne("games_per_sec"),
        "positions_par_s": colonne("saved_pos_per_sec"),
        "batch": np.array([float(runs[r["run"]]["batch"] or np.nan) for r in rows]),
    }


def mediane_glissante(x, y, fenetre):
    out = np.full_like(y, np.nan, dtype=float)
    for i in range(len(y)):
        debut = max(0, i - fenetre // 2)
        fin = min(len(y), i + fenetre // 2 + 1)
        vals = y[debut:fin]
        vals = vals[~np.isnan(vals)]
        if len(vals):
            out[i] = np.median(vals)
    return out


def fig_supervise_lichess(sup):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    ax = axes[0]
    ax.plot(sup["step"], sup["policy"], color=BLEU, lw=1.3, label="perte policy")
    ax.plot(sup["step"], sup["value"], color=VERT, lw=1.3, label="perte value")
    ax.axvline(9746, color=GRIS, ls=":", lw=1)
    ax.annotate("frontière d'époque\n(9 746 steps)", xy=(9746, 1.35),
                xytext=(10400, 1.28), fontsize=7, color=GRIS,
                arrowprops={"arrowstyle": "-", "color": GRIS, "lw": 0.7})
    ax.annotate("la perte policy\ndémarre à 5,14\n(hors cadre)", xy=(300, 1.85),
                xytext=(1300, 1.12), fontsize=7, color=BLEU,
                arrowprops={"arrowstyle": "->", "color": BLEU, "lw": 0.7})
    ax.set_ylim(0, 2)
    ax.set_xlabel("steps d'optimisation (lot 4096)")
    ax.set_ylabel("perte")
    ax.set_title("Pertes, prétraining Lichess")
    ax.grid(True, color="#DDDDDD", lw=0.5)
    ax.legend(loc="lower right")

    ax = axes[1]
    ax.plot(sup["step"], 100 * sup["acc"], color=BLEU, lw=1.3)
    ax.axvline(9746, color=GRIS, ls=":", lw=1)
    ax.annotate("5,0 % au départ\n(poids aléatoires)", xy=(49, 5), xytext=(2600, 12),
                fontsize=7, color=BLEU,
                arrowprops={"arrowstyle": "->", "color": BLEU, "lw": 0.7})
    ax.annotate("50,0 % à la fin", xy=(19449, 50), xytext=(12300, 41),
                fontsize=7, color=BLEU,
                arrowprops={"arrowstyle": "->", "color": BLEU, "lw": 0.7})
    ax.set_xlabel("steps d'optimisation (lot 4096)")
    ax.set_ylabel("précision top-1 (%)")
    ax.set_title("Précision du coup joué")
    ax.set_ylim(0, 55)
    ax.grid(True, color="#DDDDDD", lw=0.5)

    fig.tight_layout()
    fig.savefig(SORTIE / "fig_supervise_lichess.pdf")
    plt.close(fig)


def fig_supervise_gm(gm):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    x = gm["step"] - gm["step"][0]
    ax = axes[0]
    ax.plot(x, gm["policy"], color=AMBRE, lw=1.3, label="perte policy")
    ax.plot(x, gm["value"], color=VERT, lw=1.3, label="perte value")
    ax.set_xlabel("steps d'optimisation (lot 2048, relatifs)")
    ax.set_ylabel("perte")
    ax.set_ylim(0, 2)
    ax.set_title("Pertes, affinage grands maîtres")
    ax.grid(True, color="#DDDDDD", lw=0.5)
    ax.legend(loc="center right")

    ax = axes[1]
    ax.plot(x, 100 * gm["acc"], color=AMBRE, lw=1.3)
    ax.annotate("40,4 % au départ\n(modèle Lichess)", xy=(0, 40.4), xytext=(2600, 30),
                fontsize=7, color=AMBRE,
                arrowprops={"arrowstyle": "->", "color": AMBRE, "lw": 0.7})
    ax.annotate("49,3 % à la fin", xy=(18900, 49.3), xytext=(11200, 44.5),
                fontsize=7, color=AMBRE,
                arrowprops={"arrowstyle": "->", "color": AMBRE, "lw": 0.7})
    ax.set_xlabel("steps d'optimisation (lot 2048, relatifs)")
    ax.set_ylabel("précision top-1 (%)")
    ax.set_ylim(0, 55)
    ax.set_title("Précision du coup joué")
    ax.grid(True, color="#DDDDDD", lw=0.5)

    fig.tight_layout()
    fig.savefig(SORTIE / "fig_supervise_gm.pdf")
    plt.close(fig)


def fig_transfert():
    runs = [
        ("hamgvtvc", "transfert, run 1", BLEU),
        ("c3t86nnd", "transfert, run 2", VERT),
        ("jhqhhx5q", "poids aléatoires (lignée retenue)", ROUGE),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    for fragment, nom, couleur in runs:
        s = lire_supervise(fragment)
        axes[0].plot(s["step"], 100 * s["acc"], color=couleur, lw=1.3, label=nom)
        axes[1].plot(s["step"], s["value"], color=couleur, lw=1.3, label=nom)
    axes[0].set_xlabel("steps d'optimisation (lot 4096)")
    axes[0].set_ylabel("précision top-1 (%)")
    axes[0].set_title("Précision du coup joué")
    axes[0].set_ylim(0, 55)
    axes[0].grid(True, color="#DDDDDD", lw=0.5)
    axes[0].legend(loc="lower right")
    axes[1].set_xlabel("steps d'optimisation (lot 4096)")
    axes[1].set_ylabel("perte value")
    axes[1].set_title("Perte value")
    axes[1].set_ylim(0.5, 2.2)
    axes[1].grid(True, color="#DDDDDD", lw=0.5)
    axes[1].legend(loc="center right")
    fig.tight_layout()
    fig.savefig(SORTIE / "fig_transfert.pdf")
    plt.close(fig)


def fig_selfplay_pertes(sp):
    it = sp["iteration"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    ax = axes[0]
    for cle, nom, couleur in [
        ("perte", "perte totale", BLEU),
        ("perte_policy", "perte policy", VERT),
        ("perte_value", "perte value", AMBRE),
    ]:
        y = sp[cle]
        ax.plot(it, y, color=couleur, lw=0.7, alpha=0.35)
        ax.plot(it, mediane_glissante(it, y, 21), color=couleur, lw=1.4, label=nom)
    ax.set_xlabel("itération")
    ax.set_ylabel("perte (moyenne par itération)")
    ax.set_title("Pertes du self-play")
    ax.grid(True, color="#DDDDDD", lw=0.5)
    ax.legend(loc="center right")

    ax = axes[1]
    ax.step(it, sp["lr"], where="post", color=ROUGE, lw=1.3)
    ax.set_yscale("log")
    ax.set_xlabel("itération")
    ax.set_ylabel("learning rate")
    ax.set_title("Learning rate (AdamW)")
    ax.grid(True, which="both", color="#DDDDDD", lw=0.5)
    ax.set_ylim(3e-6, 1.2e-4)
    for x, y, texte in [
        (1, 5e-6, "5e-6"),
        (10, 5e-5, "5e-5"),
        (29, 2e-5, "2e-5"),
        (150, 4e-5, "4e-5"),
    ]:
        ax.text(x, y * 1.35, texte, fontsize=7, color=ROUGE, ha="left", va="bottom")

    fig.tight_layout()
    fig.savefig(SORTIE / "fig_selfplay_pertes.pdf")
    plt.close(fig)


def fig_selfplay_volume(sp):
    it = sp["iteration"]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6))

    ax = axes[0]
    ax.bar(it, sp["new_positions"] / 1000, width=0.9, color=BLEU_CLAIR)
    imax = int(np.nanargmax(sp["new_positions"]))
    ax.annotate("36,6 k\n(itér. 36)", xy=(it[imax], sp["new_positions"][imax] / 1000),
                xytext=(it[imax] + 40, 30), fontsize=6.5, color=GRIS,
                arrowprops={"arrowstyle": "->", "color": GRIS, "lw": 0.6})
    ax.set_xlabel("itération")
    ax.set_ylabel("nouvelles positions (milliers)")
    ax.set_title("Positions par itération")
    ax.grid(True, axis="y", color="#DDDDDD", lw=0.5)

    ax = axes[1]
    ax.plot(it, sp["buffer"] / 1000, color=ROUGE, lw=1.3)
    ax.axhline(750, color=GRIS, ls=":", lw=1)
    ax.annotate("750 k", xy=(10, 750), xytext=(25, 800), fontsize=7, color=GRIS)
    ax.annotate("pic à 1,09 M\n(cap porté à 2 M\nles 15-16 avril)", xy=(96, 1088),
                xytext=(130, 950), fontsize=6.5, color=ROUGE,
                arrowprops={"arrowstyle": "->", "color": ROUGE, "lw": 0.6})
    ax.set_xlabel("itération")
    ax.set_ylabel("buffer (milliers de positions)")
    ax.set_title("Taille du buffer")
    ax.grid(True, color="#DDDDDD", lw=0.5)

    ax = axes[2]
    cumul_positions = np.nancumsum(np.nan_to_num(sp["new_positions"])) / 1e6
    cumul_plies = np.nancumsum(np.nan_to_num(sp["games"] * sp["longueur"])) / 1e6
    ax.plot(it, cumul_positions, color=BLEU, lw=1.4, label="positions conservées")
    ax.plot(it, cumul_plies, color=GRIS, lw=1.2, ls="--", label="coups joués")
    ax.set_xlabel("itération")
    ax.set_ylabel("cumul (millions)")
    ax.set_title("Volume cumulé")
    ax.grid(True, color="#DDDDDD", lw=0.5)
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(SORTIE / "fig_selfplay_volume.pdf")
    plt.close(fig)


def fig_selfplay_qualite(sp):
    it = sp["iteration"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    ax = axes[0]
    ax.plot(it, sp["longueur"], color=BLEU, lw=1.3)
    ax.set_xlabel("itération")
    ax.set_ylabel("longueur moyenne des parties (plies)", color=BLEU)
    ax.tick_params(axis="y", colors=BLEU)
    ax.set_title("Longueur et nulles")
    ax.grid(True, color="#DDDDDD", lw=0.5)
    ax2 = ax.twinx()
    ax2.plot(it, 100 * sp["nulles"], color=VERT, lw=1.2)
    ax2.set_ylabel("taux de nulles (%)", color=VERT)
    ax2.tick_params(axis="y", colors=VERT)
    ax2.spines["top"].set_visible(False)

    ax = axes[1]
    causes = [
        ("nulles_rep", "répétition", BLEU),
        ("nulles_50", "règle des 50 coups", VERT),
        ("nulles_pat", "pat", AMBRE),
        ("nulles_mat", "matériel insuffisant", ROUGE),
        ("nulles_max", "limite de coups", GRIS),
    ]
    bas = np.zeros_like(it, dtype=float)
    for cle, nom, couleur in causes:
        y = np.nan_to_num(sp[cle])
        y = np.where(y >= 1, y / np.nan_to_num(sp["games"], nan=512), y)
        y = np.clip(y, 0, 1)
        ax.fill_between(it, 100 * bas, 100 * (bas + y), color=couleur, alpha=0.75, label=nom, lw=0)
        bas = bas + y
    ax.set_xlabel("itération")
    ax.set_ylabel("part des parties (%)")
    ax.set_title("Origine des nulles")
    ax.set_ylim(0, 100)
    ax.grid(True, color="#DDDDDD", lw=0.5)
    ax.legend(loc="upper right", ncol=2, fontsize=6.5)

    fig.tight_layout()
    fig.savefig(SORTIE / "fig_selfplay_qualite.pdf")
    plt.close(fig)


def fig_elo(sp):
    it = sp["iteration"]
    elo = sp["elo"]
    wr = sp["winrate"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    ax = axes[0]
    ax.scatter(it, elo, s=10, color=BLEU, alpha=0.55, label="évaluation (16 parties)")
    ax.plot(it, mediane_glissante(it, elo, 21), color=BLEU, lw=1.5, label="médiane glissante")
    for niveau, x0, x1 in [(2200, 1, 37), (2300, 38, 96), (2450, 97, 236), (2600, 237, 436)]:
        ax.hlines(niveau, x0, x1, color=ROUGE, lw=1.0, ls=":")
    ax.annotate("Stockfish 2200", xy=(4, 2200), xytext=(6, 2240), fontsize=7, color=ROUGE)
    ax.annotate("2450", xy=(150, 2450), xytext=(128, 2480), fontsize=7, color=ROUGE)
    ax.annotate("2600", xy=(330, 2600), xytext=(305, 2630), fontsize=7, color=ROUGE)
    ax.set_ylim(2100, 3000)
    ax.set_xlabel("itération")
    ax.set_ylabel("Elo estimé du modèle")
    ax.set_title("Niveau face à Stockfish (200k nœuds)")
    ax.grid(True, color="#DDDDDD", lw=0.5)
    ax.legend(loc="lower right")

    ax = axes[1]
    ax.scatter(it, 100 * wr, s=10, color=VERT, alpha=0.55)
    ax.plot(it, 100 * mediane_glissante(it, wr, 21), color=VERT, lw=1.5)
    ax.axhline(50, color=GRIS, lw=0.8, ls="--")
    ax.set_xlabel("itération")
    ax.set_ylabel("score contre Stockfish (%)")
    ax.set_title("Score face à l'ancre")
    ax.grid(True, color="#DDDDDD", lw=0.5)

    fig.tight_layout()
    fig.savefig(SORTIE / "fig_elo.pdf")
    plt.close(fig)


def fig_timeline():
    fig, ax = plt.subplots(figsize=(7.4, 2.6))

    def d(texte):
        return dt.datetime.strptime(texte, "%Y-%m-%d")

    barres = [
        (0, d("2026-03-07"), d("2026-03-11"), BLEU_CLAIR, "gén. 1 : supervisé (GM puis Lichess)"),
        (1, d("2026-03-10"), d("2026-04-04"), VERT_CLAIR, "gén. 1 : self-play, 431 itérations"),
        (2, d("2026-04-06"), d("2026-04-06") + dt.timedelta(hours=3), BLEU, "prétraining Lichess (2 h 50)"),
        (2, d("2026-04-06") + dt.timedelta(hours=22), d("2026-04-07") + dt.timedelta(hours=1), AMBRE, "affinage GM (1 h 33)"),
        (3, d("2026-04-07"), d("2026-04-30"), VERT, "self-play, 436 itérations (23 jours)"),
    ]
    for ligne, debut, fin, couleur, nom in barres:
        ax.broken_barh([(debut, fin - debut)], (ligne - 0.3, 0.6),
                       facecolors=couleur, edgecolor="white", lw=0.8, label=nom)

    evenements = [
        ("2026-04-13", "GPU"),
        ("2026-04-14", "threads"),
        ("2026-04-15", "shards, x14"),
        ("2026-04-23", "puzzles"),
        ("2026-04-27", "amnésie"),
        ("2026-04-30", "iter436"),
    ]
    for i, (date, nom) in enumerate(evenements):
        x = d(date)
        ax.axvline(x, color=GRIS, lw=0.7, ls=":", alpha=0.8)
        ax.annotate(nom, xy=(x, 4.05 + 0.75 * (i % 3)), rotation=28, fontsize=6.5, color=GRIS,
                    ha="left", va="bottom")

    ax.axvline(d("2026-04-06"), color=ROUGE, lw=1.2)
    ax.annotate("changement d'architecture (SE)\n+ reprise de zéro", xy=(d("2026-04-06"), 1.55),
                xytext=(d("2026-03-16"), 1.6), fontsize=7, color=ROUGE,
                arrowprops={"arrowstyle": "->", "color": ROUGE, "lw": 0.8})

    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["supervisé gén. 1", "self-play gén. 1", "supervisé gén. 2", "self-play gén. 2"],
                       fontsize=8)
    ax.set_ylim(-0.6, 6.5)
    ax.set_xlim(d("2026-03-05"), d("2026-05-03"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.grid(True, axis="x", color="#DDDDDD", lw=0.5)
    ax.set_title("Chronologie du projet (mars et avril 2026)")
    fig.tight_layout()
    fig.savefig(SORTIE / "fig_chronologie.pdf")
    plt.close(fig)


def fig_volumes():
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.6))
    phases = ["Lichess", "grands maîtres", "self-play"]
    parties = [1_200_000, 210_000, 211_200]
    positions_uniques = [39.9e6, 19.45e6, 5.93e6]
    positions_vues = [79.8e6, 38.9e6, 105.9e6]

    ax = axes[0]
    barres = ax.barh(phases, parties, color=[BLEU, AMBRE, VERT], height=0.55)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(5e4, 5e6)
    ax.bar_label(barres, labels=["1,2 M", "210 k", "211 k"], padding=3, fontsize=8)
    ax.set_xlabel("parties (échelle log)")
    ax.set_title("Parties impliquées")
    ax.grid(True, axis="x", which="both", color="#DDDDDD", lw=0.5)

    ax = axes[1]
    largeur = 0.35
    y = np.arange(len(phases))
    b1 = ax.barh(y + largeur / 2, positions_uniques, height=largeur,
                 color=[BLEU_CLAIR, AMBRE, VERT_CLAIR], label="uniques")
    b2 = ax.barh(y - largeur / 2, positions_vues, height=largeur,
                 color=[BLEU, AMBRE, VERT], label="vues à l'entraînement")
    ax.invert_yaxis()
    ax.set_yticks(y)
    ax.set_yticklabels(phases)
    ax.set_xscale("log")
    ax.set_xlim(1e6, 3e8)
    ax.bar_label(b1, labels=["39,9 M", "19,5 M", "5,9 M"], padding=3, fontsize=7)
    ax.bar_label(b2, labels=["79,8 M", "38,9 M", "105,9 M"], padding=3, fontsize=7)
    ax.set_xlabel("positions (échelle log)")
    ax.set_title("Positions uniques et vues")
    ax.legend(loc="upper center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, -0.25))
    ax.grid(True, axis="x", which="both", color="#DDDDDD", lw=0.5)

    fig.tight_layout()
    fig.savefig(SORTIE / "fig_volumes.pdf")
    plt.close(fig)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(RACINE / "python_src"))
    sup = lire_supervise("jhqhhx5q")
    gm = lire_supervise("ifvh81l0")
    sp = lire_selfplay()

    fig_supervise_lichess(sup)
    fig_supervise_gm(gm)
    fig_transfert()
    fig_selfplay_pertes(sp)
    fig_selfplay_volume(sp)
    fig_selfplay_qualite(sp)
    fig_elo(sp)
    fig_timeline()
    fig_volumes()
    print("figures generees dans", SORTIE)
