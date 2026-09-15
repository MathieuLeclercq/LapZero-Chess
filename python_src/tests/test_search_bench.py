"""Logique pure du harnais de debit. Aucun moteur, aucun modele."""
import os
import sys
from pathlib import Path

import pytest

RACINE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, RACINE)
os.add_dll_directory(RACINE)

from search_bench import (
    Mesure,
    agreger,
    inferences_par_seconde,
    remplissage_moyen,
    sims_par_seconde,
    taux_table,
)


def _m(position="depart", chemin="mcts_search", simulations=400, duree_s=1.0,
       nn_calls=300, nn_batches=300, tt_hits=100, tt_misses=300,
       terminal_hits=0, batch_size=0, cache_history_depth=1,
       tt_position_matches=100, tt_rule50_rejects=0,
       tt_context_rejects=0, tt_history_rejects=0):
    return Mesure(position=position, chemin=chemin, simulations=simulations,
                  batch_size=batch_size, duree_s=duree_s, nn_calls=nn_calls,
                  nn_batches=nn_batches, tt_hits=tt_hits,
                  tt_misses=tt_misses, terminal_hits=terminal_hits,
                  cache_history_depth=cache_history_depth,
                  tt_position_matches=tt_position_matches,
                  tt_rule50_rejects=tt_rule50_rejects,
                  tt_context_rejects=tt_context_rejects,
                  tt_history_rejects=tt_history_rejects)


def test_les_trois_grandeurs_sont_distinctes():
    """Une simulation n'est pas une inference : elle peut s'arreter sur un
    noeud terminal ou sur un succes de table. Mesure sur une position de mat,
    381 simulations sur 400 s'arretaient sans jamais appeler le reseau."""
    m = _m(simulations=400, duree_s=2.0, nn_calls=300, tt_hits=100, tt_misses=300)

    assert sims_par_seconde(m) == pytest.approx(200.0)
    assert inferences_par_seconde(m) == pytest.approx(150.0)
    assert taux_table(m) == pytest.approx(0.25)


def test_taux_table_sans_consultation_vaut_zero():
    assert taux_table(_m(tt_hits=0, tt_misses=0)) == 0.0


def test_duree_nulle_ne_divise_pas_par_zero():
    m = _m(duree_s=0.0)

    assert sims_par_seconde(m) == 0.0
    assert inferences_par_seconde(m) == 0.0


def test_remplissage_moyen_distingue_positions_et_appels_batch():
    assert remplissage_moyen(_m(nn_calls=320, nn_batches=10)) == pytest.approx(32.0)
    assert remplissage_moyen(_m(nn_calls=0, nn_batches=0)) == 0.0


def test_agreger_groupe_par_position_et_chemin():
    mesures = [
        _m(position="depart", chemin="mcts_search", duree_s=1.0),
        _m(position="depart", chemin="mcts_search", duree_s=2.0),
        _m(position="depart", chemin="step_analysis", duree_s=1.0),
        _m(position="finale", chemin="mcts_search", duree_s=1.0),
    ]

    agr = agreger(mesures)

    assert set(agr) == {("depart", "mcts_search", 0, 1),
                        ("depart", "step_analysis", 0, 1),
                        ("finale", "mcts_search", 0, 1)}
    assert agr[("depart", "mcts_search", 0, 1)]["passages"] == 2


def test_agreger_groupe_aussi_par_taille_de_batch():
    mesures = [
        _m(batch_size=0, duree_s=4.0),
        _m(batch_size=32, duree_s=1.0),
        _m(batch_size=32, duree_s=1.0),
    ]

    agr = agreger(mesures)

    assert set(agr) == {("depart", "mcts_search", 0, 1),
                        ("depart", "mcts_search", 32, 1)}
    assert agr[("depart", "mcts_search", 32, 1)]["passages"] == 2
    assert agr[("depart", "mcts_search", 32, 1)]["sims_par_seconde_median"] == pytest.approx(400.0)


def test_agreger_separe_les_profondeurs_de_cache():
    agr = agreger([
        _m(cache_history_depth=0),
        _m(cache_history_depth=1),
    ])

    assert set(agr) == {
        ("depart", "mcts_search", 0, 0),
        ("depart", "mcts_search", 0, 1),
    }


def test_agreger_prend_la_mediane_des_ratios_et_non_le_ratio_des_medianes():
    agr = agreger([
        _m(tt_hits=10, tt_misses=10, tt_position_matches=20,
           tt_rule50_rejects=10),
        _m(tt_hits=0, tt_misses=100, tt_position_matches=10,
           tt_history_rejects=10),
    ])[("depart", "mcts_search", 0, 1)]

    assert agr["tt_position_match_rate_median"] == pytest.approx(0.55)
    assert agr["tt_rule50_reject_rate_median"] == pytest.approx(0.25)
    assert agr["tt_context_reject_rate_median"] == 0.0
    assert agr["tt_history_reject_rate_median"] == pytest.approx(0.05)


def test_agreger_donne_mediane_et_etendue():
    """L'etendue est la raison d'etre des passages repetes : sans elle, on ne
    sait pas distinguer un gain de 5 pour cent d'un bruit de mesure."""
    mesures = [_m(duree_s=1.0), _m(duree_s=2.0), _m(duree_s=4.0)]

    a = agreger(mesures)[("depart", "mcts_search", 0, 1)]

    assert a["sims_par_seconde_median"] == pytest.approx(200.0)
    assert a["sims_par_seconde_min"] == pytest.approx(100.0)
    assert a["sims_par_seconde_max"] == pytest.approx(400.0)


def test_agreger_supporte_une_liste_vide():
    assert agreger([]) == {}


from search_bench import format_report

CONTEXTE = {
    "modele": "iter316.onnx",
    "iteration": 316,
    "global_step": 19415,
    "passages": 5,
    "simulations": 400,
    "c_puct": 1.4,
}


def _agr():
    return agreger([_m(duree_s=1.0), _m(duree_s=1.1)])


def test_format_report_contient_le_contexte():
    texte = format_report(_agr(), CONTEXTE, [])

    assert "iter316.onnx" in texte
    assert "19415" in texte
    assert "simulations par seconde" in texte


def test_format_report_affiche_batch_et_remplissage():
    texte = format_report(
        agreger([_m(batch_size=32, nn_calls=320, nn_batches=10)]),
        CONTEXTE, [])

    assert "batch" in texte.lower()
    assert "32.0" in texte


def test_rapport_affiche_les_rejets_semantiques():
    texte = format_report(agreger([_m(
        cache_history_depth=1,
        tt_hits=10,
        tt_misses=20,
        tt_position_matches=20,
        tt_rule50_rejects=3,
        tt_context_rejects=2,
        tt_history_rejects=5,
    )]), CONTEXTE, [])

    assert "h1" in texte
    assert "match position" in texte
    assert "50 coups" in texte
    assert "historique" in texte
    assert "66.7 %" in texte
    assert "10.0 %" in texte
    assert "6.7 %" in texte
    assert "16.7 %" in texte


def test_les_mesures_transmettent_la_profondeur_au_mcts(monkeypatch):
    import search_bench

    constructions = []

    class FauxMCTS:
        def __init__(self, evaluateur, taille_tt, depth):
            constructions.append(depth)

        def reset_analysis(self):
            pass

        def reset_counters(self):
            pass

        def mcts_search(self, *args):
            pass

        def step_analysis(self, *args):
            pass

        def get_counters(self):
            return type("C", (), dict(
                nn_calls=1, nn_batches=1, tt_hits=0, tt_misses=1,
                terminal_hits=0, tt_position_matches=0,
                tt_rule50_rejects=0, tt_context_rejects=0,
                tt_history_rejects=0))()

    monkeypatch.setattr(search_bench.chess_engine, "MCTS", FauxMCTS)
    monkeypatch.setattr(search_bench, "charger_position", lambda fen: object())

    search_bench.mesurer_mcts_search(
        object(), "fen", "position", 8, cache_history_depth=3)
    search_bench.mesurer_step_analysis(
        object(), "fen", "position", 8, cache_history_depth=7)

    assert constructions == [3, 7]


def test_main_balaie_toutes_les_profondeurs_demandees(
        monkeypatch, tmp_path):
    import search_bench
    import puzzle_bench

    model = tmp_path / "model.onnx"
    model.write_bytes(b"modele factice")
    rapport = tmp_path / "rapport.md"
    appels = []

    class Chauffe:
        def __init__(self, evaluateur, taille_tt, depth):
            appels.append(("chauffe", depth))

        def mcts_search(self, *args):
            pass

    def mesurer_recherche(*args):
        depth = args[-1]
        appels.append(("mcts_search", depth))
        return _m(cache_history_depth=depth)

    def mesurer_analyse(*args):
        depth = args[-1]
        appels.append(("step_analysis", depth))
        return _m(chemin="step_analysis", cache_history_depth=depth)

    monkeypatch.setattr(search_bench, "POSITIONS", (("depart", "fen"),))
    monkeypatch.setattr(search_bench.chess_engine, "MCTS", Chauffe)
    monkeypatch.setattr(search_bench.chess_engine, "ONNXEvaluator",
                        lambda *args: object())
    monkeypatch.setattr(search_bench, "charger_position", lambda fen: object())
    monkeypatch.setattr(search_bench, "mesurer_mcts_search", mesurer_recherche)
    monkeypatch.setattr(search_bench, "mesurer_step_analysis", mesurer_analyse)
    monkeypatch.setattr(puzzle_bench, "resoudre_modele",
                        lambda *args: (model, {}))
    monkeypatch.setattr(sys, "argv", [
        "search_bench.py", "--model", str(model), "--passages", "1",
        "--simulations", "8", "--cache-history-depths", "0", "3",
        "--out-rapport", str(rapport),
    ])

    assert search_bench.main() == 0
    assert appels == [
        ("chauffe", 0),
        ("mcts_search", 0), ("step_analysis", 0),
        ("mcts_search", 3), ("step_analysis", 3),
    ]


def test_format_report_n_ecrit_jamais_noeuds_par_seconde():
    """Le perft mesure 1,6 million de noeuds par seconde, la recherche 295
    simulations par seconde : confondre les deux serait une erreur d'un facteur
    5000."""
    texte = format_report(_agr(), CONTEXTE, []).lower()

    assert "noeuds par seconde" not in texte
    assert "nps" not in texte


def test_format_report_ne_contient_pas_de_tiret_cadratin():
    assert "—" not in format_report(_agr(), CONTEXTE, [])


def test_format_report_signale_les_violations():
    invariants = [("depart", 1, 0, 120, 5, 3,
                   ["enfant duplique, move_idx 42"])]

    texte = format_report(_agr(), CONTEXTE, invariants)

    assert "enfant duplique" in texte
    assert "3" in texte


def test_format_report_associe_les_invariants_a_leur_batch():
    invariants = [("depart", 1, 32, 120, 5, 0, [])]

    texte = format_report(_agr(), CONTEXTE, invariants)

    assert "| depart | h1 | 32 | 120 | 5 | 0 |" in texte


def test_format_report_est_muet_quand_aucune_violation():
    invariants = [("depart", 1, 0, 120, 5, 0, [])]

    texte = format_report(_agr(), CONTEXTE, invariants)

    assert "enfant duplique" not in texte
    assert "aucune violation" in texte.lower()


def test_format_report_produit_des_tables_markdown_valides():
    """Garde fou repris du banc de puzzles : un separateur dont le nombre de
    cellules ne correspond pas a l'en-tete casse le rendu en silence."""
    lignes = format_report(_agr(), CONTEXTE, []).split("\n")
    separateurs = [i for i, l in enumerate(lignes)
                   if set(l) <= set("|-") and "-" in l]

    assert separateurs
    for i in separateurs:
        cols = lignes[i].count("|") - 1
        assert lignes[i - 1].count("|") - 1 == cols, lignes[i - 1]
        j = i + 1
        while j < len(lignes) and lignes[j].startswith("|"):
            assert lignes[j].count("|") - 1 == cols, lignes[j]
            j += 1
