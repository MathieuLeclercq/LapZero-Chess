"""Logique pure du harnais de debit. Aucun moteur, aucun modele."""
import os
import sys

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
       terminal_hits=0, batch_size=0):
    return Mesure(position=position, chemin=chemin, simulations=simulations,
                  batch_size=batch_size, duree_s=duree_s, nn_calls=nn_calls,
                  nn_batches=nn_batches, tt_hits=tt_hits,
                  tt_misses=tt_misses, terminal_hits=terminal_hits)


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

    assert set(agr) == {("depart", "mcts_search", 0),
                        ("depart", "step_analysis", 0),
                        ("finale", "mcts_search", 0)}
    assert agr[("depart", "mcts_search", 0)]["passages"] == 2


def test_agreger_groupe_aussi_par_taille_de_batch():
    mesures = [
        _m(batch_size=0, duree_s=4.0),
        _m(batch_size=32, duree_s=1.0),
        _m(batch_size=32, duree_s=1.0),
    ]

    agr = agreger(mesures)

    assert set(agr) == {("depart", "mcts_search", 0),
                        ("depart", "mcts_search", 32)}
    assert agr[("depart", "mcts_search", 32)]["passages"] == 2
    assert agr[("depart", "mcts_search", 32)]["sims_par_seconde_median"] == pytest.approx(400.0)


def test_agreger_donne_mediane_et_etendue():
    """L'etendue est la raison d'etre des passages repetes : sans elle, on ne
    sait pas distinguer un gain de 5 pour cent d'un bruit de mesure."""
    mesures = [_m(duree_s=1.0), _m(duree_s=2.0), _m(duree_s=4.0)]

    a = agreger(mesures)[("depart", "mcts_search", 0)]

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
    invariants = [("depart", 0, 120, 5, 3, ["enfant duplique, move_idx 42"])]

    texte = format_report(_agr(), CONTEXTE, invariants)

    assert "enfant duplique" in texte
    assert "3" in texte


def test_format_report_associe_les_invariants_a_leur_batch():
    invariants = [("depart", 32, 120, 5, 0, [])]

    texte = format_report(_agr(), CONTEXTE, invariants)

    assert "| depart | 32 | 120 | 5 | 0 |" in texte


def test_format_report_est_muet_quand_aucune_violation():
    invariants = [("depart", 0, 120, 5, 0, [])]

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
