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
    sims_par_seconde,
    taux_table,
)


def _m(position="depart", chemin="mcts_search", simulations=400, duree_s=1.0,
       nn_calls=300, tt_hits=100, tt_misses=300, terminal_hits=0):
    return Mesure(position=position, chemin=chemin, simulations=simulations,
                  duree_s=duree_s, nn_calls=nn_calls, tt_hits=tt_hits,
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


def test_agreger_groupe_par_position_et_chemin():
    mesures = [
        _m(position="depart", chemin="mcts_search", duree_s=1.0),
        _m(position="depart", chemin="mcts_search", duree_s=2.0),
        _m(position="depart", chemin="step_analysis", duree_s=1.0),
        _m(position="finale", chemin="mcts_search", duree_s=1.0),
    ]

    agr = agreger(mesures)

    assert set(agr) == {("depart", "mcts_search"), ("depart", "step_analysis"),
                        ("finale", "mcts_search")}
    assert agr[("depart", "mcts_search")]["passages"] == 2


def test_agreger_donne_mediane_et_etendue():
    """L'etendue est la raison d'etre des passages repetes : sans elle, on ne
    sait pas distinguer un gain de 5 pour cent d'un bruit de mesure."""
    mesures = [_m(duree_s=1.0), _m(duree_s=2.0), _m(duree_s=4.0)]

    a = agreger(mesures)[("depart", "mcts_search")]

    assert a["sims_par_seconde_median"] == pytest.approx(200.0)
    assert a["sims_par_seconde_min"] == pytest.approx(100.0)
    assert a["sims_par_seconde_max"] == pytest.approx(400.0)


def test_agreger_supporte_une_liste_vide():
    assert agreger([]) == {}
