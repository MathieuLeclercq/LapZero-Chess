"""Evaluation recurrente du banc de positions : policy, value puis MCTS.

Le premier passage charge le dataset par lots de 1024 : chaque position est
rejouee dans le C++, verifiee (FEN, identite reseau, coups legaux et
etiquettes), puis le lot part dans une session ONNX GPU unique. Les
probabilites sont calculees par un softmax masque aux seuls coups legaux et
jointes aux labels Stockfish pre-calcules.

Le sous-banc MCTS parcourt 256 identifiants dans l'ordre du manifeste, sur un
seul objet MCTS et un seul evaluateur persistants : racine neuve, table de
transposition froide et compteurs remis a zero a chaque position. Aucun appel
n'est decoupe : chaque position vaut exactement le budget demande.

Aucune adaptation au temps ecoule : une cible de duree depassee est publiee
telle quelle, sans reduire les budgets.

Voir docs/superpowers/specs/2026-09-29-position-bench-design.md (section 7) et
docs/superpowers/plans/2026-09-30-position-bench.md (tache 6).
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import cast

import numpy as np
from position_bench_metrics import (
    TAILLE_POLICY,
    EvalConfig,
    RawResult,
)
from position_bench_sources import position_identity, replay_position

TOLERANCE_VALEUR = 1e-6


def _preparer_plateau(record: Mapping[str, object], exiger_labels: bool = True):
    """Rejoue et verifie une position avant toute inference.

    Le FEN, l'identite reseau, la liste triee des coups legaux et la
    couverture des etiquettes doivent correspondre exactement : une donnee
    incoherente doit faire echouer le passage, pas produire une mesure
    approximative.
    """
    position_id = str(record["position_id"])
    board = replay_position(record)
    if board.to_fen() != record["fen"]:
        raise ValueError(f"position {position_id} : FEN rejouee differente")
    if position_identity(board) != position_id:
        raise ValueError(f"position {position_id} : identite reseau differente")
    indices = sorted(board.get_legal_move_indices())
    if indices != list(cast(list[int], record["legal_indices"])):
        raise ValueError(
            f"position {position_id} : coups legaux differents de "
            "l'enregistrement")
    if exiger_labels:
        labels = cast(list, record["labels"])
        if [label["index"] for label in labels] != indices:
            raise ValueError(
                f"position {position_id} : etiquettes incompletes ou "
                "desordonnees")
    return board, indices


def _valider_sorties(logits, values, lignes: int) -> None:
    if logits.shape != (lignes, TAILLE_POLICY) or values.shape != (lignes,):
        raise RuntimeError(
            f"evaluateur : sorties de forme {logits.shape} et {values.shape} "
            f"pour {lignes} lignes")
    if not (np.all(np.isfinite(logits)) and np.all(np.isfinite(values))):
        raise RuntimeError("evaluateur : sortie non finie")
    if np.any(values < -1.0 - TOLERANCE_VALEUR) \
            or np.any(values > 1.0 + TOLERANCE_VALEUR):
        raise RuntimeError(
            "evaluateur : value hors de [-1, 1] ; sortie refusee, jamais "
            "ecretee")


def _probabilites_masquees(logits_ligne, indices: list[int]) -> np.ndarray:
    """Softmax limite aux coups legaux, en float64 puis stocke en float32."""
    choisis = np.asarray(logits_ligne[indices], dtype=np.float64)
    choisis -= choisis.max()
    exponentielles = np.exp(choisis)
    return (exponentielles / exponentielles.sum()).astype(np.float32)


def _compteurs_dict(compteurs) -> dict:
    return {
        "completed_simulations": int(compteurs.completed_simulations),
        "nn_calls": int(compteurs.nn_calls),
        "nn_batches": int(compteurs.nn_batches),
        "tt_hits": int(compteurs.tt_hits),
        "tt_misses": int(compteurs.tt_misses),
        "leaf_collisions": int(compteurs.leaf_collisions),
    }


def _passe_policy(records, evaluator, config: EvalConfig, clock,
                  exiger_labels: bool) -> dict:
    """Parcours direct par lots : un seul lot de tenseurs vit a la fois."""
    offsets = [0]
    indices_plats: list[int] = []
    probas_plates: list[float] = []
    valeurs: list[float] = []
    temps_rejeu = 0.0
    temps_policy = 0.0

    for debut in range(0, len(records), config.policy_batch_size):
        lot = records[debut:debut + config.policy_batch_size]
        marque = clock()
        plateaux = []
        legaux_lot = []
        for record in lot:
            plateau, indices = _preparer_plateau(record, exiger_labels)
            plateaux.append(np.asarray(plateau.get_alphazero_tensor(),
                                       dtype=np.float32))
            legaux_lot.append(indices)
        temps_rejeu += clock() - marque

        marque = clock()
        logits, values = evaluator.predict_batch(np.stack(plateaux))
        temps_policy += clock() - marque
        _valider_sorties(logits, values, len(lot))

        for ligne, indices in zip(logits, legaux_lot, strict=True):
            probas_plates.extend(_probabilites_masquees(ligne, indices))
            indices_plats.extend(indices)
            offsets.append(len(indices_plats))
        valeurs.extend(float(valeur) for valeur in values)

    return {
        "offsets": offsets,
        "indices": indices_plats,
        "probas": probas_plates,
        "valeurs": valeurs,
        "replay_s": temps_rejeu,
        "policy_s": temps_policy,
    }


def _passe_recherche(records, search_ids, evaluator, config: EvalConfig,
                     mcts_factory, clock) -> dict:
    """Un seul MCTS pour tout le sous-banc, racine et table froides."""
    reglages = config.search
    par_id = {str(record["position_id"]): record for record in records}

    if not search_ids:
        return {
            "offsets": [0], "indices": [], "probas": [], "compteurs": [],
            "setup_s": 0.0, "replay_s": 0.0, "mcts_s": 0.0, "durees": [],
        }

    marque = clock()
    mcts = mcts_factory(evaluator, reglages.tt_size,
                        reglages.cache_history_depth)
    mcts.set_fixed_batch(reglages.fixed_batch)
    mcts.set_tuning(reglages.virtual_loss, reglages.fpu,
                    reglages.collision_attempts)
    setup_s = clock() - marque

    offsets = [0]
    indices_plats: list[int] = []
    probas_plates: list[float] = []
    compteurs_liste: list[dict] = []
    durees: list[float] = []
    temps_rejeu = 0.0
    temps_mcts = 0.0

    for position_id in search_ids:
        record = par_id.get(str(position_id))
        if record is None:
            raise ValueError(f"recherche : position inconnue {position_id}")
        marque = clock()
        plateau = replay_position(record)
        temps_rejeu += clock() - marque

        mcts.clear_evaluation_cache()
        mcts.reset_counters()
        marque = clock()
        pi = mcts.mcts_search(plateau, reglages.simulations,
                              reglages.c_puct, False,
                              reglages.batch_size, reglages.workers)
        duree = clock() - marque
        temps_mcts += duree
        durees.append(duree)

        compteurs = _compteurs_dict(mcts.get_counters())
        if compteurs["completed_simulations"] != reglages.simulations:
            raise ValueError(
                f"recherche {position_id} : "
                f"{compteurs['completed_simulations']} simulations sur "
                f"{reglages.simulations}")
        compteurs_liste.append(compteurs)

        indices = list(cast(list[int], record["legal_indices"]))
        visites = np.asarray([pi[index] for index in indices],
                             dtype=np.float64)
        total = float(visites.sum())
        if total <= 0.0:
            raise ValueError(f"recherche {position_id} : aucune visite")
        indices_plats.extend(indices)
        probas_plates.extend((visites / total).astype(np.float32))
        offsets.append(len(indices_plats))

    return {
        "offsets": offsets,
        "indices": indices_plats,
        "probas": probas_plates,
        "compteurs": compteurs_liste,
        "setup_s": setup_s,
        "replay_s": temps_rejeu,
        "mcts_s": temps_mcts,
        "durees": durees,
    }


def evaluate_positions(records, search_ids, evaluator, config: EvalConfig, *,
                       mcts_factory,
                       clock=time.perf_counter) -> RawResult:
    """Policy/value sur toutes les positions, puis MCTS sur le sous-banc.

    `records` est dans l'ordre du manifeste et porte les annotations ;
    `search_ids` fixe l'ordre des recherches. `mcts_factory(evaluateur,
    tt_size, cache_history_depth)` est injecte pour les tests.
    """
    debut_total = clock()
    position_ids = [str(record["position_id"]) for record in records]
    policy = _passe_policy(records, evaluator, config, clock, True)
    recherche = _passe_recherche(records, search_ids, evaluator, config,
                                 mcts_factory, clock)

    resultat: dict = {
        "position_ids": np.asarray(position_ids, dtype="<U64"),
        "legal_offsets": np.asarray(policy["offsets"], dtype=np.int64),
        "legal_indices": np.asarray(policy["indices"], dtype=np.int32),
        "policy_probs": np.asarray(policy["probas"], dtype=np.float32),
        "values": np.asarray(policy["valeurs"], dtype=np.float32),
        "search_ids": np.asarray(search_ids, dtype="<U64"),
        "search_offsets": np.asarray(recherche["offsets"], dtype=np.int64),
        "search_indices": np.asarray(recherche["indices"], dtype=np.int32),
        "search_probs": np.asarray(recherche["probas"], dtype=np.float32),
        "search_counters": recherche["compteurs"],
        "timings": {
            "replay_s": policy["replay_s"] + recherche["replay_s"],
            "policy_s": policy["policy_s"],
            "mcts_setup_s": recherche["setup_s"],
            "mcts_s": recherche["mcts_s"],
            "total_s": clock() - debut_total,
            "search_position_s": recherche["durees"],
        },
    }
    return cast(RawResult, resultat)


def profile_positions(records, evaluator, config: EvalConfig, *,
                      mcts_factory, search_count: int | None = None,
                      clock=time.perf_counter) -> dict:
    """Mesure le parcours direct et la recherche sur des positions brutes.

    Mode `profile` : aucune annotation requise, aucun score, aucune version du
    banc. Sert a estimer le cout avant la longue annotation reelle.
    """
    debut = clock()
    ids = [str(record["position_id"]) for record in records]
    if search_count is None:
        retenus = ids[:32] if len(ids) > 32 else ids
    else:
        retenus = ids[:search_count]
    policy = _passe_policy(records, evaluator, config, clock, False)
    recherche = _passe_recherche(records, retenus, evaluator, config,
                                 mcts_factory, clock)
    return {
        "mode": "profile",
        "count": len(records),
        "searches": len(retenus),
        "timings": {
            "replay_s": policy["replay_s"] + recherche["replay_s"],
            "policy_s": policy["policy_s"],
            "mcts_setup_s": recherche["setup_s"],
            "mcts_s": recherche["mcts_s"],
            "total_s": clock() - debut,
            "search_position_s": recherche["durees"],
        },
    }
