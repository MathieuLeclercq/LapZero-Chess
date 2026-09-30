"""Tests de l'integration du banc de positions a la boucle self-play.

Generation, entrainement, export, sauvegarde, evaluation et W&B sont des
doubles : aucun GPU, aucun modele, aucun dataset externe.
"""
import os
import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE / "python_src"))

import run_selftrain
import train_self_play
from train_self_play import pipeline


class JournalWandb:
    def __init__(self):
        self.initialisations = []
        self.points = []
        self.fins = 0

    def init(self, **kwargs):
        self.initialisations.append(kwargs)

    def log(self, donnees, **kwargs):
        self.points.append((donnees, kwargs))

    def finish(self):
        self.fins += 1


class Modele:
    def to(self, device):
        return self

    def parameters(self):
        return []

    def load_state_dict(self, state):
        pass

    def state_dict(self):
        return {}


class Optimiseur:
    def __init__(self):
        self.param_groups = [{"lr": 0.0}]

    def load_state_dict(self, state):
        pass

    def state_dict(self):
        return {}


class Scaler:
    def load_state_dict(self, state):
        pass

    def state_dict(self):
        return {}


def _appels_vides():
    return {cle: [] for cle in (
        "generation", "entrainement", "export", "sauvegarde", "evaluation")}


def _dependances(appels, journal):
    def generer(onnx, games, concurrent, slow, fast, ratio, tt_size=0,
                diagnostics=0):
        appels["generation"].append({"onnx": str(onnx), "games": games})
        stats = {"checkmates": 2, "stalemates": 0, "repetition": 0,
                 "50_moves": 0, "insuff_mat": 0, "max_moves": 0}
        return [], 100.0, stats, None

    def entrainer(model, optimizer, scaler, device, folder, lr, **kwargs):
        appels["entrainement"].append(dict(kwargs))
        return kwargs["global_step"] + 1

    def exporter(model, chemin, device):
        appels["export"].append(str(chemin))

    def sauver(chemin, charge):
        appels["sauvegarde"].append((str(chemin), charge["iteration"]))

    def evaluer(onnx, banc, sortie, config, **kwargs):
        appels["evaluation"].append({
            "onnx": str(onnx), "banc": str(banc), "config": config,
            **kwargs})
        return {
            "status": "ok", "duration_s": 310.0,
            "completed_positions": 10000, "completed_search_positions": 256,
            "dataset_version": "v1", "protocol_id": "p" * 64,
            "dataset_sha256": "d" * 64, "model_sha256": "m" * 64,
            "metrics": {}, "comparison": None, "result_path": "x",
            "error": None, "target_s": 300.0, "over_target": True,
        }

    def charger_banc(chemin):
        assert Path(chemin).name == "banc"
        return ({"dataset_sha256": "d" * 64, "dataset_version": "v1",
                 "search_ids": [], "counts": {}}, [])

    def ecrire_buffer(new_data, folder, plafond):
        return 0

    def charger_checkpoint(chemin):
        return {"model_state_dict": {}, "optimizer_state_dict": {},
                "scaler_state_dict": {}, "iteration": 3, "global_step": 100}

    return {
        "generer_parties": generer,
        "entrainer": entrainer,
        "exporter": exporter,
        "sauver_checkpoint": sauver,
        "evaluer": evaluer,
        "charger_banc": charger_banc,
        "ecrire_buffer": ecrire_buffer,
        "charger_checkpoint": charger_checkpoint,
        "creer_modele": lambda: Modele(),
        "creer_optimiseur": lambda parametres: Optimiseur(),
        "creer_scaler": lambda: Scaler(),
        "cuda_disponible": lambda: True,
        "chercher_precedent": lambda *arguments: None,
        "protocol_id": lambda config: "p" * 64,
        "wandb": journal,
    }


def test_cadence_globale_apres_reprise():
    iterations = range(3, 9)
    evaluated = [i + 1 for i in iterations if (i + 1) % 4 == 0]

    assert evaluated == [4, 8]


def test_pipeline_evalue_aux_iterations_4_et_8(tmp_path):
    appels = _appels_vides()
    journal = JournalWandb()

    pipeline(num_iterations=6, checkpoint_path="faux.pt", eval_every=4,
             position_bench_path=tmp_path / "banc",
             eval_output_dir=tmp_path / "resultats",
             dependances=_dependances(appels, journal))

    assert [appel["iteration"] for appel in appels["evaluation"]] == [4, 8]
    assert [iteration for _, iteration in appels["sauvegarde"]] == [
        4, 5, 6, 7, 8, 9]
    assert appels["evaluation"][0]["onnx"].endswith(
        "iter4_unsupervised.onnx")
    assert appels["evaluation"][1]["onnx"].endswith(
        "iter8_unsupervised.onnx")
    assert appels["evaluation"][0]["config"].search.workers == 8
    assert appels["evaluation"][0]["config"].target_s == 300.0
    # Un seul log par iteration, et il porte le statut du banc les bonnes
    # iterations.
    points = [donnees for donnees, _ in journal.points if donnees]
    assert any("eval/position/status" in point for point in points)
    assert len(appels["generation"]) == 6
    assert len(appels["entrainement"]) == 6


def test_le_precedent_compatible_est_passe_a_l_evaluation(tmp_path):
    appels = _appels_vides()
    dependances = _dependances(appels, JournalWandb())
    dependances["chercher_precedent"] = (
        lambda *arguments: tmp_path / "precedent.json")

    pipeline(num_iterations=4, eval_every=4,
             position_bench_path=tmp_path / "banc",
             eval_output_dir=tmp_path / "resultats", dependances=dependances)

    assert len(appels["evaluation"]) == 1
    assert Path(appels["evaluation"][0]["previous"]).name == "precedent.json"


def test_eval_every_zero_desactive_le_banc():
    appels = _appels_vides()
    dependances = _dependances(appels, JournalWandb())

    def charger_interdit(chemin):
        raise AssertionError("le banc ne doit pas etre charge")

    dependances["charger_banc"] = charger_interdit

    pipeline(num_iterations=1, eval_every=0, dependances=dependances)

    assert appels["evaluation"] == []


def test_eval_every_negatif_refuse():
    with pytest.raises(ValueError):
        pipeline(eval_every=-1, dependances={"cuda_disponible": lambda: True})


def test_les_parametres_stockfish_sont_refuses():
    cas: list[dict] = [{"stockfish_path": "x"}, {"stockfish_elo": 2600},
                       {"stockfish_nodes": 200_000},
                       {"num_sim_eval_sf": 700}]
    for parametres in cas:
        with pytest.raises(ValueError, match="Stockfish"):
            pipeline(dependances={"cuda_disponible": lambda: True},
                     **parametres)


def test_eval_stockfish_every_devient_eval_every(tmp_path, capsys):
    appels = _appels_vides()

    pipeline(num_iterations=10, eval_stockfish_every=5,
             eval_every=4, position_bench_path=tmp_path / "banc",
             eval_output_dir=tmp_path / "resultats",
             dependances=_dependances(appels, JournalWandb()))

    assert [appel["iteration"] for appel in appels["evaluation"]] == [5, 10]
    assert "eval_stockfish_every" in capsys.readouterr().out


def test_un_banc_absent_arrete_avant_la_generation(tmp_path):
    appels = _appels_vides()
    dependances = _dependances(appels, JournalWandb())
    del dependances["charger_banc"]  # load_dataset reel

    with pytest.raises((FileNotFoundError, ValueError)):
        pipeline(num_iterations=1, eval_every=4,
                 position_bench_path=tmp_path / "absent",
                 eval_output_dir=tmp_path / "resultats",
                 dependances=dependances)

    assert appels["generation"] == []


def test_wandb_mode_disabled_reste_transmis(monkeypatch, tmp_path):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    appels = _appels_vides()

    pipeline(num_iterations=1, eval_every=0, wandb_mode="disabled",
             dependances=_dependances(appels, JournalWandb()))

    assert os.environ["WANDB_MODE"] == "disabled"


def test_la_cli_refuse_les_drapeaux_stockfish():
    with pytest.raises(SystemExit) as info:
        train_self_play.principal(["--stockfish", "stockfish.exe"])

    assert "Stockfish" in str(info.value)


def test_run_selftrain_cadence_par_defaut_et_transmission(monkeypatch):
    capture = {}
    monkeypatch.setattr(run_selftrain, "etat_du_buffer",
                        lambda dossier: ([], 0))
    monkeypatch.setattr(sys, "argv", ["run_selftrain.py", "--iterations", "1"])

    def pipeline_capture(**kwargs):
        capture.update(kwargs)

    monkeypatch.setattr(train_self_play, "pipeline", pipeline_capture)

    run_selftrain.main()

    assert capture["eval_every"] == 4
    assert capture["eval_search_workers"] == 8
    assert capture["eval_target_s"] == 300.0
    assert capture["position_bench_path"] is None
    assert capture["eval_output_dir"] is None


def test_run_selftrain_n_expose_plus_les_options_stockfish(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_selftrain.py"])

    args = run_selftrain.analyser_arguments()

    assert not hasattr(args, "stockfish")
    assert not hasattr(args, "stockfish_elo")
