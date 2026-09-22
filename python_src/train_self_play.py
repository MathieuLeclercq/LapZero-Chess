import warnings
import logging
import time
import gc
import os.path
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", module="requests")
import wandb
import torch
import torch.nn.functional as f
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.amp import GradScaler
from datetime import datetime

import chess_engine
from lib import (append_to_disk_buffer, export_model_to_onnx,
                 calculate_performance_rating, convert_game_results, run_with_interrupt)
from stockfish_player import evaluate_against_anchor
from model import ChessNet

# Chemin absolu : l'injection de puzzles est chargee cote C++ et un chemin
# relatif dependrait du repertoire de lancement.
PUZZLES_PATH = str(Path(__file__).resolve().parents[1]
                   / "training_data" / "puzzles_train.txt")
STOCKFISH_PATH = str(Path(__file__).resolve().parent / "lichess_bot"
                     / "stockfish" / "stockfish-windows-x86-64-universal.exe")


# ============================================================
#                     DATASET
# ============================================================
class ShardedDataset(Dataset):
    """Dataset qui charge un seul shard à la fois en mémoire."""

    def __init__(self, shard_path):
        with np.load(shard_path) as data:
            self.states = data['states'].copy()
            self.policies = data['policies'].copy()
            self.values = data['values'].copy()

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.states[idx].copy()).float(),
            torch.from_numpy(self.policies[idx].copy()).float(),
            torch.tensor(float(self.values[idx]), dtype=torch.float32)
        )


# ============================================================
#                     SELF-PLAY (GPU Batched)
# ============================================================
def generate_games(
        onnx_path,
        games_per_iter,
        concurrent_games,
        slow_sims,
        fast_sims,
        slow_ratio,
        tt_size=2097143):
    """
    Génère des parties via le SelfPlayManager C++ avec inférence batchée sur GPU.
    """
    evaluator = chess_engine.ONNXEvaluator(onnx_path, True)

    game_results = run_with_interrupt(
        chess_engine.generate_self_play_games,
        evaluator,
        concurrent_games,
        slow_sims,
        fast_sims,
        games_per_iter,
        slow_ratio,
        tt_size,
        PUZZLES_PATH
    )

    data, stats = convert_game_results(game_results)

    num_games = len(game_results)
    total_ply_played = sum(res.total_real_moves for res in game_results)
    avg_length = total_ply_played / num_games

    print(f"\n{'=' * 30}")
    print(f"      BILAN DE L'ITERATION")
    print(f"{'=' * 30}")
    print(f"  Parties jouées                     : {num_games}")
    print(f"  Positions générées (slow moves)    : {len(data)}")
    print(f"  Positions par Partie               : {avg_length:.1f}")
    print(f"{'-' * 30}")
    print(f"  Victoires (mat)                    : {stats['checkmates']}")
    print(f"  Pat (Stalemate)                    : {stats['stalemates']}")
    print(f"  Répétition                         : {stats['repetition']}")
    print(f"  Règle des 50 coups                 : {stats['50_moves']}")
    print(f"  Matériel insuffisant               : {stats['insuff_mat']}")
    print(f"  Non terminées (Max)                : {stats['max_moves']}")
    print(f"{'=' * 30}\n")

    # Libération explicite de l'évaluateur ONNX GPU
    del game_results
    del evaluator
    gc.collect()
    torch.cuda.empty_cache()

    return data, avg_length, stats


# ============================================================
#                     TRAINING
# ============================================================
def train_on_shards(model, optimizer, scaler, device, buffer_folder, learning_rate,
                    batch_size=256, global_step=0, samples_per_epoch=15000):
    import glob, random

    model.train()
    shards = sorted(glob.glob(os.path.join(buffer_folder, "shard_*.npz")))
    if not shards:
        return global_step

    shard_sizes = []
    for s in shards:
        try:
            size = int(os.path.splitext(s)[0].split("_")[-1])
        except (ValueError, IndexError):
            size = 50000
        shard_sizes.append(size)
    total_positions = sum(shard_sizes)

    paired = list(zip(shards, shard_sizes))
    random.shuffle(paired)

    epoch_loss = 0.0
    epoch_policy_loss = 0.0
    epoch_value_loss = 0.0
    num_batches = 0
    samples_remaining = samples_per_epoch

    for shard_path, shard_size in paired:
        if samples_remaining <= 0:
            break

        shard_samples = max(1, round(samples_per_epoch * shard_size / total_positions))
        shard_samples = min(shard_samples, samples_remaining)

        dataset = ShardedDataset(shard_path)
        sampler = RandomSampler(dataset, replacement=True, num_samples=shard_samples)
        loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=sampler, num_workers=0, pin_memory=True)

        for x, target_pi, y_value in loader:
            x = x.to(device)
            target_pi = target_pi.to(device)
            y_value = y_value.to(device)

            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                p_logits, v_pred = model(x)
                log_probs = f.log_softmax(p_logits, dim=1)
                policy_loss = -torch.sum(target_pi * log_probs, dim=1).mean()
                value_loss = f.mse_loss(v_pred.view(-1), y_value)
                loss = policy_loss + value_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()
            num_batches += 1
            global_step += 1

        samples_remaining -= shard_samples
        del dataset

    if num_batches > 0:
        avg_loss = epoch_loss / num_batches
        wandb.log({
            "train/epoch_loss": avg_loss,
            "train/epoch_policy_loss": epoch_policy_loss / num_batches,
            "train/epoch_value_loss": epoch_value_loss / num_batches,
            "train/global_step": global_step,
            "train/learning_rate": learning_rate,
        }, step=global_step)
        print(f"    Training — loss: {avg_loss:.4f} ({num_batches} batches)")

    torch.cuda.empty_cache()
    return global_step


# ============================================================
#                     PIPELINE
# ============================================================
def pipeline(
        num_iterations=2,
        games_per_iter=128,
        concurrent_games=128,
        slow_sims=700,
        fast_sims=100,
        slow_ratio=0.25,
        batch_size=1024,
        learning_rate=1e-4,
        num_res_blocks=10,
        num_filters=128,
        max_buffer_size=100_000,
        target_sampling_ratio=14.0,
        eval_stockfish_every=4,
        checkpoint_path=None,
        stockfish_path=None,
        stockfish_elo=2500,
        stockfish_nodes=200_000,
        num_sim_eval_sf=700
):
    logging.getLogger("torch").setLevel(logging.ERROR)
    hyperparams = locals().copy()

    timestamp = datetime.now().strftime("%Y_%m_%d_%Hh%M")
    assert torch.cuda.is_available()
    gpu_device = torch.device("cuda")
    assert os.path.isfile(stockfish_path)

    model = ChessNet(num_res_blocks=num_res_blocks, num_filters=num_filters).to(gpu_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scaler = GradScaler("cuda", enabled=True)

    global_step = 0
    start_iteration = 0

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=gpu_device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_iteration = checkpoint.get("iteration", 0)
        global_step = checkpoint.get("global_step", 0)
        print(f"Checkpoint chargé : {checkpoint_path} (Reprise à l'itération {start_iteration})")

    wandb.init(project="alphazero-chess", name=f"{timestamp}_self_play", config=hyperparams)

    buffer_folder = "replay_buffer"

    # ── INITIALISATION AVANT LA BOUCLE (Génération de l'iter 0 ou reprise) ──
    current_onnx_name = f"{timestamp}_iter{start_iteration}_unsupervised"
    current_onnx_path = f"checkpoints/{current_onnx_name}.onnx"
    export_model_to_onnx(model, current_onnx_path, gpu_device)
    print(f"Modèle ONNX initial prêt pour le self-play : {current_onnx_path}")

    for iteration in range(start_iteration, start_iteration + num_iterations):
        print(f"\n{'=' * 50}")
        print(f"  ITERATION {iteration + 1}/{start_iteration + num_iterations}")
        print(f"{'=' * 50}")

        # ── 1. Self-Play (C++ / GPU batched) ──
        start_time = time.time()
        new_data, avg_length, stats = generate_games(
            current_onnx_path, games_per_iter,
            concurrent_games, slow_sims, fast_sims, slow_ratio,
            tt_size=4_000_000
        )
        generation_time = time.time() - start_time
        games_per_sec = games_per_iter / generation_time
        saved_pos_per_sec = len(new_data) / generation_time

        print(
            f"  Vitesse : {games_per_sec:.2f} parties/s | "
            f"{saved_pos_per_sec:.0f} saved positions/s (Total: {generation_time:.1f}s)")

        # ── 2. Sauvegarde des nouvelles positions sur disque ──
        buffer_size = append_to_disk_buffer(new_data, buffer_folder, max_buffer_size)
        num_new_positions = len(new_data)
        del new_data

        # ── 3. Training (GPU / PyTorch) ──
        samples_per_epoch = round(target_sampling_ratio * num_new_positions)
        print(f"  Entraînement sur {samples_per_epoch} samples...")

        global_step = train_on_shards(
            model, optimizer, scaler, gpu_device, buffer_folder, learning_rate,
            batch_size=batch_size, global_step=global_step,
            samples_per_epoch=samples_per_epoch
        )

        num_games = max(1, games_per_iter)
        draw_rate = 1 - (stats["checkmates"] / num_games)
        wandb.log({
            "selfplay/buffer_size": buffer_size,
            "selfplay/new_positions": num_new_positions,
            "selfplay/avg_game_length": avg_length,
            "selfplay/draw_rate": draw_rate,
            "selfplay/draws_repetition": stats["repetition"] / num_games,
            "selfplay/draws_50_moves": stats["50_moves"] / num_games,
            "selfplay/draws_stalemate": stats["stalemates"] / num_games,
            "selfplay/draws_insuff_mat": stats["insuff_mat"] / num_games,
            "selfplay/draws_max_moves": stats["max_moves"] / num_games,
            "selfplay/games_per_sec": games_per_sec,
            "selfplay/saved_positions_per_sec": saved_pos_per_sec,
            "selfplay/iteration": iteration + 1,
        }, step=global_step)

        # ── 4. Sauvegarde checkpoint .pt ET .onnx ──
        ckpt_filename = f"{timestamp}_iter{iteration + 1}_unsupervised"
        save_pt_path = f"checkpoints/{ckpt_filename}.pt"

        # Sauvegarde PyTorch
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "iteration": iteration + 1,
            "global_step": global_step,
        }, save_pt_path)

        # Nouvel export ONNX qui servira pour l'itération suivante et l'évaluation
        current_onnx_path = f"checkpoints/{ckpt_filename}.onnx"
        export_model_to_onnx(model, current_onnx_path, gpu_device)

        print(f"  Checkpoints sauvegardés : {ckpt_filename} (.pt et .onnx)")

        # ── 5. Évaluation Stockfish ──
        if (iteration + 1) % eval_stockfish_every == 0:
            eval_winrate, eval_wins, eval_draws, eval_losses = evaluate_against_anchor(
                onnx_path=current_onnx_path,
                stockfish_path=stockfish_path,
                num_games=16,
                mcts_sims=num_sim_eval_sf,
                sf_elo=stockfish_elo,
                sf_nodes=stockfish_nodes,
                num_workers=8
            )

            estim_elo = calculate_performance_rating(
                stockfish_elo, eval_wins, eval_draws, eval_losses
            )

            wandb.log({
                "eval/winrate": eval_winrate,
                "eval/elo_estim": estim_elo,
                "eval/wins": eval_wins,
                "eval/draws": eval_draws,
                "eval/losses": eval_losses,
                "eval/stockfish_elo": stockfish_elo,
                "eval/iteration": iteration + 1,
            }, step=global_step)
        else:
            print(f"  Évaluation ignorée.")
        wandb.log({}, commit=True)

    wandb.finish()


if __name__ == "__main__":
    try:
        # import os
        # os.environ["WANDB_MODE"] = "disabled"

        pipeline(
            num_iterations=150,
            games_per_iter=512,
            concurrent_games=256,
            slow_sims=700,
            fast_sims=100,
            slow_ratio=0.25,
            batch_size=4096,
            learning_rate=4e-5,
            max_buffer_size=750_000,
            target_sampling_ratio=14.0,
            eval_stockfish_every=8,
            checkpoint_path="checkpoints/2026_04_30_09h53_iter436_unsupervised.pt",
            stockfish_path=STOCKFISH_PATH,
            stockfish_elo=2600,
            stockfish_nodes=200_000
        )
    except KeyboardInterrupt:
        print("\n[Interruption] Entraînement stoppé manuellement.")
        print("Synchronisation des dernières métriques avec WandB en cours...")
        wandb.finish()
        print("Arrêt propre terminé.")
