import os
import sys
import time
import argparse
import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ensure mesa/OSMesa shared libraries are discoverable
osmesa_lib_path = "/home/selim/.local/share/mesa24-osmesa/lib"
claude_prism_lib = "/home/selim/.local/share/claude-prism/app/usr/lib"
cur_ld = os.environ.get("LD_LIBRARY_PATH", "")
if osmesa_lib_path not in cur_ld:
    os.environ["LD_LIBRARY_PATH"] = f"{osmesa_lib_path}:{claude_prism_lib}:{cur_ld}"
os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.env.manipulation_env import ManipulationEnv
from src.env.libero_env import LiberoMuJoCoEnv
from src.models.vjepa_ac_interface import VJEPA2ACModel
from src.models.dino_wm import DINOWMModel
from src.models.tdmpc2_model import TDMPC2Model
from src.models.bc_policy import BCPolicyModel
from src.planning.mpc_controller import LatentMPCController
from src.utils.seed import set_seed
from src.utils.metrics import (
    compute_latent_mse,
    compute_goal_distance,
    aggregate_multi_seed_metrics,
)


def load_model(checkpoint_path: str, model_type: str, device: str = "cpu") -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})

    if model_type == "vjepa2_ac":
        model = VJEPA2ACModel(
            img_size=config.get("img_size", 224),
            patch_size=config.get("patch_size", 16),
            embed_dim=config.get("embed_dim", 128),
            predictor_embed_dim=config.get("predictor_embed_dim", 128),
            predictor_depth=config.get("predictor_depth", 4),
            action_dim=config.get("action_dim", 4),
        )
    elif model_type == "dino_wm":
        model = DINOWMModel(
            img_size=config.get("img_size", 224),
            patch_size=config.get("patch_size", 16),
            embed_dim=config.get("embed_dim", 128),
            action_dim=config.get("action_dim", 4),
        )
    elif model_type == "tdmpc2":
        model = TDMPC2Model(
            img_size=config.get("img_size", 224),
            latent_dim=config.get("latent_dim", 64),
            action_dim=config.get("action_dim", 4),
        )
    elif model_type == "bc_policy":
        model = BCPolicyModel(
            img_size=config.get("img_size", 224),
            embed_dim=config.get("embed_dim", 64),
            action_dim=config.get("action_dim", 4),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def evaluate_single_seed(
    model: torch.nn.Module,
    model_type: str,
    task_name: str,
    num_episodes: int = 10,
    max_steps: int = 25,
    seed: int = 0,
    horizon: int = 3,
    num_samples: int = 25,
    num_iterations: int = 3,
    device: str = "cpu",
    env_type: str = "manipulation",
) -> Dict[str, Any]:
    set_seed(seed)

    if env_type == "libero":
        env = LiberoMuJoCoEnv(suite_name="libero_10", task_id=0, image_size=224, max_episode_steps=max_steps, seed=seed)
    else:
        env = ManipulationEnv(task_name=task_name, image_size=224, max_episode_steps=max_steps, seed=seed)

    if model_type != "bc_policy":
        controller = LatentMPCController(
            world_model=model,
            horizon=horizon,
            num_samples=num_samples,
            num_iterations=num_iterations,
            action_dim=4,
            action_low=torch.tensor([-0.05, -0.05, -0.05, -1.0], device=device),
            action_high=torch.tensor([0.05, 0.05, 0.05, 1.0], device=device),
            device=device,
        )

    success_count = 0
    goal_distances = []
    latencies = []
    episode_lengths = []
    latent_mses_h1 = []
    latent_mses_h3 = []
    latent_mses_h5 = []
    latent_mses_h10 = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep * 13)
        goal_obs = env.get_goal_image()
        done = False
        steps = 0

        # Precompute goal representation for latent models
        with torch.no_grad():
            goal_tensor = torch.from_numpy(goal_obs).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            if model_type != "bc_policy":
                z_goal = model.encode(goal_tensor)

        ep_obs_list = []
        ep_act_list = []

        while not done:
            t0 = time.perf_counter()
            obs_t = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            ep_obs_list.append(obs_t)

            if model_type == "bc_policy":
                with torch.no_grad():
                    action_t = model(obs_t)[0].cpu().numpy()
            else:
                action_tensor = controller.get_action(obs_t, goal_tensor)
                action_t = action_tensor.cpu().numpy()

            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
            ep_act_list.append(torch.from_numpy(action_t).float().unsqueeze(0))

            obs, reward, terminated, truncated, info = env.step(action_t)
            done = terminated or truncated
            steps += 1

        if info.get("success", False):
            success_count += 1

        current_pose = info["state"]
        goal_pose = info.get("goal_state", np.zeros_like(current_pose))
        goal_distances.append(compute_goal_distance(current_pose, goal_pose))
        episode_lengths.append(steps)

        # Multi-horizon latent rollout evaluation along executed trajectory
        if model_type != "bc_policy" and len(ep_obs_list) >= 2:
            with torch.no_grad():
                z_init = model.encode(ep_obs_list[0])
                T_ep = len(ep_obs_list)
                curr_z = z_init
                for step_idx in range(min(T_ep - 1, 10)):
                    act = ep_act_list[step_idx]
                    curr_z = model.predict(curr_z, act)
                    gt_z = model.encode(ep_obs_list[step_idx + 1])
                    mse_k = compute_latent_mse(curr_z, gt_z)
                    if step_idx == 0:
                        latent_mses_h1.append(mse_k)
                    elif step_idx == 2:
                        latent_mses_h3.append(mse_k)
                    elif step_idx == 4:
                        latent_mses_h5.append(mse_k)
                    elif step_idx == 9:
                        latent_mses_h10.append(mse_k)

    env.close()

    return {
        "success_rate": (success_count / num_episodes) * 100.0,
        "goal_distance_cm": float(np.mean(goal_distances)),
        "latency_ms": float(np.mean(latencies)),
        "episode_length": float(np.mean(episode_lengths)),
        "latent_mse_h1": float(np.mean(latent_mses_h1)) if latent_mses_h1 else 0.0,
        "latent_mse_h3": float(np.mean(latent_mses_h3)) if latent_mses_h3 else 0.0,
        "latent_mse_h5": float(np.mean(latent_mses_h5)) if latent_mses_h5 else 0.0,
        "latent_mse_h10": float(np.mean(latent_mses_h10)) if latent_mses_h10 else 0.0,
    }


def run_benchmark(
    checkpoint_dir: str = "results/checkpoints",
    output_dir: str = "results/tables",
    tasks: List[str] = ["reach", "push", "pick_place", "drawer_open"],
    models: List[str] = ["vjepa2_ac", "dino_wm", "tdmpc2", "bc_policy"],
    seeds: List[int] = [0],
    num_episodes_per_seed: int = 10,
    device: str = "cpu",
    env_type: str = "manipulation",
) -> Dict[str, Any]:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    all_results = {}
    csv_rows = []

    print("==================================================")
    print("STARTING CLOSED-LOOP BENCHMARK EVALUATION")
    print(f"Seeds: {seeds} | Tasks: {tasks} | Models: {models} | Rollouts/ep: {num_episodes_per_seed}")
    print("==================================================")

    for task in tasks:
        all_results[task] = {}
        for model_name in models:
            ckpt_path = os.path.join(checkpoint_dir, f"{model_name}_{task}_best.pt")
            if not os.path.exists(ckpt_path):
                print(f"Warning: Checkpoint not found: {ckpt_path}. Skipping.")
                continue

            print(f"\nEvaluating {model_name} on {task}...")
            model = load_model(ckpt_path, model_type=model_name, device=device)

            seed_runs = []
            for seed in seeds:
                res = evaluate_single_seed(
                    model=model,
                    model_type=model_name,
                    task_name=task,
                    num_episodes=num_episodes_per_seed,
                    seed=seed,
                    device=device,
                    env_type=env_type,
                )
                seed_runs.append(res)
                print(
                    f"  Seed {seed}: SR = {res['success_rate']:.1f}%, GoalDist = {res['goal_distance_cm']:.2f}cm, "
                    f"Latency = {res['latency_ms']:.1f}ms, H1_MSE = {res['latent_mse_h1']:.4f}, H5_MSE = {res['latent_mse_h5']:.4f}"
                )

                csv_rows.append({
                    "task": task,
                    "model": model_name,
                    "seed": seed,
                    "success_rate": res["success_rate"],
                    "goal_distance_cm": res["goal_distance_cm"],
                    "latency_ms": res["latency_ms"],
                    "latent_mse_h1": res["latent_mse_h1"],
                    "latent_mse_h3": res["latent_mse_h3"],
                    "latent_mse_h5": res["latent_mse_h5"],
                    "latent_mse_h10": res["latent_mse_h10"],
                })

            summary = aggregate_multi_seed_metrics(seed_runs)
            all_results[task][model_name] = summary
            print(f"  --> {model_name} Success Rate: {summary['success_rate']['mean']:.1f}% ± {summary['success_rate']['std']:.1f}%")

    # Save comprehensive benchmark results to JSON
    json_path = out_path / "comparative_benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nBenchmark results saved to: {json_path}")

    # Save CSV table
    csv_path = out_path / "eval_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "task", "model", "seed", "success_rate", "goal_distance_cm",
            "latency_ms", "latent_mse_h1", "latent_mse_h3", "latent_mse_h5", "latent_mse_h10"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Metrics CSV saved to: {csv_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Closed-loop evaluation benchmark.")
    parser.add_argument("--ckpt_dir", type=str, default="results/checkpoints")
    parser.add_argument("--output_dir", type=str, default="results/tables")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--env", type=str, default="manipulation")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    run_benchmark(
        checkpoint_dir=args.ckpt_dir,
        output_dir=args.output_dir,
        seeds=args.seeds,
        num_episodes_per_seed=args.num_episodes,
        device=args.device,
        env_type=args.env,
    )
