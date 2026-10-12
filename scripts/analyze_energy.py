"""Planning-energy landscape versus true outcome over a 2-D slice of action space.

At the first decision of a held-out episode, the first planned macro action is perturbed over a grid of
(dx, dy) commands while the rest of the CEM plan is kept. For each grid point the script records
(i) the latent planning energy predicted by the world model and (ii) the true goal error obtained by
executing the same plan open loop in the simulator. The CEM mean of the first action at every iteration is
saved so the optimiser's path can be drawn over both surfaces.

    python scripts/analyze_energy.py --agent vjepa2_ac --task turn_on_the_stove --episode 0 --grid 15
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.agents import LatentPlannerAgent  # noqa: E402
from src.data.libero_data import load_demos  # noqa: E402
from src.env.libero_env import TASK_KEYS, LiberoGoalEnv  # noqa: E402
from src.planning.cem import CEMConfig  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="vjepa2_ac")
    ap.add_argument("--task", default="turn_on_the_stove")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--grid", type=int, default=15)
    ap.add_argument("--execute", type=int, default=10, help="macro steps of the plan executed for the true outcome")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--out", default="results/tables")
    args = ap.parse_args()
    set_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(f"results/checkpoints/{args.agent}.pt", map_location="cpu", weights_only=False)
    kind = ck["config"]["model_type"]
    agent = LatentPlannerAgent(ck, CEMConfig(horizon=args.horizon), "l1" if kind == "vjepa2_ac" else "mse", dev, seed=0)
    demo = sorted([d for d in load_demos() if d.task == args.task and d.split == "test"], key=lambda d: d.index)[args.episode]
    env = LiberoGoalEnv(args.task)
    goal = env.set_goal_from_state(demo.sim_state[-1])
    obs, _ = env.reset(options={"init_state": demo.sim_state[0]})
    agent.reset(TASK_KEYS.index(args.task), goal)
    _, info = agent.act(obs, record=True)
    plan = torch.as_tensor(info["plan"], device=dev)
    path = np.stack([t["mean0"][:2] for t in info["cem_trace"]])
    xs = np.linspace(-1, 1, args.grid)
    energy = np.zeros((args.grid, args.grid))
    true_err = np.zeros((args.grid, args.grid))
    for i, dy in enumerate(xs):
        acts = plan.repeat(args.grid, 1, 1)
        acts[:, 0, 0] = torch.as_tensor(xs, device=dev, dtype=acts.dtype)
        acts[:, 0, 1] = float(dy)
        with torch.no_grad():
            energy[i] = agent.costs(acts).cpu().numpy()
        for j, dx in enumerate(xs):
            env.reset(options={"init_state": demo.sim_state[0]})
            a = acts[j].cpu().numpy()
            for t in range(min(args.execute, len(a))):
                _, _, term, _, inf = env.step(a[t])
                if term:
                    break
            true_err[i, j] = env.goal_error_cm()
        print(f"row {i + 1}/{args.grid}", flush=True)
    env.close()
    os.makedirs(args.out, exist_ok=True)
    out = {"agent": args.agent, "task": args.task, "horizon": args.horizon, "demo": demo.index, "grid": xs.tolist(), "energy": energy.tolist(),
           "true_goal_error_cm": true_err.tolist(), "cem_path_xy": path.tolist(), "plan": info["plan"].tolist(),
           "execute_macro_steps": args.execute,
           "spearman_energy_vs_true": float(
               np.corrcoef(np.argsort(np.argsort(energy.ravel())), np.argsort(np.argsort(true_err.ravel())))[0, 1])}
    json.dump(out, open(os.path.join(args.out, f"energy_{args.agent}_{args.task}_ep{args.episode}_H{args.horizon}.json"), "w"))
    print("spearman", out["spearman_energy_vs_true"])


if __name__ == "__main__":
    main()
