"""Closed-loop evaluation on the 10 held-out demonstrations of each task (seed 0).

Each episode starts from the first MuJoCo state of a held-out demonstration, and the goal frame is rendered
from its final, successful state. An episode ends on LIBERO's BDDL success predicate or after 300 control
steps (60 macro steps), the LIBERO-Goal limit used in OpenVLA's evaluation.

Usage:
    python scripts/evaluate.py --agent vjepa2_ac --tag vjepa2_ac
    python scripts/evaluate.py --agent vjepa2_ac --tag abl_H5 --tasks turn_on_the_stove --horizon 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.agents import BCAgent, LatentPlannerAgent, TDMPC2Agent  # noqa: E402
from src.data.libero_data import feature_path, load_demos  # noqa: E402
from src.env.libero_env import TASK_KEYS, TASKS, LiberoGoalEnv  # noqa: E402
from src.planning.cem import CEMConfig  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def channel_variance(encoder: str, demos) -> torch.Tensor:
    feats = torch.cat([torch.load(feature_path(encoder, d)).float().flatten(0, 1) for d in demos if d.split == "train"])
    return feats.var(0)


def build_agent(args, demos, dev):
    ckpt = torch.load(os.path.join(args.ckpt_dir, f"{args.agent}.pt"), map_location="cpu", weights_only=False)
    kind = ckpt["config"]["model_type"]
    if kind in ("vjepa2_ac", "dino_wm"):
        init_std = 1.0
        if args.action_prior == "data":
            # per-dimension spread of the demonstrated macro actions (training split only)
            acts = np.concatenate([np.stack([d.action[t : t + 5].mean(0) for t in range(0, len(d.action) - 5, 5)])
                                   for d in demos if d.split == "train"])
            init_std = tuple(float(x) for x in acts.std(0))
        cem = CEMConfig(horizon=args.horizon, samples=args.samples, iterations=args.iterations,
                        elite_frac=args.elite_frac, smooth=args.smooth, warm_start=args.warm_start, init_std=init_std)
        cost = args.cost or ("l1" if kind == "vjepa2_ac" else "mse")
        var = channel_variance(ckpt["config"]["encoder"], demos) if cost == "mahalanobis" else None
        return LatentPlannerAgent(ckpt, cem, cost, dev, seed=args.seed, channel_var=var), kind
    if kind == "tdmpc2":
        return TDMPC2Agent(ckpt, dev, seed=args.seed), kind
    return BCAgent(ckpt, dev), kind


def run_episode(env, agent, demo, task_id, record: bool, max_macro: int):
    goal = env.set_goal_from_state(demo.sim_state[-1])
    obs, info = env.reset(options={"init_state": demo.sim_state[0]})
    agent.reset(task_id, goal)
    frames, wrists, recs = [obs["agentview"]], [obs["wrist"]], []
    lat = {"encode_ms": [], "plan_ms": [], "extract_ms": []}
    ret, t0, success = 0.0, time.time(), False
    errs = [env.goal_error_cm()]
    meas = [{k: v.tolist() for k, v in env.task_measurements().items()}]
    for _ in range(max_macro):
        action, ainfo = agent.act(obs, record=record)
        for k in lat:
            lat[k].append(ainfo.get(k, 0.0))
        obs, r, term, trunc, info = env.step(action)
        ret += r
        errs.append(info["goal_error_cm"])
        meas.append({k: v.tolist() for k, v in env.task_measurements().items()})
        frames.append(obs["agentview"])
        wrists.append(obs["wrist"])
        if record:
            recs.append({k: v for k, v in ainfo.items() if not k.endswith("_ms")} | {"action": action})
        if term:
            success = True
            break
        if trunc:
            break
    out = {
        "demo": demo.index, "success": success, "env_steps": info["env_steps"], "macro_steps": len(lat["plan_ms"]),
        "final_goal_error_cm": errs[-1], "min_goal_error_cm": float(np.nanmin(errs)),
        "initial_goal_error_cm": errs[0], "return": ret, "wall_s": time.time() - t0,
        "latency_ms": {k: float(np.mean(v)) for k, v in lat.items()},
        "goal_error_trace": errs, "measurement_trace": meas,
    }
    return out, np.stack(frames), np.stack(wrists), goal, recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tasks", nargs="+", default=TASK_KEYS)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_env_steps", type=int, default=300)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--elite_frac", type=float, default=0.05)
    ap.add_argument("--smooth", type=float, default=0.0)
    ap.add_argument("--warm_start", action="store_true")
    ap.add_argument("--action_prior", default="uniform", choices=["uniform", "data"],
                    help="CEM initial spread: 1 in every dimension, or the per-dimension std of demo macro actions")
    ap.add_argument("--cost", default=None, choices=[None, "l1", "mse", "l2", "cosine", "mahalanobis"])
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--ckpt_dir", default="results/checkpoints")
    ap.add_argument("--out", default="results/eval")
    args = ap.parse_args()
    set_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    demos = load_demos()
    agent, kind = build_agent(args, demos, dev)
    out_dir = os.path.join(args.out, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for task in args.tasks:
        tid = TASK_KEYS.index(task)
        test = sorted([d for d in demos if d.task == task and d.split == "test"], key=lambda d: d.index)[: args.episodes]
        env = LiberoGoalEnv(task, max_env_steps=args.max_env_steps)
        episodes = []
        for j, d in enumerate(test):
            set_seed(args.seed)
            ep, frames, wrists, goal, recs = run_episode(env, agent, d, tid, args.record, args.max_env_steps // 5)
            episodes.append(ep)
            print(f"[{args.tag}] {TASKS[task].short} ep {j} demo {d.index}: success={ep['success']} "
                  f"steps={ep['env_steps']} err={ep['final_goal_error_cm']:.1f}cm plan={ep['latency_ms']['plan_ms']:.0f}ms "
                  f"wall={ep['wall_s']:.0f}s", flush=True)
            if args.record:
                rdir = os.path.join("results/rollouts", args.tag)
                os.makedirs(rdir, exist_ok=True)
                payload = {"frames": frames, "wrist": wrists, "goal": goal["agentview"]}
                for key in ["imagined_pooled", "realized_pooled", "plan_cost", "goal_cost_now", "action", "plan",
                            "imagined_next", "realized"]:
                    if recs and key in recs[0]:
                        payload[key] = np.stack([np.asarray(r[key]) for r in recs])
                if recs and "cem_trace" in recs[0]:
                    payload["cem_mean0"] = np.stack([np.stack([t["mean0"] for t in r["cem_trace"]]) for r in recs])
                    payload["cem_best"] = np.array([[t["best_cost"] for t in r["cem_trace"]] for r in recs])
                np.savez_compressed(os.path.join(rdir, f"{task}__ep{j:02d}.npz"), **payload)
        env.close()
        sr = float(np.mean([e["success"] for e in episodes]))
        summary = {
            "task": task, "agent": args.agent, "kind": kind, "tag": args.tag, "seed": args.seed,
            "episodes": len(episodes), "success_rate": sr,
            "final_goal_error_cm": float(np.mean([e["final_goal_error_cm"] for e in episodes])),
            "return": float(np.mean([e["return"] for e in episodes])),
            "steps_to_success": float(np.mean([e["env_steps"] for e in episodes if e["success"]])) if sr > 0 else None,
            "latency_ms": {k: float(np.mean([e["latency_ms"][k] for e in episodes])) for k in episodes[0]["latency_ms"]},
            "peak_vram_mb": float(torch.cuda.max_memory_allocated() / 2**20) if dev == "cuda" else None,
            "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
            "config": vars(args), "episodes_detail": episodes,
        }
        json.dump(summary, open(os.path.join(out_dir, f"{task}.json"), "w"), indent=1)
        print(f"[{args.tag}] {TASKS[task].short}: success {sr:.2f}", flush=True)


if __name__ == "__main__":
    main()
