"""Isolated latency, throughput and memory profile of every agent, without the simulator in the loop.

Each agent is queried on the first frames of the held-out demonstrations, with the matching goal frames, for
``--calls`` timed decisions after ``--warmup`` untimed ones. Reported per agent: mean and standard deviation of
the encode, plan and extract time per decision, decisions per second, peak GPU memory and parameter counts.

Usage:
    python scripts/profile_latency.py --out results/tables/latency.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.agents import BCAgent, LatentPlannerAgent, TDMPC2Agent  # noqa: E402
from src.data.libero_data import load_demos, load_images  # noqa: E402
from src.planning.cem import CEMConfig  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

AGENTS = ["vjepa2_ac", "dino_wm", "tdmpc2", "bc_resnet18"]


def count(module: torch.nn.Module, trainable: bool | None = None) -> int:
    return sum(p.numel() for p in module.parameters() if trainable is None or p.requires_grad == trainable)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", default="results/tables/latency.json")
    args = ap.parse_args()
    set_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    demos = sorted([d for d in load_demos() if d.split == "test"], key=lambda d: (d.task_id, d.index))
    frames = []
    for d in demos:
        imgs, wr = load_images(d, "agentview"), load_images(d, "wrist")
        frames.append(({"agentview": imgs[0], "wrist": wr[0], "state": d.state[0]},
                       {"agentview": imgs[-1], "wrist": wr[-1], "state": d.state[-1]}, d.task_id))
    out = {}
    for name in AGENTS:
        ck = torch.load(f"results/checkpoints/{name}.pt", map_location="cpu", weights_only=False)
        kind = ck["config"]["model_type"]
        if dev == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        if kind in ("vjepa2_ac", "dino_wm"):
            agent = LatentPlannerAgent(ck, CEMConfig(), "l1" if kind == "vjepa2_ac" else "mse", dev)
            params = {"encoder_frozen": count(agent.encoder), "trained": count(agent.model)}
        elif kind == "tdmpc2":
            agent = TDMPC2Agent(ck, dev)
            params = {"trained": sum(p.numel() for n, p in agent.model.named_parameters() if not n.startswith("target"))}
        else:
            agent = BCAgent(ck, dev)
            params = {"trained": count(agent.model)}
        times = {"encode_ms": [], "plan_ms": [], "extract_ms": []}
        for i in range(args.warmup + args.calls):
            obs, goal, tid = frames[i % len(frames)]
            agent.reset(tid, goal)
            _, info = agent.act(obs)
            if i >= args.warmup:
                for k in times:
                    times[k].append(info[k])
        total = np.array(times["encode_ms"]) + np.array(times["plan_ms"]) + np.array(times["extract_ms"])
        out[name] = {
            **{k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in times.items()},
            "total_ms": {"mean": float(total.mean()), "std": float(total.std())},
            "decisions_per_s": float(1000.0 / total.mean()),
            "peak_vram_mb": float(torch.cuda.max_memory_allocated() / 2**20) if dev == "cuda" else None,
            "params": params, "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
        }
        print(name, json.dumps(out[name]["total_ms"]), out[name]["peak_vram_mb"], params, flush=True)
        del agent
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
