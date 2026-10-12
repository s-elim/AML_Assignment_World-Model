"""Train TD-MPC2 offline on the demonstrations with LIBERO's sparse success reward.

Usage:
    python scripts/train_tdmpc2.py --config configs/model/tdmpc2.yaml
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
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402

from src.data.libero_data import StateNorm, load_demos, load_images  # noqa: E402
from src.models.tdmpc2 import TDMPC2, RunningScale, tdmpc2_update  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def heuristic_gamma(episode_macro_steps: int) -> float:
    """TD-MPC2's discount heuristic: (L/5 - 1) / (L/5), clipped to [0.95, 0.995]."""
    f = episode_macro_steps / 5
    return float(np.clip((f - 1) / f, 0.95, 0.995))


class SeqSampler:
    """Sequences of H+1 macro frames with frame-stacked 64x64 pixels, rewards, termination and validity.

    All demonstrations are concatenated into flat device tensors so a batch is gathered with index arithmetic.
    The macro action at frame f is the mean of the primitive actions in [f, min(f + k, L - 1)).
    """

    def __init__(self, demos, norm, k: int, H: int, stack: int, split: str, dev: str, res: int = 64):
        self.k, self.H, self.stack, self.dev = k, H, stack, dev
        sel = [d for d in demos if d.split == split]
        pix, state, macro, task, start = [], [], [], [], [0]
        for d in sel:
            img = torch.from_numpy(load_images(d, "agentview")).permute(0, 3, 1, 2).float()
            pix.append(F.interpolate(img, size=(res, res), mode="area").round().to(torch.uint8))
            state.append(torch.as_tensor(norm(d.state)))
            a, L = d.action[:-1], len(d.state)
            macro.append(torch.as_tensor(np.stack([a[f : min(f + k, L - 1)].mean(0) if f < L - 1 else np.zeros(7)
                                                   for f in range(L)])).float())
            task.append(d.task_id)
            start.append(start[-1] + L)
        self.pix, self.state, self.macro = torch.cat(pix).to(dev), torch.cat(state).to(dev), torch.cat(macro).to(dev)
        self.start = torch.tensor(start[:-1], device=dev)
        self.length = torch.tensor([len(d.state) for d in sel], device=dev)
        self.task = torch.tensor(task, device=dev)
        self.gen = torch.Generator(device=dev)

    def sample(self, B: int, rng: np.random.RandomState) -> dict[str, torch.Tensor]:
        k, H, dev = self.k, self.H, self.dev
        self.gen.manual_seed(int(rng.randint(2**31)))
        i = torch.randint(len(self.length), (B,), device=dev, generator=self.gen)
        L = self.length[i]
        t0 = (torch.rand(B, device=dev, generator=self.gen) * (L - 1).float()).long()
        raw = t0[:, None] + k * torch.arange(H + 1, device=dev)[None]  # (B, H+1)
        f = torch.minimum(raw, (L - 1)[:, None])
        g = self.start[i][:, None] + f  # global frame index
        stack = torch.stack([self.start[i][:, None] + (f - j * k).clamp_min(0) for j in range(self.stack - 1, -1, -1)], 2)
        pix = self.pix[stack].flatten(2, 3)  # (B, H+1, 3*stack, res, res)
        end = (raw[:, 1:] >= (L - 1)[:, None]).float().unsqueeze(-1)
        valid = (raw[:, :-1] < (L - 1)[:, None]).float().unsqueeze(-1)
        return {"pix": pix, "state": self.state[g], "action": self.macro[g[:, :-1]], "reward": end * valid,
                "done": end, "valid": valid, "task": self.task[i]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="results/checkpoints")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg["seed"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    demos = load_demos()
    norm = StateNorm.fit(demos)
    k, H = cfg["frameskip"], cfg["horizon"]
    train = SeqSampler(demos, norm, k, H, cfg["frame_stack"], "train", dev)
    held = SeqSampler(demos, norm, k, H, cfg["frame_stack"], "test", dev)
    gamma = heuristic_gamma(cfg["max_env_steps"] // k)
    model = TDMPC2(num_tasks=4, frames=cfg["frame_stack"]).to(dev)
    enc = list(model.encoder.parameters())
    rest = [p for n, p in model.named_parameters() if not n.startswith(("encoder", "pi", "target_Qs"))]
    opt = torch.optim.Adam([{"params": enc, "lr": cfg["lr"] * cfg["enc_lr_scale"]}, {"params": rest, "lr": cfg["lr"]}])
    pi_opt = torch.optim.Adam(model.pi.parameters(), lr=cfg["lr"], eps=1e-5)
    scale = RunningScale()
    rng = np.random.RandomState(cfg["seed"])
    n_params = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("target_Qs"))
    print(f"tdmpc2: {n_params / 1e6:.2f}M params, gamma={gamma:.3f}", flush=True)
    hist = {"step": [], "loss": [], "consistency": [], "reward": [], "value": [], "pi": [], "heldout_consistency": [],
            "time_s": []}
    t0, acc = time.time(), []
    for step in range(1, cfg["updates"] + 1):
        model.train()
        acc.append(tdmpc2_update(model, opt, pi_opt, scale, train.sample(cfg["batch_size"], rng), gamma))
        if step % cfg["log_every"] == 0:
            model.eval()
            with torch.no_grad():
                b = held.sample(512, rng)
                z = model.encoder(b["pix"][:, 0], b["state"][:, 0])
                tgt = model.encoder(b["pix"][:, 1:].flatten(0, 1), b["state"][:, 1:].flatten(0, 1)).view(512, H, -1)
                cons = 0.0
                for t in range(H):
                    z = model.next(z, b["action"][:, t], b["task"])
                    cons += float(((z - tgt[:, t]).pow(2).mean(-1, keepdim=True) * b["valid"][:, t]).mean())
            hist["step"].append(step)
            for key in ["loss", "consistency", "reward", "value", "pi"]:
                hist[key].append(float(np.mean([a[key] for a in acc])))
            hist["heldout_consistency"].append(cons / H)
            hist["time_s"].append(time.time() - t0)
            acc = []
            print(f"step {step} loss {hist['loss'][-1]:.4f} cons {hist['consistency'][-1]:.5f} rew "
                  f"{hist['reward'][-1]:.4f} val {hist['value'][-1]:.4f} held-out cons {cons / H:.5f} "
                  f"{time.time() - t0:.0f}s", flush=True)
    os.makedirs(args.out, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg, "state_norm": norm.state_dict(), "gamma": gamma,
                "params": n_params}, os.path.join(args.out, "tdmpc2.pt"))
    hist.update({"params": n_params, "gamma": gamma, "device": dev})
    json.dump(hist, open("results/logs/tdmpc2_history.json", "w"), indent=1)


if __name__ == "__main__":
    main()
