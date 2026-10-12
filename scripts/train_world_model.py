"""Train an action-conditioned latent predictor on cached frozen-encoder features (one model for all 4 tasks).

    python scripts/train_world_model.py --config configs/model/vjepa2_ac.yaml
    python scripts/train_world_model.py --config configs/model/dino_wm.yaml

No checkpoint selection: the final checkpoint is used. The held-out loss is computed on the 10 test
demonstrations per task for monitoring only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from src.data.libero_data import StateNorm, clip_index, feature_path, load_demos  # noqa: E402
from src.models.dino_wm import DinoWM  # noqa: E402
from src.models.vjepa2_ac import VJEPA2AC  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


class ClipSampler:
    """Holds all features on the device and assembles (z, s, a) clip batches on the fly."""

    def __init__(self, demos, feats, norm: StateNorm, T: int, k: int, split: str, device: str):
        self.T, self.k, self.device = T, k, device
        self.index = clip_index(demos, T, k, split)
        self.feats = feats
        pad = np.zeros((k, 7), np.float32)
        self.states = [torch.as_tensor(norm(d.state), device=device) for d in demos]
        self.actions = [torch.as_tensor(np.concatenate([d.action, pad]), device=device) for d in demos]

    def __len__(self) -> int:
        return len(self.index)

    def batch(self, ids: np.ndarray, concat_actions: bool):
        T, k = self.T, self.k
        z, s, a = [], [], []
        for j in ids:
            i, t0 = self.index[j]
            steps = torch.arange(T, device=self.device) * k + t0
            z.append(self.feats[i][steps])
            s.append(self.states[i][steps])
            win = self.actions[i][t0 : t0 + T * k].view(T, k, 7)
            a.append(win.reshape(T, k * 7) if concat_actions else win.mean(1)[: T - 1])
        return torch.stack(z).float(), torch.stack(s), torch.stack(a)


def lr_at(step: int, total: int, warmup: int, peak: float, start: float) -> float:
    if step < warmup:
        return start + (peak - start) * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * peak * (1 + math.cos(math.pi * p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="results/checkpoints")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg["seed"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    name, kind = cfg["name"], cfg["model_type"]

    demos = load_demos()
    norm = StateNorm.fit(demos)
    feats = [torch.load(feature_path(cfg["encoder"], d)).to(dev) for d in demos]
    k = cfg["frameskip"]
    T = cfg["clip_frames"]
    train = ClipSampler(demos, feats, norm, T, k, "train", dev)
    held = ClipSampler(demos, feats, norm, T, k, "test", dev)
    concat = kind == "dino_wm"

    if kind == "vjepa2_ac":
        model = VJEPA2AC(embed_dim=cfg["embed_dim"], pred_dim=cfg["pred_dim"], depth=cfg["depth"],
                         auto_steps=cfg["auto_steps"]).to(dev)
        # Fit the planning-time state integrator on all training macro transitions.
        s0, a0, s1 = [], [], []
        for d in demos:
            if d.split != "train":
                continue
            sn = norm(d.state)
            for t in range(0, len(sn) - k):
                s0.append(sn[t]); a0.append(d.action[t : t + k].mean(0)); s1.append(sn[t + k])
        r2 = model.integrator.fit(torch.tensor(np.array(s0)), torch.tensor(np.array(a0)), torch.tensor(np.array(s1)))
        model.integrator.to(dev)
    elif kind == "dino_wm":
        model = DinoWM(visual_dim=cfg["embed_dim"], frameskip=k, num_hist=cfg["num_hist"], depth=cfg["depth"],
                       heads=cfg["heads"], dim_head=64, mlp=cfg["mlp_dim"]).to(dev)
        r2 = None
    else:
        raise ValueError(kind)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"], betas=(0.9, 0.999))
    bs, epochs = cfg["batch_size"], cfg["epochs"]
    steps_per_epoch = len(train) // bs
    total = steps_per_epoch * epochs
    warmup = int(cfg["warmup_frac"] * total)
    rng = np.random.RandomState(cfg["seed"])
    hist = {"epoch": [], "train_loss": [], "heldout_loss": [], "lr": [], "time_s": []}
    t0, step = time.time(), 0
    print(f"{name}: {n_params / 1e6:.2f}M params, {len(train)} train clips, {len(held)} held-out clips, "
          f"{total} steps, state-integrator R2={r2}", flush=True)
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(train))
        run = 0.0
        for b in range(steps_per_epoch):
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, warmup, cfg["lr"], cfg["lr"] * cfg["start_lr_frac"])
            z, s, a = train.batch(perm[b * bs : (b + 1) * bs], concat)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                out = model.loss(z, a, s) if kind == "vjepa2_ac" else model.loss(z, s, a)
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
            opt.step()
            run += float(out["loss"])
            step += 1
        model.eval()
        with torch.no_grad():
            hl, ids = [], np.arange(len(held))
            for b in range(0, len(held), bs):
                z, s, a = held.batch(ids[b : b + bs], concat)
                with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                    o = model.loss(z, a, s) if kind == "vjepa2_ac" else model.loss(z, s, a)
                hl.append(float(o["loss"]) * len(z))
        hist["epoch"].append(ep + 1)
        hist["train_loss"].append(run / steps_per_epoch)
        hist["heldout_loss"].append(sum(hl) / len(held))
        hist["lr"].append(opt.param_groups[0]["lr"])
        hist["time_s"].append(time.time() - t0)
        print(f"epoch {ep + 1}/{epochs} train {hist['train_loss'][-1]:.4f} held-out {hist['heldout_loss'][-1]:.4f} "
              f"{time.time() - t0:.0f}s", flush=True)

    os.makedirs(args.out, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg, "state_norm": norm.state_dict(),
                "params": n_params, "integrator_r2": r2}, os.path.join(args.out, f"{name}.pt"))
    os.makedirs("results/logs", exist_ok=True)
    hist.update({"params": n_params, "integrator_r2": r2, "train_clips": len(train), "device": dev,
                 "device_name": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"})
    json.dump(hist, open(f"results/logs/{name}_history.json", "w"), indent=1)


if __name__ == "__main__":
    main()
