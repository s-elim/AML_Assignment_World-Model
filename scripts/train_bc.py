"""Train a behaviour-cloning policy (one model for all four tasks) by Gaussian negative log-likelihood.

    python scripts/train_bc.py --config configs/model/bc_resnet18.yaml
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
import yaml  # noqa: E402

from src.data.libero_data import StateNorm, feature_path, load_demos, load_images  # noqa: E402
from src.models.bc_policy import BCPolicy, random_shift  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def macro_labels(action: np.ndarray, k: int) -> np.ndarray:
    """Label for observation t: mean of the (up to) k demonstrated actions that follow it."""
    a = action[:-1]  # last row is padding
    return np.stack([a[t : min(t + k, len(a))].mean(0) for t in range(len(a))])


class FrameSet:
    def __init__(self, demos, norm, cfg, split, dev):
        sel = [d for d in demos if d.split == split]
        k = cfg["frameskip"]
        self.tokens = cfg["backbone"].endswith("_frozen")
        imgs, wrists, states, acts, tasks = [], [], [], [], []
        for d in sel:
            n = len(d.state) - 1
            if self.tokens:
                imgs.append(torch.load(feature_path(cfg["backbone"].split("_")[0], d))[:n])
            else:
                imgs.append(torch.from_numpy(load_images(d, "agentview")[:n]))
            if cfg.get("wrist"):
                wrists.append(torch.from_numpy(load_images(d, "wrist")[:n]))
            states.append(torch.from_numpy(norm(d.state[:n])))
            acts.append(torch.from_numpy(macro_labels(d.action, k)).float())
            tasks.append(torch.full((n,), d.task_id, dtype=torch.long))
        self.img = torch.cat(imgs).to(dev)
        self.wrist = torch.cat(wrists).to(dev) if wrists else None
        self.state, self.action, self.task = torch.cat(states).to(dev), torch.cat(acts).to(dev), torch.cat(tasks).to(dev)

    def __len__(self) -> int:
        return len(self.state)

    def obs(self, ids, augment: bool):
        img = self.img[ids]
        if augment and not self.tokens:
            img = random_shift(img)
        o = {"image": img, "state": self.state[ids], "task": self.task[ids]}
        if self.wrist is not None:
            w = self.wrist[ids]
            o["wrist"] = random_shift(w) if augment else w
        return o


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
    train, held = FrameSet(demos, norm, cfg, "train", dev), FrameSet(demos, norm, cfg, "test", dev)
    token_dim = train.img.shape[-1] if train.tokens else None
    model = BCPolicy(cfg["backbone"], wrist=cfg.get("wrist", False), token_dim=token_dim).to(dev)
    backbone_params = [p for n, p in model.named_parameters() if n.startswith(("trunk.net", "wrist_trunk"))]
    other = [p for n, p in model.named_parameters() if not n.startswith(("trunk.net", "wrist_trunk"))]
    opt = torch.optim.AdamW([{"params": other, "lr": cfg["lr"]},
                             {"params": backbone_params, "lr": cfg["lr"] * cfg.get("backbone_lr_scale", 1.0)}],
                            weight_decay=cfg["weight_decay"])
    bs, epochs = cfg["batch_size"], cfg["epochs"]
    spe = len(train) // bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in opt.param_groups],
                                                total_steps=spe * epochs, pct_start=0.05)
    rng = np.random.RandomState(cfg["seed"])
    hist = {"epoch": [], "train_nll": [], "heldout_nll": [], "heldout_action_mse": [], "time_s": []}
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{cfg['name']}: {n_params / 1e6:.2f}M trainable params, {len(train)} train frames", flush=True)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm, run = rng.permutation(len(train)), 0.0
        for b in range(spe):
            ids = torch.as_tensor(perm[b * bs : (b + 1) * bs], device=dev)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                loss = model.nll(train.obs(ids, True), train.action[ids])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += float(loss)
        model.eval()
        nll, mse = [], []
        with torch.no_grad():
            for b in range(0, len(held), 256):
                ids = torch.arange(b, min(b + 256, len(held)), device=dev)
                with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                    o = held.obs(ids, False)
                    nll.append(float(model.nll(o, held.action[ids])) * len(ids))
                    mse.append(float((model.act(o).float() - held.action[ids]).pow(2).mean()) * len(ids))
        hist["epoch"].append(ep + 1)
        hist["train_nll"].append(run / spe)
        hist["heldout_nll"].append(sum(nll) / len(held))
        hist["heldout_action_mse"].append(sum(mse) / len(held))
        hist["time_s"].append(time.time() - t0)
        print(f"epoch {ep + 1}/{epochs} train nll {hist['train_nll'][-1]:.3f} held-out nll "
              f"{hist['heldout_nll'][-1]:.3f} mse {hist['heldout_action_mse'][-1]:.4f} {time.time() - t0:.0f}s", flush=True)
    os.makedirs(args.out, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg, "state_norm": norm.state_dict(), "params": n_params,
                "token_dim": token_dim}, os.path.join(args.out, f"{cfg['name']}.pt"))
    hist.update({"params": n_params, "train_frames": len(train), "device": dev})
    json.dump(hist, open(f"results/logs/{cfg['name']}_history.json", "w"), indent=1)


if __name__ == "__main__":
    main()
