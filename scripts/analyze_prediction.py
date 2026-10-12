"""Open-loop multi-step prediction error of the world models on the held-out demonstrations.

From every macro step of every held-out demonstration, each model rolls out k = 1..10 macro steps under the
demonstrated actions, as it does at planning time. For each k the script reports the L1 and MSE between
imagined and encoded tokens and their cosine similarity, the L1 and MSE of the copy-last baseline
(z_{t+k} = z_t), and the same metrics when the actions are taken from a randomly permuted window (action
shuffle). TD-MPC2 is scored in its own SimNorm latent space.

Usage:
    python scripts/analyze_prediction.py --out results/tables/prediction_error.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.data.libero_data import StateNorm, feature_path, load_demos, load_images  # noqa: E402
from src.models.dino_wm import DinoWM  # noqa: E402
from src.models.tdmpc2 import TDMPC2  # noqa: E402
from src.models.vjepa2_ac import VJEPA2AC  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

K_MAX = 10


def windows(demos, k: int):
    """(demo, start) pairs with K_MAX macro steps of future inside the demonstration."""
    return [(d, t) for d in demos for t in range(0, len(d.state) - K_MAX * k, k)]


def macro_actions(d, t: int, k: int, n: int) -> np.ndarray:
    a = d.action
    return np.stack([a[t + j * k : t + (j + 1) * k].mean(0) for j in range(n)])


def summarise(pred, true, z0):
    """pred/true (N, K, P, D), z0 (N, P, D) -> per-k metric lists."""
    out = {"l1": [], "mse": [], "cos": [], "copy_l1": [], "copy_mse": []}
    for j in range(pred.size(1)):
        p, t = pred[:, j], true[:, j]
        out["l1"].append(float((p - t).abs().mean()))
        out["mse"].append(float((p - t).pow(2).mean()))
        out["cos"].append(float(F.cosine_similarity(p.flatten(1), t.flatten(1), dim=1).mean()))
        out["copy_l1"].append(float((z0 - t).abs().mean()))
        out["copy_mse"].append(float((z0 - t).pow(2).mean()))
    return out


@torch.no_grad()
def eval_latent(name: str, demos, dev: str, rng: np.random.RandomState) -> dict:
    ck = torch.load(f"results/checkpoints/{name}.pt", map_location="cpu", weights_only=False)
    cfg, k = ck["config"], ck["config"]["frameskip"]
    norm = StateNorm.from_dict(ck["state_norm"])
    if cfg["model_type"] == "vjepa2_ac":
        m = VJEPA2AC(embed_dim=cfg["embed_dim"], pred_dim=cfg["pred_dim"], depth=cfg["depth"])
    else:
        m = DinoWM(visual_dim=cfg["embed_dim"], frameskip=k, num_hist=cfg["num_hist"], depth=cfg["depth"],
                   heads=cfg["heads"], mlp=cfg["mlp_dim"])
    m.load_state_dict(ck["model"])
    m.to(dev).eval()
    feats = {d.path: torch.load(feature_path(cfg["encoder"], d)).float() for d in demos}
    W = windows(demos, k)
    perm = rng.permutation(len(W))
    res = {}
    for mode in ["true", "shuffled"]:
        preds, trues, z0s = [], [], []
        for b in range(0, len(W), 64):
            batch = W[b : b + 64]
            z0 = torch.stack([feats[d.path][t] for d, t in batch]).to(dev)
            s0 = torch.stack([torch.as_tensor(norm(d.state[t])) for d, t in batch]).to(dev).float()
            src = batch if mode == "true" else [W[perm[(b + i) % len(W)]] for i in range(len(batch))]
            acts = torch.stack([torch.as_tensor(macro_actions(d, t, k, K_MAX)) for d, t in src]).to(dev).float()
            true = torch.stack([feats[d.path][t + k : t + (K_MAX + 1) * k : k] for d, t in batch]).to(dev)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                if cfg["model_type"] == "vjepa2_ac":
                    pred = m.rollout(z0, s0, acts)
                else:
                    pred = m.rollout(z0[:, None], s0[:, None], torch.zeros(len(batch), 0, 7 * k, device=dev), acts)
            preds.append(pred.float().cpu()); trues.append(true.cpu()); z0s.append(z0.cpu())
        res[mode] = summarise(torch.cat(preds), torch.cat(trues), torch.cat(z0s))
    res["windows"] = len(W)
    return res


@torch.no_grad()
def eval_tdmpc2(demos, dev: str, rng: np.random.RandomState) -> dict:
    ck = torch.load("results/checkpoints/tdmpc2.pt", map_location="cpu", weights_only=False)
    cfg, k, S = ck["config"], ck["config"]["frameskip"], ck["config"]["frame_stack"]
    norm = StateNorm.from_dict(ck["state_norm"])
    m = TDMPC2(num_tasks=4, frames=S)
    m.load_state_dict(ck["model"])
    m.to(dev).eval()
    pix = {}
    for d in demos:
        img = torch.from_numpy(load_images(d, "agentview")).permute(0, 3, 1, 2).float()
        pix[d.path] = F.interpolate(img, size=(64, 64), mode="area").round().to(torch.uint8)

    def enc(d, f):
        idx = [max(0, f - j * k) for j in range(S - 1, -1, -1)]
        x = pix[d.path][idx].flatten(0, 1)[None].to(dev)
        return m.encoder(x, torch.as_tensor(norm(d.state[f]), device=dev).float()[None])[0]

    W = windows(demos, k)
    perm = rng.permutation(len(W))
    res = {}
    for mode in ["true", "shuffled"]:
        preds, trues, z0s = [], [], []
        for i, (d, t) in enumerate(W):
            z = z0 = enc(d, t)
            src = (d, t) if mode == "true" else W[perm[i]]
            acts = torch.as_tensor(macro_actions(src[0], src[1], k, K_MAX), device=dev).float()
            tid = torch.tensor(d.task_id, device=dev)
            seq = []
            for j in range(K_MAX):
                z = m.next(z[None], acts[j][None], tid[None])[0]
                seq.append(z)
            preds.append(torch.stack(seq)[:, None, :].cpu())
            trues.append(torch.stack([enc(d, t + (j + 1) * k) for j in range(K_MAX)])[:, None, :].cpu())
            z0s.append(z0[None].cpu())
        res[mode] = summarise(torch.stack(preds), torch.stack(trues), torch.stack(z0s))
    res["windows"] = len(W)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/tables/prediction_error.json")
    args = ap.parse_args()
    set_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    demos = [d for d in load_demos() if d.split == "test"]
    out = {"vjepa2_ac": eval_latent("vjepa2_ac", demos, dev, np.random.RandomState(0)),
           "dino_wm": eval_latent("dino_wm", demos, dev, np.random.RandomState(0)),
           "tdmpc2": eval_tdmpc2(demos, dev, np.random.RandomState(0))}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    for name, r in out.items():
        t, s = r["true"], r["shuffled"]
        print(name, "k=1,3,5,10 L1/copy:", [round(t["l1"][j] / t["copy_l1"][j], 3) for j in (0, 2, 4, 9)],
              "shuffled:", [round(s["l1"][j] / s["copy_l1"][j], 3) for j in (0, 2, 4, 9)])


if __name__ == "__main__":
    main()
