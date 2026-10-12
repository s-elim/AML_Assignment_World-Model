"""Cache the 8x8 token grid of a frozen encoder for every frame of the dataset.

Usage:
    python scripts/encode_features.py --encoder vjepa2   # V-JEPA 2 ViT-L/16, (L, 64, 1024) per demo
    python scripts/encode_features.py --encoder dinov2   # DINOv2 ViT-S/14, (L, 64, 384) per demo
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from src.data.libero_data import feature_path, load_demos, load_images  # noqa: E402
from src.models.encoders import build_encoder  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=["vjepa2", "dinov2"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--camera", default="agentview", choices=["agentview", "wrist"])
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = build_encoder(args.encoder).to(dev)
    demos = load_demos()
    t0, n = time.time(), 0
    for d in demos:
        out = feature_path(args.encoder if args.camera == "agentview" else f"{args.encoder}_{args.camera}", d)
        if os.path.exists(out):
            continue
        frames = torch.from_numpy(load_images(d, args.camera))
        feats = []
        for i in range(0, len(frames), args.batch):
            with torch.autocast(dev, dtype=torch.bfloat16):
                feats.append(enc(frames[i : i + args.batch].to(dev)).float().cpu())
        os.makedirs(os.path.dirname(out), exist_ok=True)
        torch.save(torch.cat(feats).half(), out)
        n += len(frames)
    print(f"{args.encoder}: encoded {n} frames in {time.time() - t0:.0f}s on {dev}")


if __name__ == "__main__":
    main()
