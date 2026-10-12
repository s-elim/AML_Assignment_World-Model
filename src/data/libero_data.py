"""Loading the regenerated LIBERO-Goal demonstrations and cutting them into macro-step clips.

A macro step spans ``frameskip`` control steps (5 steps = 0.25 s at LIBERO's 20 Hz, i.e. the 4 frames per
second used by V-JEPA 2-AC). The macro action is the mean of the primitive actions it spans (the interface
the policies execute under a zero-order hold); DINO-WM additionally receives the concatenation of the
primitive actions. Every offset in [0, frameskip) yields a distinct subsequence, so a demonstration of
length L gives about L - (T - 1) * frameskip clips of T frames.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

import numpy as np
import torch

from src.env.libero_env import TASK_KEYS

DATA_DIR = "data/libero_goal4"
FEAT_DIR = "data/features"


@dataclass
class Demo:
    task: str
    task_id: int
    index: int
    split: str  # "train" or "test"
    state: np.ndarray  # (L, 7)
    action: np.ndarray  # (L, 7); action[t] follows observation t, last row is padding
    sim_state: np.ndarray  # (L, 79)
    path: str


def load_meta(task: str, data_dir: str = DATA_DIR) -> dict:
    with open(os.path.join(data_dir, f"{task}__meta.json")) as fh:
        return json.load(fh)


def load_demos(data_dir: str = DATA_DIR, tasks: list[str] | None = None, images: bool = False) -> list[Demo]:
    demos = []
    for tid, task in enumerate(TASK_KEYS):
        if tasks is not None and task not in tasks:
            continue
        meta = load_meta(task, data_dir)
        for path in sorted(glob.glob(os.path.join(data_dir, f"{task}__demo*.npz"))):
            idx = int(path.split("demo")[-1][:2])
            with np.load(path) as d:
                demo = Demo(task, tid, idx, "test" if idx in meta["test_demos"] else "train",
                            d["state"], d["action"], d["sim_state"], path)
                if images:
                    demo.agentview = d["agentview"]  # type: ignore[attr-defined]
                    demo.wrist = d["wrist"]  # type: ignore[attr-defined]
            demos.append(demo)
    return demos


def load_images(demo: Demo, key: str = "agentview") -> np.ndarray:
    with np.load(demo.path) as d:
        return d[key]


def feature_path(encoder: str, demo: Demo, feat_dir: str = FEAT_DIR) -> str:
    return os.path.join(feat_dir, encoder, f"{demo.task}__demo{demo.index:02d}.pt")


def macro_action(action: np.ndarray, t: int, k: int) -> np.ndarray:
    """Mean of the k primitive actions following observation t."""
    return action[t : t + k].mean(0)


def concat_action(action: np.ndarray, t: int, k: int) -> np.ndarray:
    """Concatenation of the k primitive actions following observation t (DINO-WM's frameskip input)."""
    return action[t : t + k].reshape(-1)


class StateNorm:
    """Per-dimension standardisation of the 7-D proprioceptive state, fitted on training demonstrations."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean, self.std = mean.astype(np.float32), np.maximum(std, 1e-3).astype(np.float32)

    @classmethod
    def fit(cls, demos: list[Demo]) -> "StateNorm":
        s = np.concatenate([d.state for d in demos if d.split == "train"])
        return cls(s.mean(0), s.std(0))

    def __call__(self, s):
        if isinstance(s, torch.Tensor):
            return (s - torch.as_tensor(self.mean, device=s.device)) / torch.as_tensor(self.std, device=s.device)
        return (s - self.mean) / self.std

    def state_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "StateNorm":
        return cls(np.array(d["mean"]), np.array(d["std"]))


def clip_index(demos: list[Demo], T: int, k: int, split: str) -> list[tuple[int, int]]:
    """(demo position, start step) for every clip of T macro frames that fits inside a demonstration."""
    out = []
    for i, d in enumerate(demos):
        if d.split != split:
            continue
        L = len(d.state)
        for t0 in range(0, L - (T - 1) * k):
            out.append((i, t0))
    return out
