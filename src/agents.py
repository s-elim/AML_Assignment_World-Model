"""Agents that map an observation (and the goal observation) to a 7-D macro action.

Every agent receives the same per-episode inputs: the current agent-view frame, the 7-D proprioceptive
state and the task identity, plus the goal frame. The latent planners use the goal frame; BC and TD-MPC2
use the task identity, which LIBERO-Goal ties to one goal predicate.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from src.data.libero_data import StateNorm
from src.models.bc_policy import BCPolicy
from src.models.dino_wm import DinoWM
from src.models.encoders import build_encoder
from src.models.tdmpc2 import MPPI, TDMPC2
from src.models.vjepa2_ac import VJEPA2AC
from src.planning.cem import CEM, CEMConfig
from src.planning.costs import build_cost


def _sync(dev: str) -> None:
    if dev == "cuda":
        torch.cuda.synchronize()


class Timer:
    def __init__(self, dev: str):
        self.dev, self.t = dev, {}

    def __call__(self, key: str):
        timer = self

        class _Ctx:
            def __enter__(self):
                _sync(timer.dev)
                self.t0 = time.perf_counter()

            def __exit__(self, *a):
                _sync(timer.dev)
                timer.t[key] = timer.t.get(key, 0.0) + 1e3 * (time.perf_counter() - self.t0)

        return _Ctx()


class LatentPlannerAgent:
    """V-JEPA 2-AC or DINO-WM with CEM over imagined latent rollouts."""

    def __init__(self, ckpt: dict, cem_cfg: CEMConfig, cost: str, dev: str, seed: int = 0,
                 channel_var: torch.Tensor | None = None):
        cfg = ckpt["config"]
        self.kind, self.dev, self.seed = cfg["model_type"], dev, seed
        self.norm = StateNorm.from_dict(ckpt["state_norm"])
        self.encoder = build_encoder(cfg["encoder"]).to(dev)
        if self.kind == "vjepa2_ac":
            self.model = VJEPA2AC(embed_dim=cfg["embed_dim"], pred_dim=cfg["pred_dim"], depth=cfg["depth"])
        else:
            self.model = DinoWM(visual_dim=cfg["embed_dim"], frameskip=cfg["frameskip"], num_hist=cfg["num_hist"],
                                depth=cfg["depth"], heads=cfg["heads"], mlp=cfg["mlp_dim"])
        self.model.load_state_dict(ckpt["model"])
        self.model.to(dev).eval()
        self.frameskip = cfg["frameskip"]
        self.cem_cfg = cem_cfg
        self.cost = build_cost(cost, None if channel_var is None else channel_var.to(dev))
        self.cost_name = cost

    def encode(self, img: np.ndarray) -> torch.Tensor:
        with torch.autocast(self.dev, dtype=torch.bfloat16, enabled=self.dev == "cuda"):
            return self.encoder(torch.from_numpy(img[None]).to(self.dev)).float()

    def reset(self, task_id: int, goal_obs: dict) -> None:
        self.cem = CEM(self.cem_cfg, 7, self.dev, self.seed)
        self.z_goal = self.encode(goal_obs["agentview"])
        self.hist_z, self.hist_s, self.hist_a = [], [], []

    def imagine(self, actions: torch.Tensor) -> torch.Tensor:
        """Imagined frames for action sequences (S, H, 7) from the current context -> (S, H, P, D)."""
        S = actions.shape[0]
        with torch.autocast(self.dev, dtype=torch.bfloat16, enabled=self.dev == "cuda"):
            if self.kind == "vjepa2_ac":
                z = self.model.rollout(self.z0.expand(S, -1, -1), self.s0.expand(S, -1), actions)
            else:
                z = self.model.rollout(self.zh.expand(S, -1, -1, -1), self.sh.expand(S, -1, -1),
                                       self.ah.expand(S, -1, -1), actions)
        return z.float()

    def costs(self, actions: torch.Tensor) -> torch.Tensor:
        return self.cost(self.imagine(actions)[:, -1], self.z_goal)

    @torch.no_grad()
    def act(self, obs: dict, record: bool = False) -> tuple[np.ndarray, dict]:
        timer = Timer(self.dev)
        with timer("encode_ms"):
            z = self.encode(obs["agentview"])
        s = torch.as_tensor(self.norm(obs["state"]), device=self.dev).float()[None]
        self.hist_z.append(z[0])
        self.hist_s.append(s[0])
        self.z0, self.s0 = z, s
        if self.kind == "dino_wm":
            h = self.model.num_hist
            self.zh = torch.stack(self.hist_z[-h:])[None]
            self.sh = torch.stack(self.hist_s[-h:])[None]
            a_prev = self.hist_a[-(self.zh.size(1) - 1):] if self.zh.size(1) > 1 else []
            self.ah = (torch.stack(a_prev)[None] if a_prev else torch.zeros(1, 0, 7 * self.frameskip, device=self.dev))
        with timer("plan_ms"):
            plan = self.cem.plan(self.costs, record=record)
        with timer("extract_ms"):
            a = plan[0].clamp(-1, 1)
            a_np = a.cpu().numpy()
        self.hist_a.append(a.repeat(self.frameskip))
        info = dict(timer.t)
        if record:
            imag = self.imagine(plan[None])[0]  # (H, P, D)
            info["imagined_pooled"] = imag.mean(1).cpu().numpy()
            info["imagined_next"] = imag[0].cpu().numpy().astype(np.float16)
            info["realized_pooled"] = z[0].mean(0).cpu().numpy()
            info["realized"] = z[0].cpu().numpy().astype(np.float16)
            info["plan_cost"] = float(self.cost(imag[-1:], self.z_goal)[0])
            info["goal_cost_now"] = float(self.cost(z, self.z_goal)[0])
            info["cem_trace"] = [{"mean0": t["mean"][0].numpy(), "best_cost": t["best_cost"]} for t in self.cem.trace]
            info["plan"] = plan.cpu().numpy()
        return a_np, info


class TDMPC2Agent:
    def __init__(self, ckpt: dict, dev: str, seed: int = 0):
        cfg = ckpt["config"]
        self.dev, self.k, self.stack = dev, cfg["frameskip"], cfg["frame_stack"]
        self.norm = StateNorm.from_dict(ckpt["state_norm"])
        self.model = TDMPC2(num_tasks=4, frames=self.stack)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(dev).eval()
        self.gamma, self.seed = ckpt["gamma"], seed

    def reset(self, task_id: int, goal_obs: dict) -> None:
        torch.manual_seed(self.seed)
        self.mppi = MPPI(self.model, self.gamma, seed=self.seed)
        self.tid = torch.tensor([task_id], device=self.dev)
        self.frames: list[torch.Tensor] = []

    @torch.no_grad()
    def act(self, obs: dict, record: bool = False) -> tuple[np.ndarray, dict]:
        timer = Timer(self.dev)
        with timer("encode_ms"):
            img = torch.from_numpy(obs["agentview"]).to(self.dev).permute(2, 0, 1)[None].float()
            img = F.interpolate(img, size=(64, 64), mode="area").round().to(torch.uint8)[0]
            self.frames.append(img)
            fr = [self.frames[max(0, len(self.frames) - 1 - j)] for j in range(self.stack - 1, -1, -1)]
            s = torch.as_tensor(self.norm(obs["state"]), device=self.dev).float()[None]
            z = self.model.encoder(torch.cat(fr)[None], s)
        with timer("plan_ms"):
            a = self.mppi.plan(z, self.tid)
        with timer("extract_ms"):
            a_np = a.cpu().numpy()
        return a_np, dict(timer.t)


class BCAgent:
    def __init__(self, ckpt: dict, dev: str):
        cfg = ckpt["config"]
        self.dev, self.backbone = dev, cfg["backbone"]
        self.norm = StateNorm.from_dict(ckpt["state_norm"])
        self.model = BCPolicy(cfg["backbone"], wrist=cfg.get("wrist", False), token_dim=ckpt.get("token_dim"))
        self.model.load_state_dict(ckpt["model"])
        self.model.to(dev).eval()
        self.encoder = build_encoder(self.backbone.split("_")[0]).to(dev) if self.backbone.endswith("_frozen") else None
        self.wrist = cfg.get("wrist", False)

    def reset(self, task_id: int, goal_obs: dict) -> None:
        self.tid = torch.tensor([task_id], device=self.dev)

    @torch.no_grad()
    def act(self, obs: dict, record: bool = False) -> tuple[np.ndarray, dict]:
        timer = Timer(self.dev)
        with timer("encode_ms"):
            img = torch.from_numpy(obs["agentview"][None]).to(self.dev)
            if self.encoder is not None:
                with torch.autocast(self.dev, dtype=torch.bfloat16, enabled=self.dev == "cuda"):
                    img = self.encoder(img).float()
        o = {"image": img, "state": torch.as_tensor(self.norm(obs["state"]), device=self.dev).float()[None],
             "task": self.tid}
        if self.wrist:
            o["wrist"] = torch.from_numpy(obs["wrist"][None]).to(self.dev)
        with timer("plan_ms"):
            a = self.model.act(o)[0]
        with timer("extract_ms"):
            a_np = a.cpu().numpy()
        return a_np, dict(timer.t)
