"""Gymnasium wrapper around four LIBERO-Goal tasks (LIBERO, Liu et al., NeurIPS 2023 Datasets and Benchmarks).

All four tasks share one kitchen scene, one Franka Panda and LIBERO's OSC_POSE controller, so a single
dataset and a single model can serve all of them. The wrapper adds three things the benchmark does not
provide: a fixed decision rate (one macro step = ``frameskip`` control steps under a zero-order hold),
episodes that start from a stored MuJoCo state, and per-task goal-error measurements in centimetres.

Action space (native LIBERO OSC_POSE, normalised to [-1, 1]^7): the first three entries scale to a
position offset of at most 0.05 m per control step, the next three to an axis-angle offset of at most
0.5 rad, and the last one is the gripper command (-1 open, +1 close).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIBERO_ROOT = os.environ.get("LIBERO_ROOT", "/nas/Selim/selim_sarowar/GST-VLA/LIBERO")
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.join(PROJECT_ROOT, "configs", "libero"))


def _prime_numba() -> None:
    """Compile one Numba kernel before MuJoCo loads OSMesa.

    The user-space Mesa build exports its own LLVM symbols; a first Numba compilation after it is loaded
    (robosuite compiles controller kernels on the first reset) segfaults. Compiling once beforehand binds
    Numba's LLVM first.
    """
    if "mujoco" in sys.modules or os.environ.get("MUJOCO_GL") != "osmesa":
        return
    import numba

    numba.njit(lambda x: x + 1)(1)


_prime_numba()
if LIBERO_ROOT not in sys.path:
    sys.path.insert(0, LIBERO_ROOT)

import gymnasium as gym  # noqa: E402
from gymnasium import spaces  # noqa: E402

SUITE = "libero_goal"


@dataclass(frozen=True)
class TaskSpec:
    key: str  # LIBERO task name (BDDL file stem)
    short: str  # label used in tables and figures
    skill: str  # skill category
    goal_body: str | None  # body whose final position defines the goal error, None = end effector
    goal_joint: str | None  # prismatic joint whose displacement defines the goal error
    contact_body: str  # body the gripper must touch, used to diagnose approach versus contact failures


TASKS: dict[str, TaskSpec] = {
    "turn_on_the_stove": TaskSpec(
        "turn_on_the_stove", "Stove", "reach and press", None, None, "flat_stove_1_button"
    ),
    "put_the_bowl_on_the_plate": TaskSpec(
        "put_the_bowl_on_the_plate", "Bowl-Plate", "pick and place", "akita_black_bowl_1_main", None,
        "akita_black_bowl_1_main"
    ),
    "push_the_plate_to_the_front_of_the_stove": TaskSpec(
        "push_the_plate_to_the_front_of_the_stove", "Push-Plate", "non-prehensile push", "plate_1_main", None,
        "plate_1_main"
    ),
    "open_the_middle_drawer_of_the_cabinet": TaskSpec(
        "open_the_middle_drawer_of_the_cabinet", "Drawer", "articulated", None, "wooden_cabinet_1_middle_level",
        "wooden_cabinet_1_cabinet_middle"
    ),
}
TASK_KEYS = list(TASKS)

ACTION_DIM = 7
STATE_DIM = 7  # end-effector position (3), axis-angle orientation (3), finger opening (1)


def proprio_from_obs(obs: dict[str, np.ndarray]) -> np.ndarray:
    """7-D proprioceptive state in the same convention as the LIBERO hdf5 files."""
    import robosuite.utils.transform_utils as T

    q = obs["robot0_gripper_qpos"]
    return np.concatenate(
        [obs["robot0_eef_pos"], T.quat2axisangle(obs["robot0_eef_quat"]), [q[0] - q[1]]]
    ).astype(np.float32)


def bddl_path(task: str) -> str:
    from libero.libero import get_libero_path

    return os.path.join(get_libero_path("bddl_files"), SUITE, f"{task}.bddl")


def dataset_path(task: str) -> str:
    from libero.libero import get_libero_path

    return os.path.join(get_libero_path("datasets"), SUITE, f"{task}_demo.hdf5")


class LiberoGoalEnv(gym.Env):
    """One LIBERO-Goal task behind the Gymnasium API, stepped at the macro-step rate."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, task: str, image_size: int = 128, frameskip: int = 5, max_env_steps: int = 300):
        super().__init__()
        if task not in TASKS:
            raise ValueError(f"unknown task {task!r}; choose from {TASK_KEYS}")
        from libero.libero.envs import OffScreenRenderEnv

        self.spec_ = TASKS[task]
        self.task = task
        self.image_size = image_size
        self.frameskip = frameskip
        self.max_env_steps = max_env_steps
        self._env = OffScreenRenderEnv(
            bddl_file_name=bddl_path(task), camera_heights=image_size, camera_widths=image_size
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32)
        img = spaces.Box(0, 255, shape=(image_size, image_size, 3), dtype=np.uint8)
        self.observation_space = spaces.Dict(
            {
                "agentview": img,
                "wrist": img,
                "state": spaces.Box(-np.inf, np.inf, shape=(STATE_DIM,), dtype=np.float32),
            }
        )
        self._env_steps = 0
        self._last_raw: dict[str, Any] = {}
        self.goal_reference: dict[str, np.ndarray] | None = None

    # ------------------------------------------------------------------ helpers
    @property
    def sim(self):
        return self._env.sim

    def _pack(self, raw: dict[str, Any]) -> dict[str, np.ndarray]:
        self._last_raw = raw
        return {
            "agentview": raw["agentview_image"].copy(),
            "wrist": raw["robot0_eye_in_hand_image"].copy(),
            "state": proprio_from_obs(raw),
        }

    def task_measurements(self) -> dict[str, np.ndarray]:
        """Positions used by the goal-error metric, read from the simulator."""
        m = self.sim.model
        out = {"ee": np.array(self._last_raw["robot0_eef_pos"], dtype=np.float64),
               "contact": np.array(self.sim.data.body_xpos[m.body_name2id(self.spec_.contact_body)])}
        if self.spec_.goal_body is not None:
            out["body"] = np.array(self.sim.data.body_xpos[m.body_name2id(self.spec_.goal_body)])
        if self.spec_.goal_joint is not None:
            adr = m.get_joint_qpos_addr(self.spec_.goal_joint)
            out["joint"] = np.array([self.sim.data.qpos[adr]])
        return out

    def goal_error_cm(self) -> float:
        """Distance (cm) between the current and the goal configuration of the task-relevant part.

        Body position for Bowl-Plate and Push-Plate, drawer-joint travel for Drawer, and the end effector for
        Stove, whose button does not translate.
        """
        if self.goal_reference is None:
            return float("nan")
        cur = self.task_measurements()
        if "joint" in cur:
            return float(100.0 * abs(cur["joint"][0] - self.goal_reference["joint"][0]))
        key = "body" if "body" in cur else "ee"
        return float(100.0 * np.linalg.norm(cur[key] - self.goal_reference[key]))

    # ------------------------------------------------------------------ API
    def set_goal_from_state(self, state: np.ndarray, last_action: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Load a goal MuJoCo state, record goal measurements and the goal observation, then leave the env dirty.

        Callers must ``reset`` afterwards. ``last_action`` replays the final demonstration action so that the
        goal frame matches the last stored observation of a LIBERO demonstration.
        """
        self._env.reset()
        raw = self._env.set_init_state(state)
        if last_action is not None:
            raw, _, _, _ = self._env.step(last_action)
        goal_obs = self._pack(raw)
        self.goal_reference = self.task_measurements()
        return goal_obs

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if options is None or "init_state" not in options:
            raise ValueError("LiberoGoalEnv.reset needs options={'init_state': mujoco_state}")
        self._env.reset()
        raw = self._env.set_init_state(options["init_state"])
        # LIBERO's evaluation protocol lets objects settle with zero actions before control starts.
        for _ in range(options.get("settle_steps", 0)):
            raw, _, _, _ = self._env.step(np.zeros(ACTION_DIM))
        self._env_steps = 0
        return self._pack(raw), {"success": bool(self._env.check_success()), "env_steps": 0}

    def step(self, action: np.ndarray):
        """Execute one macro step: ``frameskip`` control steps holding ``action`` constant."""
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        reward, success, raw = 0.0, False, None
        for _ in range(self.frameskip):
            raw, r, done, _ = self._env.step(action)
            self._env_steps += 1
            reward += float(r)
            if done:
                success = True
                break
            if self._env_steps >= self.max_env_steps:
                break
        obs = self._pack(raw)
        truncated = (not success) and self._env_steps >= self.max_env_steps
        info = {"success": success, "env_steps": self._env_steps, "goal_error_cm": self.goal_error_cm()}
        return obs, reward, success, truncated, info

    def render(self) -> np.ndarray:
        return self._last_raw["agentview_image"].copy()

    def close(self) -> None:
        self._env.close()
