import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np

# Ensure mesa and OSMesa libraries are in library search path for headless rendering
osmesa_lib_path = "/home/selim/.local/share/mesa24-osmesa/lib"
claude_prism_lib = "/home/selim/.local/share/claude-prism/app/usr/lib"
cur_ld = os.environ.get("LD_LIBRARY_PATH", "")
if osmesa_lib_path not in cur_ld:
    os.environ["LD_LIBRARY_PATH"] = f"{osmesa_lib_path}:{claude_prism_lib}:{cur_ld}"
os.environ.setdefault("MUJOCO_GL", "osmesa")

# Add LIBERO to path if available
libero_path = "/nas/Selim/selim_sarowar/LIBERO-X"
if libero_path not in sys.path:
    sys.path.insert(0, libero_path)


class LiberoMuJoCoEnv:
    """
    Standardized Gymnasium-compatible wrapper for LIBERO simulation tasks under MuJoCo.
    Adheres to the official LIBERO evaluation protocol established in Lifelong Robot Learning (Liu et al., NeurIPS 2023)
    and aligned with the AGI programmatic closed-loop framework.
    """

    def __init__(
        self,
        suite_name: str = "libero_10",
        task_id: int = 0,
        image_size: int = 128,
        max_episode_steps: int = 100,
        seed: int = 0,
    ):
        self.suite_name = suite_name
        self.task_id = task_id
        self.image_size = image_size
        self.max_episode_steps = max_episode_steps
        self.seed = seed
        self._step_count = 0
        self._env = None
        self._task = None
        self._goal_image = None

        self._init_upstream_env()

    def _init_upstream_env(self):
        try:
            from libero.libero import benchmark, get_libero_path
            from libero.libero.envs import OffScreenRenderEnv

            task_suite = benchmark.get_benchmark_dict()[self.suite_name]()
            self._task = task_suite.get_task(self.task_id)
            bddl_dir = get_libero_path("bddl_files")
            bddl_full = os.path.join(bddl_dir, self._task.problem_folder, self._task.bddl_file)

            env_args = {
                "bddl_file_name": bddl_full,
                "camera_heights": self.image_size,
                "camera_widths": self.image_size,
            }
            self._env = OffScreenRenderEnv(**env_args)
            self._env.seed(self.seed)
        except Exception as e:
            # Fallback for environments without native headless display
            print(f"[Warning] Failed to instantiate upstream LIBERO OffScreenRenderEnv ({e}). Using mock mode.")
            self._env = None

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._step_count = 0
        effective_seed = seed if seed is not None else self.seed

        if self._env is not None:
            self._env.seed(effective_seed)
            obs = self._env.reset()
            img = obs.get("agentview_image", np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8))
            # Cache goal observation if available
            self._goal_image = img.copy()
            task_prompt = self._task.language if self._task is not None else ""
            info = {
                "task_prompt": task_prompt,
                "step": 0,
                "success": False,
                "state": obs.get("robot0_eef_pos", np.zeros(3, dtype=np.float32)),
            }
            return img, info

        # Fallback synthetic frame
        img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        self._goal_image = img.copy()
        return img, {"task_prompt": "libero_task", "step": 0, "success": False, "state": np.zeros(3, dtype=np.float32)}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self._step_count += 1
        if self._env is not None:
            # Pad or truncate action to 7-DoF [dx, dy, dz, droll, dpitch, dyaw, grip]
            if len(action) == 4:
                # Map 4D [dx, dy, dz, grip] to 7D
                action_7d = np.array([action[0], action[1], action[2], 0.0, 0.0, 0.0, action[3]], dtype=np.float32)
            else:
                action_7d = np.asarray(action, dtype=np.float32)

            obs, reward, done, info = self._env.step(action_7d)
            img = obs.get("agentview_image", np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8))
            success = bool(reward > 0.5 or info.get("success", False))
            terminated = success or done
            truncated = self._step_count >= self.max_episode_steps
            info["success"] = success
            info["state"] = obs.get("robot0_eef_pos", np.zeros(3, dtype=np.float32))
            info["goal_state"] = np.zeros(3, dtype=np.float32)
            return img, float(reward), terminated, truncated, info

        # Fallback
        img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        truncated = self._step_count >= self.max_episode_steps
        return img, 0.0, False, truncated, {"success": False, "state": np.zeros(3), "goal_state": np.zeros(3)}

    def get_goal_image(self) -> np.ndarray:
        if self._goal_image is not None:
            return self._goal_image
        return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

    def close(self):
        if self._env is not None:
            self._env.close()
