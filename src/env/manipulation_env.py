from typing import Tuple, Dict, Any, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.env.camera_utils import render_scene


SUPPORTED_TASKS = ["reach", "push", "pick_place", "drawer_open"]


class ManipulationEnv(gym.Env):
    """
    Standardized robotic manipulation benchmark environment for visual world model evaluation.
    Conforms to Gymnasium API with continuous 4-DoF end-effector control (dx, dy, dz, gripper)
    and RGB camera observations.
    """

    def __init__(
        self,
        task_name: str = "reach",
        image_size: int = 224,
        max_episode_steps: int = 150,
        seed: Optional[int] = None,
    ):
        super().__init__()
        assert task_name in SUPPORTED_TASKS, f"Task '{task_name}' not supported. Options: {SUPPORTED_TASKS}"
        self.task_name = task_name
        self.image_size = image_size
        self.max_episode_steps = max_episode_steps

        # Table bounds: x in [-0.25, 0.25], y in [-0.15, 0.25]
        self.table_bounds = (-0.25, 0.25, -0.15, 0.25)

        # Action: [dx, dy, dz, gripper_action]
        # dx, dy, dz in [-0.05, 0.05], gripper in [-1.0, 1.0] (negative = open, positive = close)
        self.action_space = spaces.Box(
            low=np.array([-0.05, -0.05, -0.05, -1.0], dtype=np.float32),
            high=np.array([0.05, 0.05, 0.05, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation: RGB image (H, W, 3)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8
        )

        self._rng = np.random.RandomState(seed if seed is not None else 42)
        self._step_count = 0

        # Physical states
        self.ee_pos = np.array([0.0, -0.05, 0.18], dtype=np.float32)
        self.gripper_aperture = 1.0  # 1.0 = fully open, 0.0 = fully closed
        self.object_pos = np.zeros(3, dtype=np.float32)
        self.goal_pos = np.zeros(3, dtype=np.float32)
        self.drawer_pos = np.zeros(3, dtype=np.float32)
        self.drawer_open_dist = 0.0
        self.grasped = False

        self.reset()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.RandomState(seed)

        self._step_count = 0
        self.grasped = False

        # Reset robot arm above table
        self.ee_pos = np.array(
            [
                self._rng.uniform(-0.08, 0.08),
                self._rng.uniform(-0.10, -0.02),
                0.18,
            ],
            dtype=np.float32,
        )
        self.gripper_aperture = 1.0

        if self.task_name == "reach":
            self.goal_pos = np.array(
                [
                    self._rng.uniform(-0.15, 0.15),
                    self._rng.uniform(0.02, 0.18),
                    self._rng.uniform(0.04, 0.16),
                ],
                dtype=np.float32,
            )
            self.object_pos = None
            self.drawer_pos = None

        elif self.task_name == "push":
            self.object_pos = np.array(
                [
                    self._rng.uniform(-0.10, 0.10),
                    self._rng.uniform(-0.02, 0.06),
                    0.02,  # Resting on table
                ],
                dtype=np.float32,
            )
            self.goal_pos = np.array(
                [
                    self._rng.uniform(-0.15, 0.15),
                    self._rng.uniform(0.12, 0.20),
                    0.02,
                ],
                dtype=np.float32,
            )
            self.drawer_pos = None

        elif self.task_name == "pick_place":
            self.object_pos = np.array(
                [
                    self._rng.uniform(-0.10, 0.10),
                    self._rng.uniform(-0.04, 0.04),
                    0.02,
                ],
                dtype=np.float32,
            )
            self.goal_pos = np.array(
                [
                    self._rng.uniform(-0.16, 0.16),
                    self._rng.uniform(0.10, 0.18),
                    0.08,  # Raised target height
                ],
                dtype=np.float32,
            )
            self.drawer_pos = None

        elif self.task_name == "drawer_open":
            self.drawer_pos = np.array([0.0, 0.15, 0.08], dtype=np.float32)
            self.drawer_open_dist = 0.0
            self.goal_pos = np.array([0.0, 0.05, 0.08], dtype=np.float32)  # Target open pos
            self.object_pos = None

        obs = self.render()
        info = self._get_info()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self._step_count += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Update end-effector position
        self.ee_pos[:3] += action[:3]
        # Keep EE within reasonable workspace bounds
        self.ee_pos[0] = np.clip(self.ee_pos[0], self.table_bounds[0], self.table_bounds[1])
        self.ee_pos[1] = np.clip(self.ee_pos[1], self.table_bounds[2], self.table_bounds[3])
        self.ee_pos[2] = np.clip(self.ee_pos[2], 0.01, 0.35)

        # Update gripper aperture
        grip_cmd = action[3]
        if grip_cmd > 0.0:
            self.gripper_aperture = max(0.0, self.gripper_aperture - 0.2)
        else:
            self.gripper_aperture = min(1.0, self.gripper_aperture + 0.2)

        # Physics simulation per task
        if self.task_name == "reach":
            dist = np.linalg.norm(self.ee_pos - self.goal_pos)
            success = dist < 0.035
            reward = -float(dist) + (10.0 if success else 0.0)

        elif self.task_name == "push":
            ee_to_obj = np.linalg.norm(self.ee_pos[:2] - self.object_pos[:2])
            # Contact mechanics: if EE touches object and is near table, displace object
            if ee_to_obj < 0.04 and self.ee_pos[2] < 0.06:
                push_dir = self.object_pos[:2] - self.ee_pos[:2]
                norm_push = np.linalg.norm(push_dir)
                if norm_push > 1e-4:
                    self.object_pos[:2] += (push_dir / norm_push) * 0.02
                else:
                    self.object_pos[:2] += action[:2] * 0.8
            self.object_pos[0] = np.clip(self.object_pos[0], self.table_bounds[0], self.table_bounds[1])
            self.object_pos[1] = np.clip(self.object_pos[1], self.table_bounds[2], self.table_bounds[3])

            dist = np.linalg.norm(self.object_pos[:2] - self.goal_pos[:2])
            success = dist < 0.04
            reward = -float(dist) + (10.0 if success else 0.0)

        elif self.task_name == "pick_place":
            ee_to_obj = np.linalg.norm(self.ee_pos - self.object_pos)
            # Grasp condition: close proximity + closed gripper
            if not self.grasped:
                if ee_to_obj < 0.04 and self.gripper_aperture < 0.35:
                    self.grasped = True
            else:
                if self.gripper_aperture > 0.65:
                    self.grasped = False
                else:
                    # Object follows EE
                    self.object_pos = self.ee_pos.copy() - np.array([0.0, 0.0, 0.015])

            dist = np.linalg.norm(self.object_pos - self.goal_pos)
            success = dist < 0.04 and self.object_pos[2] > 0.05
            reward = -float(dist) + (10.0 if success else 0.0)

        elif self.task_name == "drawer_open":
            handle_pos = self.drawer_pos.copy()
            handle_pos[1] -= self.drawer_open_dist * 0.15
            ee_to_handle = np.linalg.norm(self.ee_pos - handle_pos)
            # Grasp or hook handle
            if not self.grasped:
                if ee_to_handle < 0.045 and self.gripper_aperture < 0.5:
                    self.grasped = True
            else:
                if self.gripper_aperture > 0.7:
                    self.grasped = False
                elif action[1] < 0.0:
                    pull_delta = abs(action[1]) * 1.5
                    self.drawer_open_dist = min(1.0, self.drawer_open_dist + pull_delta)

            if ee_to_handle < 0.05 and action[1] < -0.005:
                pull_delta = abs(action[1]) * 1.2
                self.drawer_open_dist = min(1.0, self.drawer_open_dist + pull_delta)

            success = self.drawer_open_dist > 0.65
            reward = float(self.drawer_open_dist) * 5.0 + (10.0 if success else 0.0)

        terminated = bool(success)
        truncated = bool(self._step_count >= self.max_episode_steps)

        obs = self.render()
        info = self._get_info(success=success)
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        return render_scene(
            img_size=self.image_size,
            table_bounds=self.table_bounds,
            ee_pos=self.ee_pos,
            gripper_open=self.gripper_aperture,
            object_pos=self.object_pos,
            goal_pos=self.goal_pos,
            drawer_pos=self.drawer_pos,
            drawer_open_dist=self.drawer_open_dist,
            task_name=self.task_name,
        )

    def get_goal_image(self) -> np.ndarray:
        """Render the goal state observation."""
        if self.task_name == "reach":
            goal_ee = self.goal_pos.copy()
            return render_scene(
                img_size=self.image_size,
                table_bounds=self.table_bounds,
                ee_pos=goal_ee,
                gripper_open=0.0,
                goal_pos=self.goal_pos,
                task_name=self.task_name,
            )
        elif self.task_name == "push":
            target_obj = self.goal_pos.copy()
            ee_target = self.goal_pos.copy() + np.array([0.0, -0.03, 0.02])
            return render_scene(
                img_size=self.image_size,
                table_bounds=self.table_bounds,
                ee_pos=ee_target,
                gripper_open=0.0,
                object_pos=target_obj,
                goal_pos=self.goal_pos,
                task_name=self.task_name,
            )
        elif self.task_name == "pick_place":
            target_obj = self.goal_pos.copy()
            ee_target = self.goal_pos.copy() + np.array([0.0, 0.0, 0.02])
            return render_scene(
                img_size=self.image_size,
                table_bounds=self.table_bounds,
                ee_pos=ee_target,
                gripper_open=0.0,
                object_pos=target_obj,
                goal_pos=self.goal_pos,
                task_name=self.task_name,
            )
        elif self.task_name == "drawer_open":
            handle_pos = self.drawer_pos.copy()
            handle_pos[1] -= 0.75 * 0.15
            return render_scene(
                img_size=self.image_size,
                table_bounds=self.table_bounds,
                ee_pos=handle_pos,
                gripper_open=0.0,
                drawer_pos=self.drawer_pos,
                drawer_open_dist=0.8,
                task_name=self.task_name,
            )
        return self.render()

    def get_expert_action(self) -> np.ndarray:
        """
        Scripted operational space expert controller for data collection and demonstration generation.
        """
        action = np.zeros(4, dtype=np.float32)

        if self.task_name == "reach":
            delta = self.goal_pos - self.ee_pos
            norm = np.linalg.norm(delta)
            if norm > 1e-4:
                step_size = min(0.03, norm)
                action[:3] = (delta / norm) * step_size
            action[3] = 1.0  # Closed gripper

        elif self.task_name == "push":
            push_vector = self.goal_pos[:2] - self.object_pos[:2]
            push_dist = np.linalg.norm(push_vector)
            if push_dist < 1e-4:
                push_dir = np.array([0.0, 1.0], dtype=np.float32)
            else:
                push_dir = push_vector / push_dist

            behind_obj = np.zeros(3, dtype=np.float32)
            behind_obj[:2] = self.object_pos[:2] - push_dir * 0.05
            behind_obj[2] = 0.03

            # Phase 1: Navigate to position behind object
            if np.linalg.norm(self.ee_pos[:2] - behind_obj[:2]) > 0.025:
                delta = behind_obj - self.ee_pos
                norm = np.linalg.norm(delta)
                action[:3] = (delta / norm) * min(0.04, norm)
                action[3] = -1.0
            else:
                # Phase 2: Push along push direction toward goal
                action[:2] = push_dir * 0.035
                action[2] = -0.01  # Keep low near table
                action[3] = 1.0

        elif self.task_name == "pick_place":
            above_obj = self.object_pos.copy() + np.array([0.0, 0.0, 0.06])
            at_obj = self.object_pos.copy() + np.array([0.0, 0.0, 0.015])

            if not self.grasped:
                if np.linalg.norm(self.ee_pos[:2] - self.object_pos[:2]) > 0.02 and self.ee_pos[2] > 0.05:
                    delta = above_obj - self.ee_pos
                    action[:3] = (delta / np.linalg.norm(delta)) * min(0.03, np.linalg.norm(delta))
                    action[3] = -1.0
                elif np.linalg.norm(self.ee_pos - at_obj) > 0.015:
                    delta = at_obj - self.ee_pos
                    action[:3] = (delta / np.linalg.norm(delta)) * min(0.02, np.linalg.norm(delta))
                    action[3] = -1.0
                else:
                    action[3] = 1.0
            else:
                if self.ee_pos[2] < self.goal_pos[2]:
                    action[2] = 0.025
                    action[3] = 1.0
                else:
                    delta = self.goal_pos - self.object_pos
                    norm = np.linalg.norm(delta)
                    if norm > 0.02:
                        action[:3] = (delta / norm) * min(0.025, norm)
                        action[3] = 1.0
                    else:
                        action[3] = -1.0

        elif self.task_name == "drawer_open":
            handle_pos = self.drawer_pos.copy()
            handle_pos[1] -= self.drawer_open_dist * 0.15

            if not self.grasped:
                delta = handle_pos - self.ee_pos
                norm = np.linalg.norm(delta)
                if norm > 0.02:
                    action[:3] = (delta / norm) * min(0.035, norm)
                    action[3] = -1.0
                else:
                    action[3] = 1.0
            else:
                action[1] = -0.04
                action[3] = 1.0

        return np.clip(action, self.action_space.low, self.action_space.high)

    def _get_info(self, success: bool = False) -> Dict[str, Any]:
        # 7D state vector: [x, y, z, roll=0, pitch=0, yaw=0, gripper]
        state = np.array(
            [
                self.ee_pos[0],
                self.ee_pos[1],
                self.ee_pos[2],
                0.0,
                0.0,
                0.0,
                self.gripper_aperture,
            ],
            dtype=np.float32,
        )
        return {
            "state": state,
            "goal_state": self.goal_pos.copy() if self.goal_pos is not None else np.zeros(3),
            "step": self._step_count,
            "success": bool(success),
            "task_name": self.task_name,
        }
