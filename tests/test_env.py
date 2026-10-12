import numpy as np
import pytest

from src.env.manipulation_env import ManipulationEnv, SUPPORTED_TASKS


def test_supported_tasks():
    assert "reach" in SUPPORTED_TASKS
    assert "push" in SUPPORTED_TASKS
    assert "pick_place" in SUPPORTED_TASKS
    assert "drawer_open" in SUPPORTED_TASKS


@pytest.mark.parametrize("task_name", ["reach", "push", "pick_place", "drawer_open"])
def test_env_initialization_and_reset(task_name):
    env = ManipulationEnv(task_name=task_name, image_size=224, seed=42)
    obs, info = env.reset()

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (224, 224, 3)
    assert obs.dtype == np.uint8

    assert "state" in info
    assert info["state"].shape == (7,)
    assert "goal_state" in info
    assert "success" in info
    assert not info["success"]


def test_env_step_mechanics():
    env = ManipulationEnv(task_name="reach", image_size=224, seed=42)
    obs, info = env.reset()
    init_state = info["state"].copy()

    action = np.array([0.02, -0.01, 0.0, 0.0], dtype=np.float32)
    next_obs, reward, terminated, truncated, next_info = env.step(action)

    assert next_obs.shape == (224, 224, 3)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert not np.allclose(init_state[:3], next_info["state"][:3])


def test_goal_image_generation():
    env = ManipulationEnv(task_name="push", image_size=224, seed=42)
    env.reset()
    goal_img = env.get_goal_image()

    assert isinstance(goal_img, np.ndarray)
    assert goal_img.shape == (224, 224, 3)
    assert goal_img.dtype == np.uint8


def test_scripted_expert_reaches_success():
    env = ManipulationEnv(task_name="reach", image_size=224, seed=42)
    obs, info = env.reset()
    done = False
    step_count = 0

    while not done and step_count < 100:
        action = env.get_expert_action()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1

    assert info["success"] is True


def test_libero_mujoco_env():
    from src.env.libero_env import LiberoMuJoCoEnv
    env = LiberoMuJoCoEnv(suite_name="libero_10", task_id=0, image_size=128, max_episode_steps=5, seed=0)
    obs, info = env.reset(seed=0)
    assert obs.shape == (128, 128, 3)
    assert "task_prompt" in info
    action = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    next_obs, reward, term, trunc, next_info = env.step(action)
    assert next_obs.shape == (128, 128, 3)
    env.close()
