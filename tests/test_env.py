"""LIBERO wrapper: a held-out demonstration replayed at the 4 Hz macro rate reaches the BDDL goal.

Needs the regenerated dataset and an offscreen MuJoCo backend (MUJOCO_GL=osmesa or egl); skipped otherwise.
"""

import os

import numpy as np
import pytest

from src.data.libero_data import load_demos

pytestmark = pytest.mark.skipif(
    not os.path.exists("data/libero_goal4") or os.environ.get("MUJOCO_GL") not in ("osmesa", "egl"),
    reason="needs data/libero_goal4 and an offscreen MuJoCo backend",
)


def test_macro_replay_reaches_goal() -> None:
    from src.env.libero_env import LiberoGoalEnv

    demo = [d for d in load_demos(tasks=["put_the_bowl_on_the_plate"]) if d.split == "test"][0]
    env = LiberoGoalEnv("put_the_bowl_on_the_plate")
    env.set_goal_from_state(demo.sim_state[-1])
    obs, info = env.reset(options={"init_state": demo.sim_state[0]})
    assert obs["agentview"].shape == (128, 128, 3) and obs["wrist"].shape == (128, 128, 3)
    assert np.allclose(obs["state"], demo.state[0], atol=1e-3)
    start_err = env.goal_error_cm()
    a = demo.action[:-1]
    success = False
    for t in range(0, len(a), 5):
        obs, r, term, trunc, info = env.step(a[t : t + 5].mean(0))
        if term:
            success = True
            break
    assert success and info["goal_error_cm"] < start_err
    env.close()
