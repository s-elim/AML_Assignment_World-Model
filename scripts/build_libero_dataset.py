"""Build the four-task LIBERO-Goal dataset used by every model in this study.

For each task the 50 human demonstrations released with LIBERO are read from the official hdf5 file. The
stored MuJoCo states are replayed through the evaluation renderer so that training images and evaluation
images come from the same renderer (the released frames differ from a re-render of the same state by 3 to
5 intensity levels on average, measured on 2026-10-12). Per control step the output holds the agent-view
and wrist RGB images (128x128), the 7-D proprioceptive state, the 7-D action that FOLLOWS the observation,
and the sim state, plus a fixed 40/10 train/test split of demonstration indices.

LIBERO's hdf5 convention: ``states[i]`` precedes ``actions[i]`` and ``obs[i]`` is recorded after it, so
``obs[i]`` corresponds to ``states[i + 1]``. This script re-aligns everything to (observation, next action).

    python scripts/build_libero_dataset.py --task turn_on_the_stove --out data/libero_goal4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from src.env.libero_env import TASK_KEYS, LiberoGoalEnv, dataset_path, proprio_from_obs  # noqa: E402

N_DEMOS = 50
N_TEST = 10
SPLIT_SEED = 0


def split_indices() -> tuple[list[int], list[int]]:
    perm = np.random.RandomState(SPLIT_SEED).permutation(N_DEMOS)
    return sorted(perm[N_TEST:].tolist()), sorted(perm[:N_TEST].tolist())


def render_demo(env: LiberoGoalEnv, states: np.ndarray, actions: np.ndarray) -> dict[str, np.ndarray]:
    """Observation t is rendered at states[t]; the last one by executing the final action from states[-1]."""
    agent, wrist, prop, sims = [], [], [], []
    raw_env = env._env
    raw_env.reset()
    for t in range(len(states)):
        raw = raw_env.set_init_state(states[t])
        agent.append(raw["agentview_image"])
        wrist.append(raw["robot0_eye_in_hand_image"])
        prop.append(proprio_from_obs(raw))
        sims.append(states[t])
    raw, _, done, _ = raw_env.step(actions[-1])
    agent.append(raw["agentview_image"])
    wrist.append(raw["robot0_eye_in_hand_image"])
    prop.append(proprio_from_obs(raw))
    sims.append(raw_env.get_sim_state())
    # Action following observation t is actions[t]; the final observation has none (pad with zeros).
    act = np.concatenate([actions, np.zeros((1, actions.shape[1]))], 0)
    return {
        "agentview": np.stack(agent).astype(np.uint8),
        "wrist": np.stack(wrist).astype(np.uint8),
        "state": np.stack(prop).astype(np.float32),
        "action": act.astype(np.float32),
        "sim_state": np.stack(sims).astype(np.float64),
        "final_success": np.array(bool(done)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=TASK_KEYS)
    ap.add_argument("--out", default="data/libero_goal4")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    train_ids, test_ids = split_indices()
    env = LiberoGoalEnv(args.task)
    t0 = time.time()
    with h5py.File(dataset_path(args.task), "r") as f:
        for i in range(N_DEMOS):
            g = f["data"][f"demo_{i}"]
            demo = render_demo(env, g["states"][:], g["actions"][:])
            np.savez_compressed(os.path.join(args.out, f"{args.task}__demo{i:02d}.npz"), **demo)
            print(f"{args.task} demo {i} T={len(demo['state'])} success={bool(demo['final_success'])} "
                  f"{time.time() - t0:.0f}s", flush=True)
    meta = {"task": args.task, "train_demos": train_ids, "test_demos": test_ids, "split_seed": SPLIT_SEED,
            "source": dataset_path(args.task), "image_size": 128}
    with open(os.path.join(args.out, f"{args.task}__meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    env.close()


if __name__ == "__main__":
    main()
