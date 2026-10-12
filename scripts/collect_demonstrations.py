import os
import argparse
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

from src.env.manipulation_env import ManipulationEnv
from src.utils.seed import set_seed


def collect_demonstrations(
    task_name: str,
    num_episodes: int,
    save_dir: str,
    image_size: int = 224,
    seed: int = 42,
    train_split: float = 0.8,
) -> None:
    set_seed(seed)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    env = ManipulationEnv(task_name=task_name, image_size=image_size, seed=seed)

    all_obs = []
    all_actions = []
    all_states = []
    all_rewards = []
    all_goals = []
    successes = 0

    print(f"Collecting {num_episodes} demonstration episodes for task '{task_name}'...")

    for ep in tqdm(range(num_episodes)):
        obs, info = env.reset(seed=seed + ep)
        ep_obs = [obs]
        ep_actions = []
        ep_states = [info["state"]]
        ep_rewards = []
        done = False

        while not done:
            action = env.get_expert_action()
            next_obs, reward, terminated, truncated, info = env.step(action)
            ep_actions.append(action)
            ep_obs.append(next_obs)
            ep_states.append(info["state"])
            ep_rewards.append(reward)
            done = terminated or truncated

        if info["success"]:
            successes += 1

        all_obs.append(np.array(ep_obs, dtype=np.uint8))
        all_actions.append(np.array(ep_actions, dtype=np.float32))
        all_states.append(np.array(ep_states, dtype=np.float32))
        all_rewards.append(np.array(ep_rewards, dtype=np.float32))
        all_goals.append(info["goal_state"].astype(np.float32))

    success_rate = (successes / num_episodes) * 100.0
    print(f"Demonstration collection complete. Expert Success Rate: {success_rate:.1f}%")

    # Pad or save as object arrays / fixed length chunks
    # Since episodes can vary in length, we store as list of arrays or save train / val
    num_train = int(num_episodes * train_split)
    
    for split, start_idx, end_idx in [
        ("train", 0, num_train),
        ("val", num_train, num_episodes),
    ]:
        split_file = save_path / f"{split}.npz"
        np.savez_compressed(
            split_file,
            observations=np.array(all_obs[start_idx:end_idx], dtype=object),
            actions=np.array(all_actions[start_idx:end_idx], dtype=object),
            states=np.array(all_states[start_idx:end_idx], dtype=object),
            rewards=np.array(all_rewards[start_idx:end_idx], dtype=object),
            goals=np.array(all_goals[start_idx:end_idx], dtype=np.float32),
        )
        print(f"Saved {split} set ({end_idx - start_idx} episodes) to {split_file}")

    metadata = {
        "task_name": task_name,
        "total_episodes": num_episodes,
        "train_episodes": num_train,
        "val_episodes": num_episodes - num_train,
        "expert_success_rate": success_rate,
        "image_size": image_size,
        "seed": seed,
    }
    with open(save_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect robot trajectory demonstrations.")
    parser.add_argument("--env", type=str, default="reach", help="Task name")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--save_dir", type=str, default="data/reach", help="Directory to save dataset")
    parser.add_argument("--image_size", type=int, default=224, help="Observation image size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    collect_demonstrations(
        task_name=args.env,
        num_episodes=args.num_episodes,
        save_dir=args.save_dir,
        image_size=args.image_size,
        seed=args.seed,
    )
