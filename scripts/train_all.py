import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train_ac_predictor import train_model

tasks = ["reach", "push", "pick_place", "drawer_open"]
models = ["vjepa2_ac", "dino_wm", "tdmpc2", "bc_policy"]

base_dir = Path(__file__).resolve().parent.parent

for model in models:
    config_path = base_dir / f"configs/model/{model}.yaml"
    for task in tasks:
        data_dir = base_dir / f"data/{task}"
        print(f"\n==========================================")
        print(f"TRAINING {model.upper()} ON {task.upper()}")
        print(f"==========================================")
        train_model(
            config_path=str(config_path),
            data_dir=str(data_dir),
            output_dir=str(base_dir / "results"),
            device="cpu",
            seed=42,
        )

print("\nALL 16 MODEL TRAININGS SUCCESSFULLY COMPLETED!")
