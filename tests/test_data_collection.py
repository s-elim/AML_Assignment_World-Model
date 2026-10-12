import os
import shutil
import tempfile
import numpy as np
import pytest

from scripts.collect_demonstrations import collect_demonstrations


def test_collect_demonstrations_execution():
    temp_dir = tempfile.mkdtemp()
    try:
        collect_demonstrations(
            task_name="reach",
            num_episodes=4,
            save_dir=temp_dir,
            image_size=64,  # Fast image size for test
            seed=42,
            train_split=0.5,
        )

        train_path = os.path.join(temp_dir, "train.npz")
        val_path = os.path.join(temp_dir, "val.npz")
        meta_path = os.path.join(temp_dir, "metadata.json")

        assert os.path.exists(train_path)
        assert os.path.exists(val_path)
        assert os.path.exists(meta_path)

        train_data = np.load(train_path, allow_pickle=True)
        assert len(train_data["observations"]) == 2
        assert len(train_data["actions"]) == 2
        assert train_data["observations"][0].shape[1:] == (64, 64, 3)
    finally:
        shutil.rmtree(temp_dir)
