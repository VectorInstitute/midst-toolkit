import os
import shutil


def start_save_dir(save_dir: str) -> None:
    # ruff: noqa: D103
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir + "models/", exist_ok=True)
