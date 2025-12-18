# Functions used for attacks across multiple examples of the MIDST competition submissions.

from collections.abc import Generator
from pathlib import Path


def iterate_model_folders(
    input_data_path: Path, diffusion_model_names: list[str]
) -> Generator[tuple[str, Path, str, str]]:
    """
    Iterates over the competition's shadow model folder structure and yields model information.

    Args:
        input_data_path: The base path for the input data.
        diffusion_model_names: A list of diffusion model names to iterate over.

    Yields:
        A tuple containing the model name (e.g. tabddpm), the path to the model's data,
        the model folder name (e.g. tabddpm_1), and mode (train, dev, final).
    """
    modes = ["train", "dev", "final"]
    for model_name in diffusion_model_names:
        model_path = input_data_path / f"{model_name}_black_box"
        for mode in modes:
            current_path = model_path / mode
            if not current_path.exists():
                continue

            model_folders = [entry for entry in current_path.iterdir() if entry.is_dir()]
            for model_folder_path in model_folders:
                yield model_name, model_folder_path, model_folder_path.name, mode
