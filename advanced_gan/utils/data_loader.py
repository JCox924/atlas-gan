import os
import numpy as np


def load_dataset(dataset_name="celeba", data_dir=None):
    """
    Loads a preprocessed dataset based on the dataset name.

    Parameters:
        dataset_name (str): Name of the dataset to load (e.g., "celeba", "mnist").
        data_dir (str): Path to the data directory. If None, it assumes the data directory is one level up.

    Returns:
        dict: A dictionary containing the dataset arrays.
    """
    if data_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.abspath(os.path.join(current_dir, "..", "data"))

    if dataset_name.lower() == "celeba":
        file_path = os.path.join(data_dir, "celeba_preprocessed.npz")
    elif dataset_name.lower() == "mnist":
        file_path = os.path.join(data_dir, "mnist_preprocessed.npz")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    data = np.load(file_path)
    return dict(data)