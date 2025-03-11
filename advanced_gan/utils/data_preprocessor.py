import os
import cv2  # or PIL.Image if preferred
import numpy as np
from data_loader import load_dataset


def preprocess_celeba(raw_dir, save_path, target_size=(64, 64)):
    """
    Processes raw CelebA images by reading, resizing, and normalizing them.
    If the preprocessed dataset already exists, load it using load_dataset.

    Parameters:
        raw_dir (str): Directory containing raw CelebA images.
        save_path (str): Path to save the preprocessed dataset (e.g., celeba_preprocessed.npz).
        target_size (tuple): Target image size (width, height). Default is (64, 64).

    Returns:
        dict: A dictionary containing the preprocessed dataset arrays.
    """
    if os.path.exists(save_path):
        print("Preprocessed CelebA dataset already exists. Loading dataset using data_loader...")
        data_dir = os.path.dirname(save_path)
        return load_dataset(dataset_name="celeba", data_dir=data_dir)

    print("Preprocessing CelebA images from raw directory:", raw_dir)
    image_files = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if
                   f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    images = []

    for file in image_files:
        print(f"Processing image: {file}")
        img = cv2.imread(file)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        images.append(img)

    if not images:
        raise ValueError("No images found in the specified raw data directory.")

    images = np.array(images).astype('float32') / 255.0
    images = images * 2.0 - 1.0

    np.savez_compressed(save_path, images=images)
    print(f"Preprocessed CelebA dataset saved to: {save_path}")

    data_dir = os.path.dirname(save_path)
    return load_dataset(dataset_name="celeba", data_dir=data_dir)


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    raw_celeba_dir = os.path.abspath(os.path.join(current_dir, '..', 'raw_celeba/img_align_celeba/img_align_celeba'))
    data_dir = os.path.abspath(os.path.join(current_dir, '..', 'data'))
    os.makedirs(data_dir, exist_ok=True)
    output_file = os.path.join(data_dir, 'celeba_preprocessed.npz')

    dataset = preprocess_celeba(raw_celeba_dir, output_file, target_size=(64, 64))
    print("Loaded dataset keys:", dataset.keys())
