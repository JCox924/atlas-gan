import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


def preprocess_and_save_mnist(npz_dir):
    # Download the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Preprocess the data:
    # 1. Normalize images to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # 2. Reshape to add a channel dimension (useful for CNNs)
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    # 3. One-hot encode the labels
    num_classes = 10
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # Ensure the directory exists; if not, create it.
    os.makedirs(npz_dir, exist_ok=True)

    # Define the path where the dataset will be saved
    save_path = os.path.join(npz_dir, "mnist_preprocessed.npz")

    # Save the data in a compressed .npz file
    np.savez_compressed(save_path,
                        x_train=x_train, y_train=y_train,
                        x_test=x_test, y_test=y_test)
    print(f"Dataset saved to: {save_path}")


def visualize_samples(x, y, num_samples=9):
    """
    Visualize a grid of sample images from the dataset.

    Parameters:
      x: numpy array of image data.
      y: numpy array of labels (one-hot encoded).
      num_samples: number of images to display (default 9).
    """
    plt.figure(figsize=(8, 8))
    for i in range(num_samples):
        plt.subplot(3, 3, i + 1)
        plt.imshow(x[i].squeeze(), cmap='gray')
        label = np.argmax(y[i])
        plt.title(f"Label: {label}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
