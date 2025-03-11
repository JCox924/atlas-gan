import os
import numpy as np
import tensorflow as tf

def download_and_preprocess_mnist(save_path):
    """
    Downloads the MNIST dataset, preprocesses it, and saves it as a compressed .npz file.

    Preprocessing steps:
      - Convert images to float32 and normalize pixel values to [0, 1]
      - Rescale images to [-1, 1] for compatibility with DCGAN (tanh activation)
      - Expand image dimensions to include a channel (from (28,28) to (28,28,1))
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    x_train = x_train * 2.0 - 1.0
    x_test = x_test * 2.0 - 1.0

    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    np.savez_compressed(save_path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    print(f"Preprocessed MNIST dataset saved to: {save_path}")

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'data')
    os.makedirs(data_dir, exist_ok=True)

    output_file = os.path.join(data_dir, 'mnist_preprocessed.npz')

    download_and_preprocess_mnist(output_file)