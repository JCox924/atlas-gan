import tensorflow as tf
from tensorflow.keras import layers


def build_generator(latent_dim):
    """
    Build the generator model for the DCGAN.

    Parameters:
        latent_dim (int): Dimension of the latent space (input noise vector).

    Returns:
        tf.keras.Sequential: The generator model.
    """
    model = tf.keras.Sequential(name="generator")
    # Project the latent space to a 7x7x128 feature map.
    model.add(layers.Dense(7 * 7 * 128, input_dim=latent_dim))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Reshape((7, 7, 128)))

    # Upsample: 7x7 -> 14x14
    model.add(layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    # Upsample: 14x14 -> 28x28, output single channel image with tanh activation.
    model.add(layers.Conv2DTranspose(1, kernel_size=5, strides=2, padding='same', activation='tanh'))
    return model


def build_discriminator(img_shape):
    """
    Build the discriminator model for the DCGAN.

    Parameters:
        img_shape (tuple): Shape of the input image (e.g., (28, 28, 1)).

    Returns:
        tf.keras.Sequential: The discriminator model.
    """
    model = tf.keras.Sequential(name="discriminator")
    model.add(layers.Conv2D(64, kernel_size=5, strides=2, padding='same', input_shape=img_shape))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, kernel_size=5, strides=2, padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model
