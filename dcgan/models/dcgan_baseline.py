import tensorflow as tf
from tensorflow.keras import layers


def build_generator(latent_dim, k_size=5, filter_size=32, s1=2, s2=2):
    """
    Builds a generator model with adjustable parameters for kernel size, filter count, and separate strides for each upsampling block.

    Parameters:
        latent_dim (int): Dimension of the latent noise vector.
        k_size (int): Kernel size for Conv2DTranspose layers.
        filter_size (int): Number of filters for the first Conv2DTranspose layer.
        s1 (int): Stride for the first upsampling layer.
        s2 (int): Stride for the second upsampling layer.

    Returns:
        tf.keras.Sequential: The generator model.
    """
    model = tf.keras.Sequential(name="generator")

    model.add(layers.Dense(7 * 7 * 128, input_dim=latent_dim))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Reshape((7, 7, 128)))

    model.add(layers.Conv2DTranspose(filters=filter_size,
                                     kernel_size=k_size,
                                     strides=s1,
                                     padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2DTranspose(filters=1,
                                     kernel_size=k_size,
                                     strides=s2,
                                     padding='same',
                                     activation='tanh'))
    return model


def build_discriminator(img_shape, k_size=5, alpha=0.1, s=1):
    """
    Builds a adjustable discriminator model

    Parameters:
        img_shape (tuple): Shape of the input image (e.g., (28, 28, 1)).
        k_size (int): Kernel size for Conv2D layers.
        alpha (float): Negative slope coefficient for LeakyReLU.
        s (int): Stride for the Conv2D layers.

    Returns:
        tf.keras.Sequential: The discriminator model.
    """
    model = tf.keras.Sequential(name="discriminator")
    model.add(layers.InputLayer(input_shape=img_shape))

    model.add(layers.Conv2D(filters=64, kernel_size=k_size, strides=s, padding='same'))
    model.add(layers.LeakyReLU(alpha=alpha))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(filters=128, kernel_size=k_size, strides=s, padding='same'))
    model.add(layers.LeakyReLU(alpha=alpha))
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))

    return model
