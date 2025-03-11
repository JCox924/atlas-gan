import tensorflow as tf
from tensorflow.keras import layers


def build_generator(latent_dim, k_size=5, filter_size=64, s1=2, s2=2, s3=2):
    """
    Builds a generator model for CelebA images (64x64x3).

    Parameters:
        latent_dim (int): Dimension of the latent space.
        k_size (int): Kernel size for Conv2DTranspose layers.
        filter_size (int): Base number of filters for the first upsampling block.
        s1 (int): Stride for the first upsampling layer.
        s2 (int): Stride for the second upsampling layer.
        s3 (int): Stride for the third upsampling layer.

    Returns:
        tf.keras.Sequential: The generator model.
    """
    model = tf.keras.Sequential(name="generator")

    model.add(layers.Dense(8 * 8 * 128, input_dim=latent_dim))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Reshape((8, 8, 128)))  # Output shape: (8, 8, 128)

    model.add(layers.Conv2DTranspose(filters=filter_size * 2,
                                     kernel_size=k_size,
                                     strides=s1,
                                     padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2DTranspose(filters=filter_size,
                                     kernel_size=k_size,
                                     strides=s2,
                                     padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2DTranspose(filters=3,
                                     kernel_size=k_size,
                                     strides=s3,
                                     padding='same',
                                     activation='tanh'))
    return model


def build_discriminator(img_shape, k_size=5, alpha=0.2, s=2):
    """
    Builds a discriminator model for CelebA images (e.g., 64x64x3).

    Parameters:
        img_shape (tuple): Shape of the input image (e.g., (64, 64, 3)).
        k_size (int): Kernel size for Conv2D layers.
        alpha (float): Negative slope for LeakyReLU.
        s (int): Stride for the convolution layers.

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

    model.add(layers.Conv2D(filters=256, kernel_size=k_size, strides=s, padding='same'))
    model.add(layers.LeakyReLU(alpha=alpha))
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))

    return model
