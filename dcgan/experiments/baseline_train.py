import os
import sys
import yaml
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from


(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_train = np.expand_dims(x_train, axis=-1)
x_train = x_train * 2.0 - 1.0

# Set hyperparameters
latent_dim = 100
img_shape = x_train.shape[1:]
epochs = 50
batch_size = 128
num_batches = x_train.shape[0] // batch_size

# Build the generator and discriminator models
generator = build_generator(latent_dim)
discriminator = build_discriminator(img_shape)

# Compile the discriminator
discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

# Build the combined model by stacking the generator and discriminator.
# Freeze the discriminator's weights when training the generator.
discriminator.trainable = False
noise_input = tf.keras.Input(shape=(latent_dim,))
generated_img = generator(noise_input)
validity = discriminator(generated_img)
combined = tf.keras.Model(noise_input, validity)
combined.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                 loss='binary_crossentropy')


# Function to visualize generated images during training
def display_generated_images(generator, latent_dim, epoch, examples=16, dim=(4, 4), figsize=(4, 4)):
    noise = np.random.normal(0, 1, (examples, latent_dim))
    generated_images = generator.predict(noise)
    # Rescale images from [-1, 1] to [0, 1] for display
    generated_images = (generated_images + 1) / 2.0

    fig, axs = plt.subplots(dim[0], dim[1], figsize=figsize, sharex=True, sharey=True)
    cnt = 0
    for i in range(dim[0]):
        for j in range(dim[1]):
            axs[i, j].imshow(generated_images[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    plt.suptitle(f'Epoch {epoch}')
    plt.tight_layout()
    plt.show()


# Training loop for the DCGAN
for epoch in range(1, epochs + 1):
    for batch in range(num_batches):
        # ---------------------
        #  Train Discriminator
        # ---------------------
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        real_imgs = x_train[idx]

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_imgs = generator.predict(noise)

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        d_loss_real = discriminator.train_on_batch(real_imgs, valid)
        d_loss_fake = discriminator.train_on_batch(fake_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train Generator
        # ---------------------
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = combined.train_on_batch(noise, valid)

    print(f"Epoch {epoch}/{epochs} [D loss: {d_loss[0]:.4f}, acc: {d_loss[1] * 100:.2f}%] [G loss: {g_loss:.4f}]")

    # Display generated images at the first epoch and every 5 epochs
    if epoch == 1 or epoch % 5 == 0:
        display_generated_images(generator, latent_dim, epoch)