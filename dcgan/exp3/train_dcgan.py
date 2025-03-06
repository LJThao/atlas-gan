import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import csv
from generator import build_generator
from discriminator import build_discriminator

# Set mixed precision for better speed & efficiency
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy("mixed_float16")  # Enables mixed precision training

# Define experiment paths
image_folder = "/content/dcgan/exp3/generated_images/"
log_folder = "/content/dcgan/exp3/logs/"
os.makedirs(image_folder, exist_ok=True)
os.makedirs(log_folder, exist_ok=True)

# Load MNIST dataset
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype("float32")
train_images = (train_images - 127.5) / 127.5  # Normalize to [-1,1]

# Experiment 3 Hyperparameters
BUFFER_SIZE = 60000
BATCH_SIZE = 128  # Keeping same as exp2
EPOCHS = 50
NOISE_DIM = 200  # Keeping increased noise dim for more complex images
LEARNING_RATE = 0.0001
BETA_1 = 0.6

# Prepare dataset
dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Initialize generator & discriminator with float16
generator = build_generator()
discriminator = build_discriminator()

# Use float32 loss to avoid underflow issues
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True, dtype=tf.float32)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

# Mixed Precision Optimizers
generator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=BETA_1)
discriminator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=BETA_1)

# Initialize CSV logging
log_file = os.path.join(log_folder, "exp3_training_log.csv")
with open(log_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Generator Loss", "Discriminator Loss"])

# Training function
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM], dtype=tf.float16)  # 🔹 Explicit dtype set to float16

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

# Save generated images every 10 epochs
def save_generated_images(epoch, generator):
    if (epoch + 1) % 10 == 0:
        noise = tf.random.normal([16, NOISE_DIM])
        generated_images = generator(noise, training=False)

        fig, axes = plt.subplots(4, 4, figsize=(6, 6))
        for i, ax in enumerate(axes.flat):
            ax.imshow(generated_images[i, :, :, 0], cmap="gray")
            ax.axis("off")

        file_path = f"{image_folder}/epoch_{epoch+1}.png"
        plt.savefig(file_path)
        plt.close()
        print(f" Saved {file_path}")

# Training Loop
def train(dataset, epochs):
    for epoch in range(epochs):
        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)

        print(f"Epoch {epoch+1}/{epochs} - Gen Loss: {gen_loss.numpy():.4f}, Disc Loss: {disc_loss.numpy():.4f}")

        # Save logs
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, gen_loss.numpy(), disc_loss.numpy()])

        save_generated_images(epoch, generator)

# Start training
train(dataset, EPOCHS)

# Save models
generator.save("/content/dcgan/exp3/exp3_generator.h5")
discriminator.save("/content/dcgan/exp3/exp3_discriminator.h5")

# Freeze the discriminator before saving the full DCGAN model
discriminator.trainable = False

# Create and save the full DCGAN model
dcgan_model = tf.keras.models.Sequential([generator, discriminator])
dcgan_model.save("/content/dcgan/exp3/exp3_dcgan_model.h5")

print(" Experiment 3 models saved!")
