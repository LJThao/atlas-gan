import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import csv
from generator import build_generator
from discriminator import build_discriminator
from load_data import load_mnist_data

# Load dataset
train_images = load_mnist_data()

# Define experiment paths
image_folder = "/content/dcgan/exp2/generated_images/"
log_folder = "/content/dcgan/exp2/logs/"
os.makedirs(image_folder, exist_ok=True)
os.makedirs(log_folder, exist_ok=True)

# Corrected CSV log file path
log_file = os.path.join(log_folder, "exp2_training_log.csv")

# Experiment 2: Hyperparameter Tuning
BUFFER_SIZE = 60000
BATCH_SIZE = 128  # Reduced from 256 (Test smaller batches)
EPOCHS = 50       # Keep at 50 for now
NOISE_DIM = 200   # Increased noise dimension (More complexity in generated images)
LEARNING_RATE = 0.0001  # Lower learning rate for stability
BETA_1 = 0.6  # Changed Adam's beta_1 parameter for momentum adjustments

# Prepare dataset
dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Initialize generator & discriminator
generator = build_generator()
discriminator = build_discriminator()

# Define loss functions
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

# Modified Optimizers for Experiment 2
generator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=BETA_1)
discriminator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=BETA_1)

# Initialize CSV logging
with open(log_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Generator Loss", "Discriminator Loss"])

# Training function
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

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
def save_generated_images(epoch, generator, noise_dim=100, num_examples=16):
    if (epoch + 1) % 10 == 0:
        noise = tf.random.normal([num_examples, noise_dim])
        generated_images = generator(noise, training=False)

        fig, axes = plt.subplots(4, 4, figsize=(6, 6))
        for i, ax in enumerate(axes.flat):
            ax.imshow(generated_images[i, :, :, 0], cmap="gray")
            ax.axis("off")

        file_path = f"{image_folder}/epoch_{epoch+1}.png"
        plt.savefig(file_path)
        plt.close()
        print(f"Saved {file_path}")

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

        save_generated_images(epoch, generator, noise_dim=NOISE_DIM)

# Start training
train(dataset, EPOCHS)

# Save models in Experiment 2 folder
generator.save("/content/dcgan/exp2/exp2_generator.h5")
discriminator.save("/content/dcgan/exp2/exp2_discriminator.h5")

# Freeze the discriminator before adding to DCGAN
discriminator.trainable = False  

# Create and save the full DCGAN model
dcgan_model = tf.keras.models.Sequential([generator, discriminator])
dcgan_model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=BETA_1))
dcgan_model.save("/content/dcgan/exp2/exp2_dcgan_model.h5")
print("Experiment 2: DCGAN model saved successfully!")
