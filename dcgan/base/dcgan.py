import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from generator import build_generator
from discriminator import build_discriminator

# Create generator and discriminator
generator = build_generator()
discriminator = build_discriminator()

# Compile discriminator
discriminator.compile(loss="binary_crossentropy",
                      optimizer=Adam(0.0002, 0.5),
                      metrics=["accuracy"])

# DCGAN Model: Freeze discriminator during GAN training
discriminator.trainable = False

# Create DCGAN model
dcgan = tf.keras.Sequential([generator, discriminator])

# Compile DCGAN
dcgan.compile(loss="binary_crossentropy", optimizer=Adam(0.0002, 0.5))

# Print DCGAN summary
dcgan.summary()
