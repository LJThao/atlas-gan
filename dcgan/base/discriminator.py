import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_discriminator():
    model = keras.Sequential()

    # Input: 28x28x1 grayscale image
    model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), padding="same", input_shape=(28, 28, 1)))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    # Second conv layer: Downsample
    model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    # Flatten and output (NO sigmoid for logits)
    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

# Create and summarize the discriminator
if __name__ == "__main__":
    discriminator = build_discriminator()
    discriminator.summary()
