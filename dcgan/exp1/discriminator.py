import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_discriminator():
    model = keras.Sequential()

    # Input: 28x28x1 image
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same", input_shape=(28, 28, 1)))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.4))

    # Additional Conv layer (New)
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.4))

    # Flatten and classify as real/fake
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="sigmoid"))

    return model
