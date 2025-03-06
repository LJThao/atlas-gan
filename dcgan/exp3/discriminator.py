import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_discriminator():
    model = keras.Sequential(name="Discriminator")

    # Input: 28x28x1 Image
    model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), padding="same", input_shape=(28, 28, 1), dtype="float16"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding="same", dtype="float16"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten(dtype="float16"))
    model.add(layers.Dense(1, activation="sigmoid", dtype="float32"))

    return model