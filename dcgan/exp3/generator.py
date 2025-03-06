import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_generator():
    model = keras.Sequential(name="Generator")

    # 🔹 Update input shape to match Exp3 (200 instead of 100)
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(200,), dtype="float16"))
    model.add(layers.BatchNormalization(dtype="float16"))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Reshape((7, 7, 256), dtype="float16"))

    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same", use_bias=False, dtype="float16"))
    model.add(layers.BatchNormalization(dtype="float16"))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding="same", use_bias=False, dtype="float16"))
    model.add(layers.BatchNormalization(dtype="float16"))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2DTranspose(1, (4, 4), strides=(1, 1), padding="same", activation="tanh", use_bias=False, dtype="float32"))

    return model