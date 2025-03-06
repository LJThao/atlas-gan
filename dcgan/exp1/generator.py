import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_generator():
    model = keras.Sequential()

    # Input: Random noise vector
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    # Reshape to (7,7,256)
    model.add(layers.Reshape((7, 7, 256)))

    # First upsampling to (14,14,128) with larger filter size
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    # Second upsampling to (28,28,64)
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    # Additional upsampling layer to improve image details
    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(1, 1), padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    # Output layer: (28,28,1) with tanh activation
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(1, 1), padding="same", activation="tanh", use_bias=False))

    return model
