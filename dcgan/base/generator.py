import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_generator():
    model = keras.Sequential()

    # Input: Random noise vector (latent space)
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    # Reshape to (7,7,256)
    model.add(layers.Reshape((7, 7, 256)))

    # Upsample to (14,14,128)
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    # Upsample to (28,28,64)
    model.add(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    # Output layer: (28,28,1) with tanh activation
    model.add(layers.Conv2DTranspose(1, (4, 4), strides=(1, 1), padding="same", activation="tanh", use_bias=False))

    return model

# Create and summarize the generator
if __name__ == "__main__":
    generator = build_generator()
    generator.summary()
