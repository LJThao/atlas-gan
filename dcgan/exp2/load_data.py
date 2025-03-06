import numpy as np
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
def load_mnist_data():
    (train_images, _), (_, _) = mnist.load_data()
    
    # Normalize to [-1,1] range for GAN training
    train_images = (train_images - 127.5) / 127.5
    train_images = np.expand_dims(train_images, axis=-1)

    print(f" Dataset loaded. Shape: {train_images.shape}")
    return train_images

if __name__ == "__main__":
    data = load_mnist_data()
