import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import PIL.Image

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Define the generator model
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 3)

    return model

# Generate random noise for the generator
def generate_noise(batch_size, noise_dim):
    return tf.random.normal([batch_size, noise_dim])

# Create a function to generate alien images using the generator model
def generate_aliens(generator, num_images, noise_dim):
    noise = generate_noise(num_images, noise_dim)
    generated_images = generator(noise, training=False)
    generated_images = 0.5 * (generated_images + 1.0)
    generated_images = np.array(generated_images)
    return generated_images

# Load the pre-trained generator model
generator = make_generator_model()
generator.load_weights('generator_weights.h5')

# Generate 10 alien images
num_images = 10
noise_dim = 100
generated_aliens = generate_aliens(generator, num_images, noise_dim)

# Save the generated images
for i in range(num_images):
    # Clip pixel values between 0 and 255
    clipped_image = np.clip(generated_aliens[i] * 255, 0, 255)

    # Check for NaN or infinity values
    clipped_image[np.isnan(clipped_image)] = 0.0
    clipped_image[np.isinf(clipped_image)] = 255.0

    image = PIL.Image.fromarray(np.uint8(clipped_image))
    image.save(f'alien_{i}.png')