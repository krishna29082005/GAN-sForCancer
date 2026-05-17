import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from PIL import Image

# Paths
data_dir = r"C:\Users\Krishna Mehra\Desktop\GAN project\data\brain_glioma"
output_dir = r"C:\Users\Krishna Mehra\Desktop\GAN project\output"
generated_dir = os.path.join(output_dir, "generated_images")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(generated_dir, exist_ok=True)

# Hyperparameters
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 300
LATENT_DIM = 100
LEARNING_RATE = 0.0002
BETA_1 = 0.5
N_CRITIC = 5  # Number of discriminator updates per generator update
LAMBDA_GP = 10  # Gradient penalty weight

# Load and preprocess images
def load_images(data_dir, img_size):
    images = []
    for img_name in os.listdir(data_dir):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(data_dir, img_name)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_size, img_size))
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = (img / 127.5) - 1.0  # normalize to [-1, 1]
            images.append(img)
    return np.array(images)

print("Loading images...")
X_train = load_images(data_dir, IMG_SIZE)
print(f"Loaded {len(X_train)} images")

# Generator model
def build_generator():
    input_layer = Input(shape=(LATENT_DIM,))
    x = Dense(8*8*256)(input_layer)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    x = Reshape((8, 8, 256))(x)

    x = Conv2DTranspose(128, kernel_size=4, strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)

    x = Conv2DTranspose(64, kernel_size=4, strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)

    x = Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='tanh')(x)

    return Model(input_layer, x, name='generator')

# Discriminator (critic) model
def build_discriminator():
    input_layer = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = Conv2D(64, kernel_size=4, strides=2, padding='same')(input_layer)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128, kernel_size=4, strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dense(1)(x)  # No activation here (linear output)
    return Model(input_layer, x, name='discriminator')

# Gradient penalty function for WGAN-GP
def gradient_penalty(discriminator, real_images, fake_images):
    batch_size = real_images.shape[0]
    alpha = tf.random.uniform([batch_size, 1, 1, 1], 0., 1.)
    interpolated = alpha * real_images + (1 - alpha) * fake_images
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = discriminator(interpolated)
    grads = tape.gradient(pred, interpolated)
    grads_l2 = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    penalty = tf.reduce_mean((grads_l2 - 1.0) ** 2)
    return penalty

# Instantiate models and optimizers
generator = build_generator()
discriminator = build_discriminator()
gen_optimizer = Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1)
disc_optimizer = Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1)

# Training step for discriminator
@tf.function
def train_discriminator(real_images):
    noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])
    with tf.GradientTape() as tape:
        fake_images = generator(noise, training=True)
        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(fake_images, training=True)
        gp = gradient_penalty(discriminator, real_images, fake_images)
        loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output) + LAMBDA_GP * gp
    grads = tape.gradient(loss, discriminator.trainable_variables)
    disc_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))
    return loss

# Training step for generator
@tf.function
def train_generator():
    noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])
    with tf.GradientTape() as tape:
        fake_images = generator(noise, training=True)
        output = discriminator(fake_images, training=True)
        loss = -tf.reduce_mean(output)
    grads = tape.gradient(loss, generator.trainable_variables)
    gen_optimizer.apply_gradients(zip(grads, generator.trainable_variables))
    return loss

# Save generated images for preview
def save_generated_images(epoch, examples=32):
    noise = tf.random.normal([examples, LATENT_DIM])
    gen_imgs = generator(noise, training=False).numpy()
    gen_imgs = 0.5 * gen_imgs + 0.5  # scale to [0, 1]
    plt.figure(figsize=(6, 6))
    for i in range(examples):
        plt.subplot(6, 6, i + 1)
        plt.imshow(gen_imgs[i])
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"epoch_{epoch}.png"))
    plt.close()

# Generate and save individual images after training
def generate_and_save_images(generator, num_images, batch_size, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    num_batches = (num_images + batch_size - 1) // batch_size
    for i in range(num_batches):
        current_batch_size = min(batch_size, num_images - i * batch_size)
        noise = tf.random.normal([current_batch_size, LATENT_DIM])
        gen_imgs = generator(noise, training=False).numpy()
        gen_imgs = 0.5 * gen_imgs + 0.5  # scale to [0, 1]
        for j in range(current_batch_size):
            img_array = (gen_imgs[j] * 255).astype(np.uint8)
            img = Image.fromarray(img_array)
            img.save(os.path.join(output_folder, f"generated_{i * batch_size + j + 1}.png"))

# Training loop
def train(dataset, epochs):
    d_losses = []
    g_losses = []
    dataset = tf.data.Dataset.from_tensor_slices(dataset).shuffle(10000).batch(BATCH_SIZE, drop_remainder=True).prefetch(1)
    for epoch in range(epochs):
        for real_images in dataset:
            for _ in range(N_CRITIC):
                d_loss = train_discriminator(real_images)
            g_loss = train_generator()
        d_losses.append(d_loss.numpy())
        g_losses.append(g_loss.numpy())
        if (epoch + 1) % 500 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | D Loss: {d_loss:.4f} | G Loss: {g_loss:.4f}")
            save_generated_images(epoch + 1)
    # Plot loss curves
    plt.figure()
    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('WGAN-GP Training Losses')
    plt.savefig(os.path.join(output_dir, "training_loss_graph.png"))
    plt.show()

# Start training
train(X_train, EPOCHS)
print(f"\nTraining complete! Generated images and loss graph saved to {output_dir}")

# Generate 6000 individual images after training
generate_and_save_images(generator, 6000, BATCH_SIZE, generated_dir)
print(f"7000 images generated and saved individually in {generated_dir}")
