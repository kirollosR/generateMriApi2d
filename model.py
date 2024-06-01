import os
# Set environment variable
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses INFO and WARNING logs

import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model
import glob
import matplotlib.pyplot as plt
import numpy as np


class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        # Create trainable parameters gamma and beta
        self.gamma = self.add_weight(shape=(input_shape[-1],),
                                     initializer='ones',
                                     trainable=True,
                                     name='gamma')
        self.beta = self.add_weight(shape=(input_shape[-1],),
                                    initializer='zeros',
                                    trainable=True,
                                    name='beta')
        super(InstanceNormalization, self).build(input_shape)

    def call(self, inputs):
        # Compute the mean and variance along the spatial dimensions
        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        normalized = (inputs - mean) / tf.sqrt(variance + self.epsilon)
        return self.gamma * normalized + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape


def conv_block(x, num_filters):
    x = L.Conv2D(num_filters, 3, padding="same")(x)
    x = InstanceNormalization()(x)
    x = L.Activation("relu")(x)

    x = L.Conv2D(num_filters, 3, padding="same")(x)
    x = InstanceNormalization()(x)
    x = L.Activation("relu")(x)

    return x


def se_block(x, num_filters, ratio=8):
    se_shape = (1, 1, num_filters)
    se = L.GlobalAveragePooling2D()(x)
    se = L.Reshape(se_shape)(se)
    se = L.Dense(num_filters // ratio, activation="relu", use_bias=False)(se)
    se = L.Dense(num_filters, activation="sigmoid", use_bias=False)(se)
    se = L.Reshape(se_shape)(se)
    x = L.Multiply()([x, se])
    return x


def encoder_block(x, num_filters):
    x = conv_block(x, num_filters)
    x = se_block(x, num_filters)
    p = L.MaxPool2D((2, 2))(x)
    return x, p


def decoder_block(x, s, num_filters):
    x = L.UpSampling2D(interpolation="bilinear")(x)
    x = L.Concatenate()([x, s])
    x = conv_block(x, num_filters)
    x = se_block(x, num_filters)
    return x


def squeeze_attention_unet(input_shape=(256, 256, 3)):
    """ Inputs """
    inputs = L.Input(input_shape)

    """ Encoder """
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)
    b1 = se_block(b1, 1024)

    """ Decoder """
    d = decoder_block(b1, s4, 512)
    d1 = decoder_block(d, s3, 256)
    d2 = decoder_block(d1, s2, 128)
    d3 = decoder_block(d2, s1, 64)

    """ Outputs """
    outputs = L.Conv2D(3, (1, 1), activation='tanh')(d3)

    """ Model """

    model = Model(inputs, outputs, name="Squeeze-Attention-UNET")
    return model


generator_g = squeeze_attention_unet()


def downsample(filters, size, apply_norm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                      kernel_initializer=initializer, use_bias=False))

    if apply_norm:
        result.add(InstanceNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result


def discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    x = inp
    down1 = downsample(64, 4, False)(x)  # (bs, 16, 16, 64)
    down2 = downsample(128, 4)(down1)
    down3 = downsample(256, 4)(down2)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)
    norm1 = InstanceNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(norm1)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(3, 4, strides=1, kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)
    return tf.keras.Model(inputs=inp, outputs=last)

discriminator_x = discriminator()

generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5 ) # , beta_1=0.5
discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5 ) # , beta_1=0.5

def load_model(ckpt_path):
    checkpoint = tf.train.Checkpoint(generator_g=generator_g,
                                     discriminator_x=discriminator_x,
                                     generator_g_optimizer=generator_g_optimizer,
                                     discriminator_x_optimizer=discriminator_x_optimizer)
    # Set up the checkpoint manager
    manager = tf.train.CheckpointManager(checkpoint, f"{ckpt_path}", max_to_keep=5)

    ckpt_restore_path = manager.latest_checkpoint
    if ckpt_restore_path:
        checkpoint.restore(ckpt_restore_path)
        print(f"Restored from {ckpt_restore_path}")
    else:
        print("Initializing from scratch.")



def clear_folder(folder_path):
    # List all files in the folder
    files = glob.glob(os.path.join(folder_path, "*.png"))
    # Delete each file
    for f in files:
        os.remove(f)


def generate_single_images_GIF(img_input, model, order, generated_results_path):
    # Clear the folder before saving new images

    # Generate model prediction
    prediction = model(img_input)
    # Extract the specific slice and rotate
    pred_vol = prediction[0, :, :, 0].numpy().copy()

    # Create a figure for displaying the image
    plt.figure(figsize=(10, 6))
    display_list = [pred_vol]
    for i in range(1):
        plt.subplot(1, 1, i + 1)
        plt.imshow(display_list[i], cmap='gray')
        plt.axis('off')

        # Format the save path to ensure filename is in the desired format
        slice_number = order + i + 6  # Calculate slice number based on order and offset
        save_path = f"{generated_results_path}\\slice_{slice_number:03d}.png"
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()

    return pred_vol

def normalize(image):
    image = (image/127.5)-1
    return image


def Generate(crop_black_boundary_path, Generated_results_path):
    Gen_A = tf.keras.preprocessing.image_dataset_from_directory(
        crop_black_boundary_path,
        seed=123,
        labels=None,
        image_size=(256, 256),
        batch_size=1,
        shuffle=False)

    Gen_B = tf.keras.preprocessing.image_dataset_from_directory(
        crop_black_boundary_path,
        seed=123,
        labels=None,
        image_size=(256, 256),
        batch_size=1,
        shuffle=False)

    Gen_A = Gen_A.map(lambda x: (normalize(x)))
    Gen_B = Gen_B.map(lambda x: (normalize(x)))

    order_2 = 0
    clear_folder(Generated_results_path)
    for image_x, image_y in tf.data.Dataset.zip((Gen_A, Gen_B)):
        generate_single_images_GIF(image_x, generator_g, order_2, Generated_results_path)
        order_2 = order_2 + 1

    print("Phase 3 'generate' Done")