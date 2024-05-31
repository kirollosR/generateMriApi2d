import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import Input, Conv3D, Activation, Add, LeakyReLU,Conv3DTranspose
from keras.models import Model
import keras

input_shape=(176, 176, 124, 1)
learning_rate = 0.0002
epochs = 2
batch_size = 1
gan_filters = 16
dis_filters = 32
downsampling =2
upsampling = 2
residual_blocks = 4
disc_downsampling_blocks=2


def get_resnet_generator_3d(input_shape=input_shape, filters=gan_filters, num_downsampling_blocks=downsampling,
                            num_residual_blocks=residual_blocks, num_upsample_blocks=upsampling, name=None):
    inputs = Input(shape=input_shape)

    # Initial Convolution
    x = Conv3D(filters, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same', use_bias=True)(inputs)
    x = tf.keras.layers.GroupNormalization(groups=-1)(x, training=True)
    x = Activation('relu')(x)

    # Downsampling
    for num_downsampl_blocks in range(num_downsampling_blocks):
        filters *= 2
        x = Conv3D(filters, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same', use_bias=True)(x)
        x = tf.keras.layers.GroupNormalization(groups=-1)(x, training=True)
        x = Activation('relu')(x)
        # print int(x.shape[-1])
    # Residual blocks
    for num_residual_block in range(num_residual_blocks):
        x_residual = x
        x = Conv3D(filters=int(x.shape[-1]), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', use_bias=True)(x)
        x = tf.keras.layers.GroupNormalization(groups=-1)(x, training=True)
        x = Activation('relu')(x)

        x = Conv3D(filters=int(x.shape[-1]), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', use_bias=True)(x)
        x = tf.keras.layers.GroupNormalization(groups=-1)(x, training=True)
        x = Add()([x_residual, x])
        x = Activation('relu')(x)

    # Upsampling
    for num_upsample_block in range(num_upsample_blocks):
        filters //= 2
        # x = UpSampling3D(size=(2, 2, 2))(x)
        x = Conv3DTranspose(filters, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same', use_bias=True)(x)
        x = tf.keras.layers.GroupNormalization(groups=-1)(x, training=True)
        x = Activation('relu')(x)

    # Final Convolution
    x = Conv3D(1, kernel_size=(7, 7, 7), strides=(1, 1, 1), padding='same')(x)
    x = Activation('tanh')(x)

    model = Model(inputs=inputs, outputs=x)
    return model


def get_discriminator_3d(input_shape=input_shape, filters=dis_filters, num_downsampling_blocks=disc_downsampling_blocks,
                         name=None):
    img_input = Input(shape=input_shape)

    # Initial Convolution
    x = Conv3D(filters, kernel_size=(4, 4, 4), strides=(2, 2, 2), padding="same", use_bias=True)(img_input)
    x = LeakyReLU(0.2)(x)

    num_filters = filters
    for num_downsample_block in range(num_downsampling_blocks):
        num_filters *= 2
        x = Conv3D(num_filters, (4, 4, 4),
                   strides=(2, 2, 2) if num_downsample_block < num_downsampling_blocks - 1 else (1, 1, 1),
                   padding="same", use_bias=False)(x)
        x = tf.keras.layers.GroupNormalization(groups=-1)(x, training=True)
        x = LeakyReLU(0.2)(x)

    x = Conv3D(1, kernel_size=(4, 4, 4), strides=(1, 1, 1), padding="same", use_bias=True)(x)
    x = Activation('sigmoid')(x)

    model = Model(inputs=img_input, outputs=x)
    return model

gen_A2B = get_resnet_generator_3d(name="generator_A")
gen_B2A = get_resnet_generator_3d(name="generator_B")

disc_A = get_discriminator_3d(name="discriminator_A")
disc_B = get_discriminator_3d(name="discriminator_B")


@keras.utils.register_keras_serializable()
class CycleGan(keras.Model):
    def __init__(
            self,
            generator_A,
            generator_B,
            discriminator_A,
            discriminator_B,
            lambda_cycle=10.0,
            lambda_identity=0.5,
    ):
        super().__init__()
        self.gen_A2B = generator_A
        self.gen_B2A = generator_B
        self.disc_A = discriminator_A
        self.disc_B = discriminator_B
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

        self.g_A_loss_tracker = keras.metrics.Mean(name="gen_A2B_loss")
        self.g_B_loss_tracker = keras.metrics.Mean(name="gen_B2A_loss")

        self.cycle_gen_A2B_tracker = keras.metrics.Mean(name="cycle_loss_A")
        self.cycle_gen_B2A_tracker = keras.metrics.Mean(name="cycle_loss_B")

        self.id_loss_A_tracker = keras.metrics.Mean(name="id_loss_A")
        self.id_loss_B_tracker = keras.metrics.Mean(name="id_loss_B")

        self.disc_A_loss_tracker = keras.metrics.Mean(name="disc_A_loss")
        self.disc_B_loss_tracker = keras.metrics.Mean(name="disc_B_loss")

        self.ssim_A_tracker = keras.metrics.Mean(name="ssim_A_score")
        self.ssim_B_tracker = keras.metrics.Mean(name="ssim_B_score")

    def call(self, inputs):
        return (
            self.disc_A(inputs),
            self.disc_B(inputs),
            self.gen_A2B(inputs),
            self.gen_B2A(inputs),
        )

    def compile(
            self,
            gen_A2B_optimizer,
            gen_B2A_optimizer,
            disc_A_optimizer,
            disc_B_optimizer,
            gen_loss_fn,
            disc_loss_fn,
    ):
        super().compile()
        self.gen_A2B_optimizer = gen_A2B_optimizer
        self.gen_B2A_optimizer = gen_B2A_optimizer
        self.disc_A_optimizer = disc_A_optimizer
        self.disc_B_optimizer = disc_B_optimizer
        self.generator_loss_fn = gen_loss_fn
        self.discriminator_loss_fn = disc_loss_fn
        self.cycle_loss_Bn = keras.losses.MeanAbsoluteError()
        self.identity_loss_fn = keras.losses.MeanAbsoluteError()

    @tf.function
    def train_step(self, batch_data):
        # x is T1 and y is T2
        real_A, real_B = batch_data

        # For CycleGAN, we need to calculate different
        # kinds of losses for the generators and discriminators.
        # We will perform the following steps here:
        #
        # 1. Pass real images through the generators and get the generated images
        # 2. Pass the generated images back to the generators to check if we
        #    can predict the original image from the generated image.
        # 3. Do an identity mapping of the real images using the generators.
        # 4. Pass the generated images in 1) to the corresponding discriminators.
        # 5. Calculate the generators total loss (adversarial + cycle + identity)
        # 6. Calculate the discriminators loss
        # 7. Update the weights of the generators
        # 8. Update the weights of the discriminators
        # 9. Return the losses in a dictionary

        with tf.GradientTape(persistent=True) as tape:
            # T1 to fake T2
            fake_B = self.gen_A2B(real_A, training=True)
            # T2 to fake T1 -> y2x
            fake_A = self.gen_B2A(real_B, training=True)

            # Cycle (T1 to fake T2 to fake T1): x -> y -> x
            cycled_A = self.gen_B2A(fake_B, training=True)
            # Cycle (T2 to fake T1 to fake T2) y -> x -> y
            cycled_B = self.gen_A2B(fake_A, training=True)

            # Identity mapping
            same_A = self.gen_B2A(real_A, training=True)
            same_B = self.gen_A2B(real_B, training=True)

            # Discriminator output
            disc_real_A = self.disc_A(real_A, training=True)
            disc_fake_A = self.disc_A(fake_A, training=True)

            disc_real_B = self.disc_B(real_B, training=True)
            disc_fake_B = self.disc_B(fake_B, training=True)

            # Generator adversarial loss
            gen_A2B_loss = self.generator_loss_fn(disc_fake_B)
            gen_B2A_loss = self.generator_loss_fn(disc_fake_A)

            # Generator cycle loss
            cycle_loss_A = self.cycle_loss_Bn(real_B, cycled_B) * self.lambda_cycle
            cycle_loss_B = self.cycle_loss_Bn(real_A, cycled_A) * self.lambda_cycle

            # Generator identity loss
            id_loss_A = (
                    self.identity_loss_fn(real_B, same_B)
                    * self.lambda_cycle
                    * self.lambda_identity
            )
            id_loss_B = (
                    self.identity_loss_fn(real_A, same_A)
                    * self.lambda_cycle
                    * self.lambda_identity
            )

            # Calculate model Structural Similarity Index Measure(SSIM)
            def calculate_ssim(real_images, generated_images):
                real_images = tf.cast(real_images, tf.float32)
                generated_images = tf.cast(generated_images, tf.float32)
                ssim_score = 0.5 * (
                tf.image.ssim(generated_images, real_images, max_val=2.0)[0]) + 0.5 * tf.reduce_mean(
                    tf.abs(generated_images - real_images))
                return ssim_score

            # Total generator loss
            total_loss_A2B = gen_A2B_loss + cycle_loss_A + id_loss_A
            # print("total_loss_A2B: ", total_loss_A2B)
            total_loss_B2A = gen_B2A_loss + cycle_loss_B + id_loss_B

            # Discriminator loss
            disc_A_loss = self.discriminator_loss_fn(disc_real_A, disc_fake_A)
            disc_B_loss = self.discriminator_loss_fn(disc_real_B, disc_fake_B)

            ssim_A_score = calculate_ssim(real_A, cycled_A)
            ssim_B_score = calculate_ssim(real_B, cycled_B)

        # Get the gradients for the generators
        grads_A2B = tape.gradient(total_loss_A2B, self.gen_A2B.trainable_variables)
        grads_B2A = tape.gradient(total_loss_B2A, self.gen_B2A.trainable_variables)

        # Get the gradients for the discriminators
        disc_A_grads = tape.gradient(disc_A_loss, self.disc_A.trainable_variables)
        disc_B_grads = tape.gradient(disc_B_loss, self.disc_B.trainable_variables)

        # Update the weights of the generators
        self.gen_A2B_optimizer.apply_gradients(
            zip(grads_A2B, self.gen_A2B.trainable_variables)
        )
        self.gen_B2A_optimizer.apply_gradients(
            zip(grads_B2A, self.gen_B2A.trainable_variables)
        )

        # Update the weights of the discriminators
        self.disc_A_optimizer.apply_gradients(
            zip(disc_A_grads, self.disc_A.trainable_variables)
        )
        self.disc_B_optimizer.apply_gradients(
            zip(disc_B_grads, self.disc_B.trainable_variables)
        )

        self.g_A_loss_tracker.update_state(gen_A2B_loss)
        self.g_B_loss_tracker.update_state(gen_B2A_loss)

        self.cycle_gen_A2B_tracker.update_state(cycle_loss_A)
        self.cycle_gen_B2A_tracker.update_state(cycle_loss_B)

        self.id_loss_A_tracker.update_state(id_loss_A)
        self.id_loss_B_tracker.update_state(id_loss_B)

        self.disc_A_loss_tracker.update_state(disc_A_loss)
        self.disc_B_loss_tracker.update_state(disc_B_loss)

        self.ssim_A_tracker.update_state(ssim_A_score)
        self.ssim_B_tracker.update_state(ssim_B_score)

        losses = {
            "gen_A2B_loss": self.g_A_loss_tracker.result(),
            "gen_B2A_loss": self.g_B_loss_tracker.result(),

            "cycle_loss_A": self.cycle_gen_A2B_tracker.result(),
            "cycle_loss_B": self.cycle_gen_B2A_tracker.result(),

            "id_loss_A": self.id_loss_A_tracker.result(),
            "id_loss_B": self.id_loss_B_tracker.result(),

            "disc_A_loss": self.disc_A_loss_tracker.result(),
            "disc_B_loss": self.disc_B_loss_tracker.result(),

            "ssim_score_A": self.ssim_A_tracker.result(),
            "ssim_score_B": self.ssim_B_tracker.result(),
        }

        return losses

    def get_config(self):
        config = {
            "generator_A": keras.utils.serialize_keras_object(self.gen_A2B),
            "generator_B": keras.utils.serialize_keras_object(self.gen_B2A),
            "discriminator_A": keras.utils.serialize_keras_object(self.disc_A),
            "discriminator_B": keras.utils.serialize_keras_object(self.disc_B),
            "lambda_cycle": self.lambda_cycle,
            "lambda_identity": self.lambda_identity,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        name = config.pop('name', None)
        trainable = config.pop('trainable', True)  # Handle the 'trainable' attribute
        dtype = config.pop('dtype', 'float32')  # Handle the 'dtype' attribute
        generator_G = keras.layers.deserialize(config.pop("generator_G"))
        generator_F = keras.layers.deserialize(config.pop("generator_F"))
        discriminator_A = keras.layers.deserialize(config.pop("discriminator_A"))
        discriminator_B = keras.layers.deserialize(config.pop("discriminator_B"))
        return cls(generator_G=generator_G, generator_F=generator_F, discriminator_A=discriminator_A,
                   discriminator_B=discriminator_B, **config)


adv_loss_fn = keras.losses.MeanSquaredError()
dis_bce_loss = keras.losses.MeanSquaredError()

# Define the loss function for the generators
def generator_loss_fn(fake):
    fake_loss = adv_loss_fn(tf.ones_like(fake), fake)
    return fake_loss

# Define the loss function for the discriminators
def discriminator_loss_fn(real, fake):
    real_loss = dis_bce_loss(tf.ones_like(real), real)
    fake_loss = dis_bce_loss(tf.zeros_like(fake), fake)
    return (real_loss + fake_loss) * 0.5


# Create cycle gan model
cycle_gan_model = CycleGan(
    generator_A=gen_A2B, generator_B=gen_B2A, discriminator_A=disc_A, discriminator_B=disc_B
)

