import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Reshape, Conv2DTranspose, BatchNormalization, Dropout, Layer
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2

class Sampling(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.sampling = Sampling()

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = self.sampling((z_mean, z_log_var))
        reconstructed = self.decoder(z)
        reconstruction_loss = tf.keras.losses.MeanSquaredError()(inputs, reconstructed)
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        self.add_loss(vae_loss)
        return reconstructed

    @classmethod
    def from_config(cls, config, custom_objects=None):
        encoder_config = config['encoder']
        decoder_config = config['decoder']
        encoder = Model.from_config(encoder_config, custom_objects=custom_objects)
        decoder = Model.from_config(decoder_config, custom_objects=custom_objects)
        return cls(encoder, decoder)

    def get_config(self):
        config = super(VAE, self).get_config()
        config.update({
            'encoder': self.encoder.get_config(),
            'decoder': self.decoder.get_config(),
        })
        return config

# モデルの定義（エンコーダーとデコーダーを再定義）
latent_dim = 128
input_shape = (64, 64, 2)

# エンコーダー
encoder_input = Input(shape=input_shape)
x = Conv2D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(encoder_input)
x = BatchNormalization()(x)
x = MaxPooling2D(2, padding='same')(x)
x = Dropout(0.5)(x)
x = Conv2D(128, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2, padding='same')(x)
x = Dropout(0.5)(x)
x = Conv2D(256, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2, padding='same')(x)
x = Dropout(0.5)(x)
x = Flatten()(x)
x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)

z_mean = Dense(latent_dim)(x)
z_log_var = Dense(latent_dim)(x)

encoder = Model(encoder_input, [z_mean, z_log_var], name='encoder')

# デコーダー
decoder_input = Input(shape=(latent_dim,))
x = Dense(8*8*256, activation='relu', kernel_regularizer=l2(0.001))(decoder_input)
x = Reshape((8, 8, 256))(x)
x = Conv2DTranspose(256, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
x = UpSampling2D(2)(x)
x = Dropout(0.5)(x)
x = Conv2DTranspose(128, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
x = UpSampling2D(2)(x)
x = Dropout(0.5)(x)
x = Conv2DTranspose(64, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
x = UpSampling2D(2)(x)
outputs = Conv2DTranspose(2, 3, activation='sigmoid', padding='same', kernel_regularizer=l2(0.001))(x)

decoder = Model(decoder_input, outputs, name='decoder')

# VAEモデルをロード
model_path = '/Users/matsuokoujirou/Documents/Data/Screening/Noemal_cells/RG2_VAE_all_pyrenoid.h5'
autoencoder = load_model(model_path, custom_objects={'Sampling': Sampling, 'VAE': VAE})

# エンコーダーとデコーダーを取得
encoder = autoencoder.encoder
decoder = autoencoder.decoder

# エンコーダーとデコーダーを保存
encoder.save('/Users/matsuokoujirou/Documents/Data/Screening/Noemal_cells/RG2_encoder.h5')
decoder.save('/Users/matsuokoujirou/Documents/Data/Screening/Noemal_cells/RG2_decoder.h5')

print("Encoder and Decoder have been saved separately.")
