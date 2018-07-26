import keras
import numpy as np
from keras import Model
from keras.layers import Input, MaxPooling1D, UpSampling1D, Conv1D, BatchNormalization, SeparableConv1D


def conv_block(x,
               kernel_size=45,
               filter_number=32,
               pool_size=4,
               ShapeChange=MaxPooling1D,
               **kwargs):

    x = Conv1D(filter_number, kernel_size, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = ShapeChange(pool_size, **kwargs)(x)

    return x


def encoder():
    def encoder(x):
        x = conv_block(x, ShapeChange=MaxPooling1D, padding='same')
        x = conv_block(x, ShapeChange=MaxPooling1D, padding='same')
        x = conv_block(x, ShapeChange=MaxPooling1D, padding='same')

        return x

    return encoder


def decoder():
    def decoder(x, n_output_channels=80):
        x = conv_block(x, ShapeChange=UpSampling1D)
        x = conv_block(x, ShapeChange=UpSampling1D)
        x = conv_block(x, ShapeChange=UpSampling1D)

        # Use a depthwise convolution to return something with a single dimension
        x = SeparableConv1D(filters=n_output_channels, kernel_size=16, padding='same')(x)

        return x

    return decoder


def AutoEncoder(input_shape=(None, 80)):
    input = Input(shape=input_shape, name='autoencoder_input')
    encoded = encoder()(input)
    decoded = decoder()(encoded)

    return Model(input, decoded)


if __name__ == '__main__':

    example = np.random.random((128, 80))
    example = np.expand_dims(example, axis=0)

    print(example.shape)

    autoencoder = AutoEncoder()
    autoencoder.compile(optimizer=keras.optimizers.sgd(),
                        loss='mean_squared_error')

    autoencoder.fit(np.repeat(example, axis=0),
                    np.repeat(example, axis=0),
                    epochs=1000,
                    verbose=1)

    print('done')