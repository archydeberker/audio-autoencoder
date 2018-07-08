import keras
import numpy as np
import pytest

from autoaudio.model import AutoEncoder


def test_model_compiles():
    autoencoder = AutoEncoder()
    autoencoder.compile(optimizer=keras.optimizers.sgd(),
                        loss='mean_squared_error')


def test_model_summary():
    autoencoder = AutoEncoder()
    print(autoencoder.summary())


@pytest.mark.slow
def test_overfit_on_single_example():

    example = np.random.random(128)
    example = np.expand_dims(example, axis=1)
    example = np.expand_dims(example, axis=0)

    print(example.shape)

    autoencoder = AutoEncoder(input_shape=(128, 1))
    autoencoder.compile(optimizer=keras.optimizers.sgd(),
                        loss='mean_squared_error')

    history = autoencoder.fit([example], [example], epochs=1000)

    assert history.history['loss'][-1] < 0.01



