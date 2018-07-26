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

    example = np.random.random((128, 80))
    example = np.expand_dims(example, axis=0)

    autoencoder = AutoEncoder()
    autoencoder.compile(optimizer=keras.optimizers.sgd(lr=0.1),
                        loss='mean_squared_error')

    history = autoencoder.fit(np.repeat(example, repeats=128, axis=0),
                              np.repeat(example, repeats=128, axis=0),
                              epochs=1000,
                              verbose=2)

    assert history.history['loss'][-1] < 0.01



