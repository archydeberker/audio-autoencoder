import numpy as np
import pytest

from data import AudioCommandDataset

data_path = '/Users/archy/Downloads/speech_commands_v0.01'


def test_file_list_is_correct_length():

    dataset = AudioCommandDataset(data_path)
    assert len(dataset.file_list) == 64721


def test_load_audio_into_numpy():

    dataset = AudioCommandDataset(data_path, batch_size=2)

    batch_getter = dataset.get_batch(dataset.train_set)

    assert isinstance(next(batch_getter), np.ndarray)


def test_length_of_batch_is_constant():
    """ Nb this is not a guarantee, can only sample so many times"""

    batch_size = 10
    dataset = AudioCommandDataset(data_path, batch_size=batch_size)
    batch_getter = dataset.get_batch(dataset.train_set)

    for i in range(100):
        batch = next(batch_getter)
        print(batch.shape)
        assert batch.shape[0] == batch_size
        assert batch.shape[1] == 16000


def test_file_sets_are_disjoint_given_disjoint_ids():
    dataset = AudioCommandDataset(data_path, batch_size=1)

    assert len(set(dataset.train_set).intersection(dataset.val_set)) is 0

    print(dataset.val_set)
    print(dataset.train_set)
    assert len(set(dataset.val_set).intersection(dataset.test_set)) is 0


def test_random_filename_generator():

    dataset = AudioCommandDataset(data_path, batch_size=1)

    sets = {'train': dataset.train_set,
            'val': dataset.val_set}

    data = {}
    for file_set in sets:
        print(sets[file_set])
        batch_getter = dataset.get_batch(sets[file_set])
        data[file_set] = []

        for i in range(10):
            data[file_set].append(next(batch_getter))

    with pytest.raises(AssertionError):
        np.testing.assert_almost_equal(data['train'], data['val'])






