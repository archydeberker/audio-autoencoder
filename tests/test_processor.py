import numpy as np

from data import AudioCommandDataset

data_path = '/Users/archy/Downloads/speech_commands_v0.01'


def test_file_list_is_correct_length():

    dataset = AudioCommandDataset(data_path)
    assert len(dataset.file_list) == 64721


def test_load_audio_into_numpy():
    dataset = AudioCommandDataset(data_path, batch_size=2)

    batch_getter = dataset.get_batch()

    assert isinstance(next(batch_getter), np.ndarray)


def test_length_of_batch_is_constant():
    """ Nb this is not a guarantee, can only sample so many times"""

    batch_size = 10
    dataset = AudioCommandDataset(data_path, batch_size=batch_size)
    batch_getter = dataset.get_batch()

    for i in range(100):
        batch = next(batch_getter)
        print(batch.shape)
        assert batch.shape[0] == batch_size
        assert batch.shape[1] == 16000


