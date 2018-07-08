import numpy as np

from data import AudioCommandDataset, pad_batch, strip_suffix

data_path = '/Users/archy/Downloads/speech_commands_v0.01_processed'


def test_file_list_is_correct_length():

    dataset = AudioCommandDataset(data_path)
    assert len(dataset.file_list) == 64721


def test_load_audio_into_numpy():

    dataset = AudioCommandDataset(data_path, batch_size=2)

    batch_getter = dataset.get_batch(dataset.train_set)

    assert isinstance(next(batch_getter), np.ndarray)


def test_number_of_timesteps_in_batch_is_constant():
    """ Nb this is not a guarantee, can only sample so many times"""

    batch_size = 10
    dataset = AudioCommandDataset(data_path, batch_size=batch_size)
    batch_getter = dataset.get_batch(dataset.train_set)

    for i in range(100):
        batch = next(batch_getter)
        assert len(batch.shape) is 3
        assert batch.shape[0] == batch_size


def test_file_sets_are_disjoint_given_disjoint_ids():
    dataset = AudioCommandDataset(data_path, batch_size=1)

    assert len(set(dataset.train_set).intersection(dataset.val_set)) is 0

    print(dataset.val_set)
    print(dataset.train_set)
    assert len(set(dataset.val_set).intersection(dataset.test_set)) is 0


def test_random_filename_generator():

    dataset = AudioCommandDataset(data_path, batch_size=10)

    sets = {'train': dataset.train_set,
            'val': dataset.val_set}

    data = {}
    for file_set in sets:
        print(sets[file_set])
        batch_getter = dataset.get_batch(sets[file_set])
        data[file_set] = []

        for i in range(10):
            data[file_set].append(next(batch_getter))


def test_pad_batch():

    a = np.asarray([[0, 1, 2, 3], [0, 1, 2, 3]])
    b = np.asarray([[0, 1, 2, 3],  [0, 1, 2, 3], [0, 1, 2, 3]])

    assert a.shape == (2, 4)
    assert b.shape == (3, 4)

    batch = [a, b]

    padded_batch = pad_batch(batch)

    assert padded_batch.shape[0] == 2
    assert padded_batch.shape[1] == 3
    assert padded_batch.shape[2] == 4


def test_suffix_removal_for_file_inclusion():

    path1 = 'bed/my_file_0_mel.npy'
    path2 = 'bed/my_file_0.wav'

    assert strip_suffix(path1) == strip_suffix(path2)
