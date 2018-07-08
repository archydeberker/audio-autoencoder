import glob
import os

import numpy as np

from autoaudio.utils import audio


class AudioCommandDataset():
    """ Class for handling pre-processed audio command datasets.

    By default, loads mel spectrograms."""

    def __init__(self, data_path, batch_size=16):

        self.data_path = data_path
        self.batch_size = batch_size

        self.file_list = get_filelist(self.data_path, suffix='_mel.npy')

        self.val_set = self._get_file_paths(text_file='validation_list.txt')
        self.test_set = self._get_file_paths(text_file='testing_list.txt')

        self.train_set = self._get_train_paths()

    def _get_train_paths(self):
        excluded = self.val_set.union(self.test_set)
        return set(self.file_list).difference(excluded)

    def _get_file_paths(self, text_file='validation_list.txt'):
        with open(os.path.join(self.data_path, text_file)) as f:
            file_list = f.readlines()

        file_list = [os.path.join(self.data_path, f).rstrip() for f in file_list]

        # Strip file suffixes to allow us to compare these lists to pre-processed data
        file_list = [self._strip_suffix(f) for f in file_list]

        return set(file_list)

    @staticmethod
    def _strip_suffix(path):
        """ Strips the suffixes introduced by pre-processing, to allow us to compare
        file paths to the validation and test set lists.
        """

        suffixes = ['.wav', '.npy', 'spec', 'mel']

        path_components = path.split('_')

        return '_'.join([p for p in path_components if p not in suffixes])

    def _random_filename(self, path_set):

        file_name = self._strip_suffix(np.random.choice(self.file_list))
        while file_name not in path_set:
            file_name = self._strip_suffix(np.random.choice(self.file_list))

        return file_name

    def get_batch(self, path_set):

        while True:
            x = []
            for i in range(self.batch_size):
                x.append(np.load(self._random_filename(path_set)))

            yield np.asarray(x)


def get_filelist(data_path, suffix='.wav'):
    """ Retrieve a list of all files in the speech_commands dataset hierarchy."""
    file_list = []
    for folder in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, folder)) and '_background_noise_' not in folder:
            for file in glob.glob(os.path.join(data_path, folder, '*'+suffix)):

                file_list.append(os.path.join(data_path, folder, file))

    return file_list


def process_utterance(out_dir, wav_path):

    """Convert an audio clip into spectogram and mel spectrogram."""

    wav = audio.load_wav(wav_path)

    spectrogram = audio.spectrogram(wav).astype(np.float32)
    n_frames = spectrogram.shape[1]
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)

    path_pieces = wav_path.split('/')[-2:]
    base_name = path_pieces[0] + '/' + path_pieces[-1].split('.')[0] + '_%s.npy'

    spectrogram_filename = base_name % 'spec'
    mel_filename = base_name % 'mel'

    np.save(os.path.join(out_dir, spectrogram_filename), spectrogram.T, allow_pickle=False)
    np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)

    return (spectrogram_filename, mel_filename, n_frames)
