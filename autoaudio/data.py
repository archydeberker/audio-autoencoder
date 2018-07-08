import glob
import os

import numpy as np
import scipy.io.wavfile as wav
from keras.preprocessing.sequence import pad_sequences

from autoaudio.utils import audio


class AudioCommandDataset():

    def __init__(self, data_path, batch_size=16, output_size=16000):

        self.data_path = data_path
        self.batch_size = batch_size
        self.file_list = get_filelist(self.data_path)

        self.val_set = self._get_file_paths(text_file='validation_list.txt')
        self.test_set = self._get_file_paths(text_file='testing_list.txt')

        self.train_set = self._get_train_paths()

        self.output_size = output_size


    def _get_train_paths(self):
        excluded = self.val_set.union(self.test_set)
        return set(self.file_list).difference(excluded)

    def _get_file_paths(self, text_file='validation_list.txt'):
        with open(os.path.join(self.data_path, text_file)) as f:
            file_list = f.readlines()

        file_list = [os.path.join(self.data_path, f).rstrip() for f in file_list]
        return set(file_list)

    def _random_filename(self, path_set):

        file_name = np.random.choice(self.file_list)
        print(file_name)
        while file_name not in path_set:
            file_name = np.random.choice(self.file_list)

        return file_name

    @staticmethod
    def load_audio(audio_file):

        _, data = wav.read(audio_file)

        return data

    def _preprocess_audio_batch(self, audio_batch):

        return pad_sequences(audio_batch, maxlen=self.output_size)

    def get_batch(self, path_set):

        while True:
            x = []
            for i in range(self.batch_size):
                x.append(self.load_audio(self._random_filename(path_set)))

            yield self._preprocess_audio_batch(np.asarray(x))


def get_filelist(data_path):

    file_list = []
    for folder in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, folder)) and '_background_noise_' not in folder:
            for file in glob.glob(os.path.join(data_path, folder, '*.wav')):

                file_list.append(os.path.join(data_path, folder, file))

    return file_list


def process_utterance(out_dir, wav_path):

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
