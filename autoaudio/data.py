import glob
import os

import numpy as np
import scipy.io.wavfile as wav
from keras.preprocessing.sequence import pad_sequences


class AudioCommandDataset():

    def __init__(self, data_path, batch_size=16, output_size=16000):

        self.data_path = data_path
        self.batch_size = batch_size
        self.file_list = self._get_filelist()

        self.val_set = self._get_file_paths(text_file='validation_list.txt')
        self.test_set = self._get_file_paths(text_file='testing_list.txt')

        self.train_set = self._get_train_paths()

        self.output_size = output_size

    def _get_filelist(self):

        file_list = []
        for folder in os.listdir(self.data_path):
            if os.path.isdir(os.path.join(self.data_path, folder)) and '_background_noise_' not in folder:
                for file in glob.glob(os.path.join(self.data_path, folder, '*.wav')):

                    file_list.append(os.path.join(self.data_path, folder, file))

        return file_list

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

