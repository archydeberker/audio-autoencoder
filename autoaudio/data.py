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
        self.output_size = output_size

    def _get_filelist(self):

        file_list = []
        for folder in os.listdir(self.data_path):
            if os.path.isdir(os.path.join(self.data_path, folder)) and '_background_noise_' not in folder:
                for file in glob.glob(os.path.join(self.data_path, folder, '*.wav')):

                    file_list.append(os.path.join(self.data_path, folder, file))

        return file_list

    def _random_filename(self):

        return np.random.choice(self.file_list)

    @staticmethod
    def load_audio(audio_file):

        _, data = wav.read(audio_file)

        return data

    def _preprocess_audio_batch(self, audio_batch):

        return pad_sequences(audio_batch, maxlen=self.output_size)

    def get_batch(self):

        while True:
            x = []
            for i in range(self.batch_size):
                x.append(self.load_audio(self._random_filename()))

            yield self._preprocess_audio_batch(np.asarray(x))

