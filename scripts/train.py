import datetime
import os

import keras
from keras.callbacks import TensorBoard
from keras.optimizers import adam

from autoaudio.data import AudioCommandDataset
from autoaudio.model import AutoEncoder

data_path = '/Users/archy/Downloads/speech_commands_v0.01_processed'

if __name__ == '__main__':

    batch_size = 16
    dataset = AudioCommandDataset(data_path, batch_size=batch_size)

    train_batch_getter = dataset.get_batch(dataset.train_set)
    val_batch_getter = dataset.get_batch(dataset.train_set)

    autoencoder = AutoEncoder()

    autoencoder.compile(loss='mean_squared_error',
                        optimizer=adam())

    ckpt_folder = './%d/' % datetime.datetime.now().timestamp()

    ckpt = keras.callbacks.ModelCheckpoint('./%s/weights.{epoch:02d}-{val_loss:.2f}.hdf5' % ckpt_folder, monitor='val_loss',
                                           verbose=0,
                                           save_best_only=False, save_weights_only=False,
                                           mode='auto', period=1)

    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)


    autoencoder.fit_generator(train_batch_getter,
                              validation_data=val_batch_getter,
                              steps_per_epoch=1000,
                              validation_steps=100,
                              epochs=100,
                              callbacks=[TensorBoard(log_dir=ckpt_folder),
                                         ckpt])
