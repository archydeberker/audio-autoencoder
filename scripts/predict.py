import keras
import matplotlib.pyplot as plt

from autoaudio.data import AudioCommandDataset

data_path = '/Users/archy/Downloads/speech_commands_v0.01_processed'


def plot_spectrogram(x):
    plt.imshow(x[:, :].T)

    plt.xlabel('Time')
    plt.ylabel('Frequency')


# TODO: update this to use click
if __name__ == '__main__':

    batch_size = 1
    dataset = AudioCommandDataset(data_path, batch_size=batch_size)

    print('Loading data')
    test_batch_getter = dataset.get_batch(dataset.test_set)
    print('Loading model')
    autoencoder = keras.models.load_model('/Users/archy/Dropbox/Code/Python/audio-autoencoder/autoaudio/checkpoints/1531180827/weights.10-0.00.hdf5')

    plt.figure()
    for i in range(5):
        x, _ = next(test_batch_getter)

        x_hat = autoencoder.predict(x)

        plt.subplot(5, 2, 2*i+1)
        plot_spectrogram(x[0, :, :])
        plt.title('Original')

        plt.subplot(5, 2, 2*i+2)
        plot_spectrogram(x_hat[0, :, :])
        plt.title('Reconstructed')

    plt.show()

