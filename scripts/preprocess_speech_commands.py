import os
from shutil import copyfile

import progressbar

from autoaudio.data import process_utterance, get_filelist


def preprocess_audio(in_dir, out_dir):
    file_list = get_filelist(in_dir)
    for file in progressbar.progressbar(file_list):
        folder = file.split('/')[-2]
        if not os.path.exists(os.path.join(out_dir, folder)):
            os.makedirs(os.path.join(out_dir, folder))

        details = process_utterance(out_dir, file)

    copyfile(os.path.join(in_dir, 'validation_list.txt'),
             os.path.join(out_dir, 'validation_list.txt'))

    copyfile(os.path.join(in_dir, 'testing_list.txt'),
             os.path.join(out_dir, 'testing_list.txt'))


if __name__ == '__main__':
    in_path = '/Users/archy/Downloads/speech_commands_v0.01'
    out_path = '/Users/archy/Downloads/speech_commands_v0.01_processed'

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    preprocess_audio(in_path, out_path)
