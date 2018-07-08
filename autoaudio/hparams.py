import tensorflow as tf
# Audio hparams, see https://github.com/keithito/tacotron/blob/master/hparams.py
hparams = tf.contrib.training.HParams(
               num_mels=80,
               num_freq=1025,
               sample_rate=16000,
               frame_length_ms=50,
               frame_shift_ms=12.5,
               preemphasis=0.97,
               min_level_db=-100,
               ref_level_db=20)
