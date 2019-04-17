import tensorflow as tf
from tensorflow.contrib.signal.python.ops import window_ops
import numpy as np
import os

import Datasets
import Models.UnetSpectrogramSeparator
import Models.UnetAudioSeparator
import functools

def test(model_config, partition, model_folder, load_model):
    # Determine input and output shapes
    disc_input_shape = [model_config["batch_size"], model_config["num_frames"], 0]  # Shape of discriminator input
    if model_config["network"] == "unet":
        separator_class = Models.UnetAudioSeparator.UnetAudioSeparator(model_config)
    elif model_config["network"] == "unet_spectrogram":
        separator_class = Models.UnetSpectrogramSeparator.UnetSpectrogramSeparator(model_config)
    else:
        raise NotImplementedError

    sep_input_shape, sep_output_shape = separator_class.get_padding(np.array(disc_input_shape))
    separator_func = separator_class.get_output

    # Creating the batch generators
    assert ((sep_input_shape[1] - sep_output_shape[1]) % 2 == 0)
    dataset = Datasets.get_dataset(model_config, sep_input_shape, sep_output_shape, partition=partition)
    iterator = dataset.make_one_shot_iterator()
    batch = iterator.get_next()

    print("Testing...")

    # BUILD MODELS
    # Separator
    separator_sources = separator_func(batch["mix"], False, not model_config["raw_audio_loss"], reuse=False)  # Sources are output in order [acc, voice] for voice separation, [bass, drums, other, vocals] for multi-instrument separation

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False, dtype=tf.int64)

    # Start session and queue input threads
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(model_config["log_dir"] + os.path.sep +  model_folder, graph=sess.graph)

    # CHECKPOINTING
    # Load pretrained model to test
    restorer = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2)
    print("Num of variables" + str(len(tf.global_variables())))
    restorer.restore(sess, load_model)
    print('Pre-trained model restored for testing')

    # Start training loop
    _global_step = sess.run(global_step)
    print("Starting!")

    total_loss = 0.0
    batch_num = 1

    # Supervised objective: MSE for raw audio, MAE for magnitude space (Jansson U-Net)
    separator_loss = 0
    for key in model_config["source_names"]:
        real_source = batch[key]
        sep_source = separator_sources[key]

        if model_config["network"] == "unet_spectrogram" and not model_config["raw_audio_loss"]:
            window = functools.partial(window_ops.hann_window, periodic=True)
            stfts = tf.contrib.signal.stft(tf.squeeze(real_source, 2), frame_length=1024, frame_step=768,
                                           fft_length=1024, window_fn=window)
            real_mag = tf.abs(stfts)
            separator_loss += tf.reduce_mean(tf.abs(real_mag - sep_source))
        else:
            separator_loss += tf.reduce_mean(tf.square(real_source - sep_source))
    separator_loss = separator_loss / float(model_config["num_sources"])  # Normalise by number of sources

    while True:
        try:
            curr_loss = sess.run(separator_loss)
            total_loss = total_loss + (1.0 / float(batch_num)) * (curr_loss - total_loss)
            batch_num += 1
        except tf.errors.OutOfRangeError as e:
            break

    summary = tf.Summary(value=[tf.Summary.Value(tag="test_loss", simple_value=total_loss)])
    writer.add_summary(summary, global_step=_global_step)

    writer.flush()
    writer.close()

    print("Finished testing - Mean MSE: " + str(total_loss))

    # Close session, clear computational graph
    sess.close()
    tf.reset_default_graph()

    return total_loss