import numpy as np
import tensorflow as tf
import librosa

import os
import json
import glob

import Models.UnetAudioSeparator
import Models.UnetSpectrogramSeparator

import musdb
import museval
import Utils

def predict(track, model_config, load_model, results_dir=None):
    '''
    Function in accordance with MUSB evaluation API. Takes MUSDB track object and computes corresponding source estimates, as well as calls evlauation script.
    Model has to be saved beforehand into a pickle file containing model configuration dictionary and checkpoint path!
    :param track: Track object
    :param results_dir: Directory where SDR etc. values should be saved
    :return: Source estimates dictionary
    '''

    # Determine input and output shapes, if we use U-net as separator
    disc_input_shape = [model_config["batch_size"], model_config["num_frames"], 0]  # Shape of discriminator input
    if model_config["network"] == "unet":
        separator_class = Models.UnetAudioSeparator.UnetAudioSeparator(model_config)
    elif model_config["network"] == "unet_spectrogram":
        separator_class = Models.UnetSpectrogramSeparator.UnetSpectrogramSeparator(model_config)
    else:
        raise NotImplementedError

    sep_input_shape, sep_output_shape = separator_class.get_padding(np.array(disc_input_shape))
    separator_func = separator_class.get_output

    # Batch size of 1
    sep_input_shape[0] = 1
    sep_output_shape[0] = 1

    mix_ph = tf.placeholder(tf.float32, sep_input_shape)

    print("Testing...")

    # BUILD MODELS
    # Separator
    separator_sources = separator_func(mix_ph, training=False, return_spectrogram=False, reuse=False)

    # Start session and queue input threads
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Load model
    # Load pretrained model to continue training, if we are supposed to
    restorer = tf.train.Saver(None, write_version=tf.train.SaverDef.V2)
    print("Num of variables" + str(len(tf.global_variables())))
    restorer.restore(sess, load_model)
    print('Pre-trained model restored for song prediction')

    mix_audio, orig_sr, mix_channels = track.audio, track.rate, track.audio.shape[1] # Audio has (n_samples, n_channels) shape
    separator_preds = predict_track(model_config, sess, mix_audio, orig_sr, sep_input_shape, sep_output_shape, separator_sources, mix_ph)

    # Upsample predicted source audio and convert to stereo. Make sure to resample back to the exact number of samples in the original input (with fractional orig_sr/new_sr this causes issues otherwise)
    pred_audio = {name : Utils.resample(separator_preds[name], model_config["expected_sr"], orig_sr)[:mix_audio.shape[0],:] for name in model_config["source_names"]}

    if model_config["mono_downmix"] and mix_channels > 1: # Convert to multichannel if mixture input was multichannel by duplicating mono estimate
        pred_audio = {name : np.tile(pred_audio[name], [1, mix_channels]) for name in list(pred_audio.keys())}

    # Evaluate using museval, if we are currently evaluating MUSDB
    if results_dir is not None:
        scores = museval.eval_mus_track(track, pred_audio, output_dir=results_dir)

        # print nicely formatted mean scores
        print(scores)

    # Close session, clear computational graph
    sess.close()
    tf.reset_default_graph()

    return pred_audio

def predict_track(model_config, sess, mix_audio, mix_sr, sep_input_shape, sep_output_shape, separator_sources, mix_context):
    '''
    Outputs source estimates for a given input mixture signal mix_audio [n_frames, n_channels] and a given Tensorflow session and placeholders belonging to the prediction network.
    It iterates through the track, collecting segment-wise predictions to form the output.
    :param model_config: Model configuration dictionary
    :param sess: Tensorflow session used to run the network inference
    :param mix_audio: [n_frames, n_channels] audio signal (numpy array). Can have higher sampling rate or channels than the model supports, will be downsampled correspondingly.
    :param mix_sr: Sampling rate of mix_audio
    :param sep_input_shape: Input shape of separator ([batch_size, num_samples, num_channels])
    :param sep_output_shape: Input shape of separator ([batch_size, num_samples, num_channels])
    :param separator_sources: List of Tensorflow tensors that represent the output of the separator network
    :param mix_context: Input tensor of the network
    :return:
    '''
    # Load mixture, convert to mono and downsample then
    assert(len(mix_audio.shape) == 2)
    if model_config["mono_downmix"]:
        mix_audio = np.mean(mix_audio, axis=1, keepdims=True)
    else:
        if mix_audio.shape[1] == 1:# Duplicate channels if input is mono but model is stereo
            mix_audio = np.tile(mix_audio, [1, 2])

    mix_audio = Utils.resample(mix_audio, mix_sr, model_config["expected_sr"])

    # Append zeros to mixture if its shorter than input size of network - this will be cut off at the end again
    if mix_audio.shape[0] < sep_input_shape[1]:
        extra_pad = sep_input_shape[1] - mix_audio.shape[0]
        mix_audio = np.pad(mix_audio, [(0, extra_pad), (0,0)], mode="constant", constant_values=0.0)
    else:
        extra_pad = 0

    # Preallocate source predictions (same shape as input mixture)
    source_time_frames = mix_audio.shape[0]
    source_preds = {name : np.zeros(mix_audio.shape, np.float32) for name in model_config["source_names"]}

    input_time_frames = sep_input_shape[1]
    output_time_frames = sep_output_shape[1]

    # Pad mixture across time at beginning and end so that neural network can make prediction at the beginning and end of signal
    pad_time_frames = (input_time_frames - output_time_frames) // 2
    mix_audio_padded = np.pad(mix_audio, [(pad_time_frames, pad_time_frames), (0,0)], mode="constant", constant_values=0.0)

    # Iterate over mixture magnitudes, fetch network rpediction
    for source_pos in range(0, source_time_frames, output_time_frames):
        # If this output patch would reach over the end of the source spectrogram, set it so we predict the very end of the output, then stop
        if source_pos + output_time_frames > source_time_frames:
            source_pos = source_time_frames - output_time_frames

        # Prepare mixture excerpt by selecting time interval
        mix_part = mix_audio_padded[source_pos:source_pos + input_time_frames,:]
        mix_part = np.expand_dims(mix_part, axis=0)

        source_parts = sess.run(separator_sources, feed_dict={mix_context: mix_part})

        # Save predictions
        # source_shape = [1, freq_bins, acc_mag_part.shape[2], num_chan]
        for name in model_config["source_names"]:
            source_preds[name][source_pos:source_pos + output_time_frames] = source_parts[name][0, :, :]

    # In case we had to pad the mixture at the end, remove those samples from source prediction now
    if extra_pad > 0:
        source_preds = {name : source_preds[name][:-extra_pad,:] for name in list(source_preds.keys())}

    return source_preds

def produce_musdb_source_estimates(model_config, load_model, musdb_path, output_path, subsets=None):
    '''
    Predicts source estimates for MUSDB for a given model checkpoint and configuration, and evaluate them.
    :param model_config: Model configuration of the model to be evaluated
    :param load_model: Model checkpoint path
    :return: 
    '''
    print("Evaluating trained model saved at " + str(load_model)+ " on MUSDB and saving source estimate audio to " + str(output_path))

    mus = musdb.DB(root_dir=musdb_path)
    predict_fun = lambda track : predict(track, model_config, load_model, output_path)
    assert(mus.test(predict_fun))
    mus.run(predict_fun, estimates_dir=output_path, subsets=subsets)

def produce_source_estimates(model_config, load_model, input_path, output_path=None):
    '''
    For a given input mixture file, saves source predictions made by a given model.
    :param model_config: Model configuration
    :param load_model: Model checkpoint path
    :param input_path: Path to input mixture audio file
    :param output_path: Output directory where estimated sources should be saved. Defaults to the same folder as the input file, if not given
    :return: Dictionary of source estimates containing the source signals as numpy arrays
    '''
    print("Producing source estimates for input mixture file " + input_path)
    # Prepare input audio as track object (in the MUSDB sense), so we can use the MUSDB-compatible prediction function
    audio, sr = Utils.load(input_path, sr=None, mono=False)
    # Create something that looks sufficiently like a track object to our MUSDB function
    class TrackLike(object):
        def __init__(self, audio, rate, shape):
            self.audio = audio
            self.rate = rate
            self.shape = shape
    track = TrackLike(audio, sr, audio.shape)

    sources_pred = predict(track, model_config, load_model) # Input track to prediction function, get source estimates

    # Save source estimates as audio files into output dictionary
    input_folder, input_filename = os.path.split(input_path)
    if output_path is None:
        # By default, set it to the input_path folder
        output_path = input_folder
    if not os.path.exists(output_path):
        print("WARNING: Given output path " + output_path + " does not exist. Trying to create it...")
        os.makedirs(output_path)
    assert(os.path.exists(output_path))
    for source_name, source_audio in list(sources_pred.items()):
        librosa.output.write_wav(os.path.join(output_path, input_filename) + "_" + source_name + ".wav", source_audio, sr)

def compute_mean_metrics(json_folder, compute_averages=True, metric="SDR"):
    '''
    Computes averages or collects evaluation metrics produced from MUSDB evaluation of a separator
     (see "produce_musdb_source_estimates" function), namely the mean, standard deviation, median, and median absolute
     deviation (MAD). Function is used to produce the results in the paper.
     Averaging ignores NaN values arising from parts where a source is silent
    :param json_folder: Path to the folder in which a collection of json files was written by the MUSDB evaluation library, one for each song.
    This is the output of the "produce_musdb_source_estimates" function.(By default, this is model_config["estimates_path"] + test or train)
    :param compute_averages: Whether to compute the average over all song segments (to get final evaluation measures) or to return the full list of segments
    :param metric: Which metric to evaluate (either "SDR", "SIR", "SAR" or "ISR")
    :return: IF compute_averages is True, returns a list with length equal to the number of separated sources, with each list element a tuple of (median, MAD, mean, SD).
    If it is false, also returns this list, but each element is now a numpy vector containing all segment-wise performance values
    '''
    files = glob.glob(os.path.join(json_folder, "*.json"))
    inst_list = None
    print("Found " + str(len(files)) + " JSON files to evaluate...")
    for path in files:
        #print(path)
        if path.__contains__("test.json"):
            print("Found test JSON, skipping...")
            continue

        with open(path, "r") as f:
            js = json.load(f)

        if inst_list is None:
            inst_list = [list() for _ in range(len(js["targets"]))]

        for i in range(len(js["targets"])):
            inst_list[i].extend([np.float(f['metrics'][metric]) for f in js["targets"][i]["frames"]])

    #return np.array(sdr_acc), np.array(sdr_voc)
    inst_list = [np.array(perf) for perf in inst_list]

    if compute_averages:
        return [(np.nanmedian(perf), np.nanmedian(np.abs(perf - np.nanmedian(perf))), np.nanmean(perf), np.nanstd(perf)) for perf in inst_list]
    else:
        return inst_list