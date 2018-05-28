from sacred import Experiment
import tensorflow as tf
import numpy as np
import os

import Datasets
from Input import Input as Input
from Input import batchgenerators as batchgen
import Utils
import Models.UnetSpectrogramSeparator
import Models.UnetAudioSeparator
import cPickle as pickle
import Test
import Evaluate

import functools
from tensorflow.contrib.signal.python.ops import window_ops

ex = Experiment('Waveunet')

@ex.config
def cfg():
    # Base configuration
    model_config = {"musdb_path" : "/home/daniel/Datasets/MUSDB18", # SET MUSDB PATH HERE, AND SET CCMIXTER PATH IN CCMixter.xml
                    "estimates_path" : "/mnt/windaten/Source_Estimates", # SET THIS PATH TO WHERE YOU WANT SOURCE ESTIMATES PRODUCED BY THE TRAINED MODEL TO BE SAVED. Folder itself must exist!

                    "model_base_dir" : "checkpoints", # Base folder for model checkpoints
                    "log_dir" : "logs", # Base folder for logs files
                    "batch_size" : 16, # Batch size
                    "init_sup_sep_lr" : 1e-4, # Supervised separator learning rate
                    "epoch_it" : 2000, # Number of supervised separator steps per epoch
                    "num_disc": 5,  # Number of discriminator iterations per separator update
                    'cache_size' : 16, # Number of audio excerpts that are cached to build batches from
                    'num_workers' : 6, # Number of processes reading audio and filling up the cache
                    "duration" : 2, # Duration in seconds of the audio excerpts in the cache. Has to be at least the output length of the network!
                    'min_replacement_rate' : 16,  # roughly: how many cache entries to replace at least per batch on average. Can be fractional
                    'num_layers' : 12, # How many U-Net layers
                    'filter_size' : 15, # For Wave-U-Net: Filter size of conv in downsampling block
                    'merge_filter_size' : 5, # For Wave-U-Net: Filter size of conv in upsampling block
                    'num_initial_filters' : 24, # Number of filters for convolution in first layer of network
                    "num_frames": 16384, # DESIRED number of time frames in the output waveform per samples (could be changed when using valid padding)
                    'expected_sr': 22050,  # Downsample all audio input to this sampling rate
                    'mono_downmix': True,  # Whether to downsample the audio input
                    'output_type' : 'direct', # Type of output layer, either "direct" or "difference". Direct output: Each source is result of tanh activation and independent. DIfference: Last source output is equal to mixture input - sum(all other sources)
                    'context' : False, # Type of padding for convolutions in separator. If False, feature maps double or half in dimensions after each convolution, and convolutions are padded with zeros ("same" padding). If True, convolution is only performed on the available mixture input, thus the output is smaller than the input
                    'network' : 'unet', # Type of network architecture, either unet or dilated,
                    'upsampling' : 'linear', # Type of technique used for upsampling the feature maps in a unet architecture, either 'linear' interpolation or 'learned' filling in of extra samples
                    'task' : 'voice', # Type of separation task. 'voice' : Separate music into voice and accompaniment. 'multi_instrument': Separate music into guitar, bass, vocals, drums and other (Sisec)
                    'augmentation' : True, # Random attenuation of source signals to improve generalisation performance (data augmentation)
                    'raw_audio_loss' : True # Only active for unet_spectrogram network. True: L2 loss on audio. False: L1 loss on spectrogram magnitudes for training and validation and test loss
                    }
    seed=1337
    experiment_id = np.random.randint(0,1000000)

    model_config["num_sources"] = 4 if model_config["task"] == "multi_instrument" else 2
    model_config["num_channels"] = 1 if model_config["mono_downmix"] else 2

@ex.named_config
def baseline():
    print("Training baseline model")

@ex.named_config
def baseline_diff():
    print("Training baseline model with difference output")
    model_config = {
        "output_type" : "difference"
    }

@ex.named_config
def baseline_context():
    print("Training baseline model with difference output and input context (valid convolutions)")
    model_config = {
        "output_type" : "difference",
        "context" : True
    }

@ex.named_config
def baseline_stereo():
    print("Training baseline model with difference output and input context (valid convolutions)")
    model_config = {
        "output_type" : "difference",
        "context" : True,
        "mono_downmix" : False
    }

@ex.named_config
def full():
    print("Training full singing voice separation model, with difference output and input context (valid convolutions) and stereo input/output, and learned upsampling layer")
    model_config = {
        "output_type" : "difference",
        "context" : True,
        "upsampling": "learned",
        "mono_downmix" : False
    }

@ex.named_config
def baseline_context_smallfilter_deep():
    model_config = {
        "output_type": "difference",
        "context": True,
        "num_layers" : 14,
        "duration" : 7,
        "filter_size" : 5,
        "merge_filter_size" : 1
    }

@ex.named_config
def full_multi_instrument():
    print("Training multi-instrument separation with best model")
    model_config = {
        "output_type": "difference",
        "context": True,
        "upsampling": "linear",
        "mono_downmix": False,
        "task" : "multi_instrument"
    }

@ex.named_config
def dilated():
    model_config = {
        "num_initial_filters": 16,
        "batch_size": 2, # Less output since model is so big. Doesn't matter since the model's output is not dependent on its output or input size (only convolutions)
        "cache_size": 2,
        "min_replacement_rate" : 2,
        "network" : "dilated",
        "output_type": "difference",
        "context": True,
        "mono_downmix": False
    }
    print("Training dilated model with same settings as other best model")

@ex.named_config
def dilated_multi_instrument():
    model_config = {
        "num_initial_filters": 16,
        "batch_size": 2, # Less output since model is so big. Doesn't matter since the model's output is not dependent on its output or input size (only convolutions)
        "cache_size": 2,
        "min_replacement_rate" : 2,
        "network" : "dilated",
        "output_type": "difference",
        "context": True,
        "mono_downmix": False,
        "task": "multi_instrument",
    }
    print("Training multi-instrument, dilated model with same settings as other best model")

@ex.named_config
def baseline_comparison():
    model_config = {
        "batch_size": 4, # Less output since model is so big. Doesn't matter since the model's output is not dependent on its output or input size (only convolutions)
        "cache_size": 4,
        "min_replacement_rate" : 4,

        "output_type": "difference",
        "context": True,
        "num_frames" : 768*127 + 1024,
        "duration" : 13,
        "expected_sr" : 8192,
        "num_initial_filters" : 34
    }

@ex.named_config
def unet_spectrogram():
    model_config = {
        "batch_size": 4, # Less output since model is so big.
        "cache_size": 4,
        "min_replacement_rate" : 4,

        "network" : "unet_spectrogram",
        "num_layers" : 6,
        "expected_sr" : 8192,
        "num_frames" : 768 * 127 + 1024, # hop_size * (time_frames_of_spectrogram_input - 1) + fft_length
        "duration" : 13,
        "num_initial_filters" : 16
    }

@ex.named_config
def unet_spectrogram_l1():
    model_config = {
        "batch_size": 4, # Less output since model is so big.
        "cache_size": 4,
        "min_replacement_rate" : 4,

        "network" : "unet_spectrogram",
        "num_layers" : 6,
        "expected_sr" : 8192,
        "num_frames" : 768 * 127 + 1024, # hop_size * (time_frames_of_spectrogram_input - 1) + fft_length
        "duration" : 13,
        "num_initial_filters" : 16,
        "loss" : "magnitudes"
    }


@ex.capture
def train(model_config, experiment_id, sup_dataset, unsup_dataset=None, load_model=None):
    # Determine input and output shapes
    disc_input_shape = [model_config["batch_size"], model_config["num_frames"], 0]  # Shape of input
    if model_config["network"] == "unet":
        separator_class = Models.UnetAudioSeparator.UnetAudioSeparator(model_config["num_layers"], model_config["num_initial_filters"],
                                                                   output_type=model_config["output_type"],
                                                                   context=model_config["context"],
                                                                   mono=model_config["mono_downmix"],
                                                                   upsampling=model_config["upsampling"],
                                                                   num_sources=model_config["num_sources"],
                                                                   filter_size=model_config["filter_size"],
                                                                   merge_filter_size=model_config["merge_filter_size"])
    elif model_config["network"] == "unet_spectrogram":
        separator_class = Models.UnetSpectrogramSeparator.UnetSpectrogramSeparator(model_config["num_layers"], model_config["num_initial_filters"],
                                                                       mono=model_config["mono_downmix"],
                                                                       num_sources=model_config["num_sources"])
    else:
        raise NotImplementedError

    sep_input_shape, sep_output_shape = separator_class.get_padding(np.array(disc_input_shape))
    separator_func = separator_class.get_output

    # Creating the batch generators
    assert((sep_input_shape[1] - sep_output_shape[1]) % 2 == 0)
    pad_durations = np.array([float((sep_input_shape[1] - sep_output_shape[1])/2), 0, 0]) / float(model_config["expected_sr"])  # Input context that the input audio has to be padded ON EACH SIDE
    sup_batch_gen = batchgen.BatchGen_Paired(
        model_config,
        sup_dataset,
        sep_input_shape,
        sep_output_shape,
        pad_durations[0]
    )

    print("Starting worker")
    sup_batch_gen.start_workers()
    print("Started worker!")

    # Placeholders and input normalisation
    mix_context, sources = Input.get_multitrack_placeholders(sep_output_shape, model_config["num_sources"], sep_input_shape, "sup")
    mix = Utils.crop(mix_context, sep_output_shape)

    print("Training...")

    # BUILD MODELS
    # Separator
    separator_sources = separator_func(mix_context, True, not model_config["raw_audio_loss"], reuse=False) # Sources are output in order [acc, voice] for voice separation, [bass, drums, other, vocals] for multi-instrument separation

    # Supervised objective: MSE in log-normalized magnitude space
    separator_loss = 0
    for (real_source, sep_source) in zip(sources, separator_sources):
        if model_config["network"] == "unet_spectrogram" and not model_config["raw_audio_loss"]:
            window = functools.partial(window_ops.hann_window, periodic=True)
            stfts = tf.contrib.signal.stft(tf.squeeze(real_source, 2), frame_length=1024, frame_step=768,
                                           fft_length=1024, window_fn=window)
            real_mag = tf.abs(stfts)
            separator_loss += tf.reduce_mean(tf.abs(real_mag - sep_source))
        else:
            separator_loss += tf.reduce_mean(tf.square(real_source - sep_source))
    separator_loss = separator_loss / float(len(sources)) # Normalise by number of sources

    # TRAINING CONTROL VARIABLES
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False, dtype=tf.int64)
    increment_global_step = tf.assign(global_step, global_step + 1)
    sep_lr = tf.get_variable('unsup_sep_lr', [],initializer=tf.constant_initializer(model_config["init_sup_sep_lr"], dtype=tf.float32), trainable=False)

    # Set up optimizers
    separator_vars = Utils.getTrainableVariables("separator")
    print("Sep_Vars: " + str(Utils.getNumParams(separator_vars)))
    print("Num of variables" + str(len(tf.global_variables())))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        with tf.variable_scope("separator_solver"):
            separator_solver = tf.train.AdamOptimizer(learning_rate=sep_lr).minimize(separator_loss, var_list=separator_vars)

    # SUMMARAIES
    tf.summary.scalar("sep_loss", separator_loss, collections=["sup"])
    sup_summaries = tf.summary.merge_all(key='sup')

    # Start session and queue input threads
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(model_config["log_dir"] + os.path.sep + str(experiment_id),graph=sess.graph)

    # CHECKPOINTING
    # Load pretrained model to continue training, if we are supposed to
    if load_model != None:
        restorer = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2)
        print("Num of variables" + str(len(tf.global_variables())))
        restorer.restore(sess, load_model)
        print('Pre-trained model restored from file ' + load_model)

    saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2)

    # Start training loop
    run = True
    _global_step = sess.run(global_step)
    _init_step = _global_step
    it = 0
    while run:
        # TRAIN SEPARATOR
        sup_batch = sup_batch_gen.get_batch()
        feed = {i:d for i,d in zip(sources, sup_batch[1:])}
        feed.update({mix_context : sup_batch[0]})
        _, _sup_summaries = sess.run([separator_solver, sup_summaries], feed)
        writer.add_summary(_sup_summaries, global_step=_global_step)

        # Increment step counter, check if maximum iterations per epoch is achieved and stop in that case
        _global_step = sess.run(increment_global_step)

        if _global_step - _init_step > model_config["epoch_it"]:
            run = False
            print("Finished training phase, stopping batch generators")
            sup_batch_gen.stop_workers()

    # Epoch finished - Save model
    print("Finished epoch!")
    save_path = saver.save(sess, model_config["model_base_dir"] + os.path.sep + str(experiment_id) + os.path.sep + str(experiment_id), global_step=int(_global_step))

    # Close session, clear computational graph
    writer.flush()
    writer.close()
    sess.close()
    tf.reset_default_graph()

    return save_path

@ex.capture
def optimise(model_config, experiment_id, dataset):
    epoch = 0
    best_loss = 10000
    model_path = None
    best_model_path = None
    for i in range(2):
        worse_epochs = 0
        if i==1:
            print("Finished first round of training, now entering fine-tuning stage")
            model_config["batch_size"] *= 2
            model_config["cache_size"] *= 2
            model_config["min_replacement_rate"] *= 2
            model_config["init_sup_sep_lr"] = 1e-5
        while worse_epochs < 20: # Early stopping on validation set after a few epochs
            print("EPOCH: " + str(epoch))
            model_path = train(sup_dataset=dataset["train_sup"], load_model=model_path)
            curr_loss = Test.test(model_config, model_folder=str(experiment_id), audio_list=dataset["train_sup"], load_model=model_path)
            epoch += 1
            if curr_loss < best_loss:
                worse_epochs = 0
                print("Performance on validation set improved from " + str(best_loss) + " to " + str(curr_loss))
                best_model_path = model_path
                best_loss = curr_loss
            else:
                worse_epochs += 1
                print("Performance on validation set worsened to " + str(curr_loss))
    print("TRAINING FINISHED - TESTING WITH BEST MODEL " + best_model_path)
    test_loss = Test.test(model_config, model_folder=str(experiment_id), audio_list=dataset["test"], load_model=best_model_path)
    return best_model_path, test_loss

@ex.automain
def dsd_100_experiment(model_config):
    print("SCRIPT START")
    # Create subfolders if they do not exist to save results
    for dir in [model_config["model_base_dir"], model_config["log_dir"]]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    # Set up data input
    if os.path.exists('dataset.pkl'):
        with open('dataset.pkl', 'r') as file:
            dataset = pickle.load(file)
        print("Loaded dataset from pickle!")
    else:
        dsd_train, dsd_test = Datasets.getMUSDB(model_config["musdb_path"])
        ccm = Datasets.getCCMixter("CCMixter.xml")

        # Pick 25 random songs for validation from MUSDB train set (this is always the same selection each time since we fix the random seed!)
        val_idx = np.random.choice(len(dsd_train), size=25, replace=False)
        train_idx = [i for i in range(len(dsd_train)) if i not in val_idx]
        print("Validation with MUSDB training songs no. " + str(train_idx))

        # Draw randomly from datasets
        dataset = dict()
        dataset["train_sup"] = [dsd_train[i] for i in train_idx] + ccm
        dataset["train_unsup"] = list() #[dsd_train[0][25:], dsd_train[1][25:], dsd_train[2][25:]] #[fma, list(), looperman]
        dataset["valid"] = [dsd_train[i] for i in val_idx]
        dataset["test"] = dsd_test

        with open('dataset.pkl', 'wb') as file:
            pickle.dump(dataset,file)
        print("Created dataset structure")

    # Setup dataset depending on task. Dataset contains sources in order: (mix, acc, bass, drums, other, vocal)
    if model_config["task"] == "voice":
        for i in range(75):
            dataset["train_sup"][i] = (dataset["train_sup"][i][0], dataset["train_sup"][i][1], dataset["train_sup"][i][5])
        for subset in ["valid", "test"]:
            for i in range(len(dataset[subset])):
                dataset[subset][i] = (dataset[subset][i][0], dataset[subset][i][1], dataset[subset][i][5])
    else: # Multitask - Remove CCMixter from training, and acc source
        dataset["train_sup"] = dataset["train_sup"][:75]
        for subset in ["train_sup", "valid", "test"]:
            for i in range(len(dataset[subset])):
                dataset[subset][i] = (dataset[subset][i][0], dataset[subset][i][2], dataset[subset][i][3], dataset[subset][i][4], dataset[subset][i][5])

    # Optimize in a +supervised fashion until validation loss worsens
    sup_model_path, sup_loss = optimise(dataset=dataset)
    print("Supervised training finished! Saved model at " + sup_model_path + ". Performance: " + str(sup_loss))
    Evaluate.produce_source_estimates(model_config, sup_model_path, model_config["musdb_path"], model_config["estimates_path"], "train")
