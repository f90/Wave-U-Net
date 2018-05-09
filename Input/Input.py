import os.path
import subprocess
import warnings

import librosa
import numpy as np
import skimage.io as io
import tensorflow as tf
from soundfile import SoundFile

import Metadata


def createSynthAudioBatch(batch_size, num_frames):
    '''
    Create three batches of audio examples [batch_size, freq_bins, time_frames, 1] and return them as a list
    [music, accompaniment, voice]
    :param batch_size: Number of examples in each batch
    :param num_frames: Number of timeframes in each sample
    :return: List of length three, each entry a numpy array of shape [batch_size, freq_bins, time_frames, 1]
    '''
    mixes, accs, voices = list(), list(), list()
    for i in range(batch_size):
        mix, acc, voice = createSynthAudio(0.5)
        mixes.append(mix[:,1:num_frames+1])
        accs.append(acc[:,1:num_frames+1])
        voices.append(voice[:,1:num_frames+1])
    mixes = np.array(mixes)
    accs = np.array(accs)
    voices = np.array(voices)

    return [mixes[:,:,:,np.newaxis], accs[:,:,:,np.newaxis], voices[:,:,:,np.newaxis]]

def createSynthAudio(signal_length, sample_rate=22050):
    '''
    Create the magnitude spectrograms for synthetic mixture, accompaniment, and voice sample. Voice is active with
    a 50% chance and is a single sinusoid. Accompaniment are four harmonic sinusoids and added to the voice to yield the mixture.
    :param signal_length: Signal length of the audio to be used in seconds
    :param sample_rate: Sampling rate of the audio signal
    :return: 
    '''
    x = np.linspace(0.0, signal_length, int(signal_length * float(sample_rate)))

    # Create voice
    voice_active = (np.random.randint(0, 2) == 1)
    if voice_active: # 50/50 chance between silence and sinusoid
        voice_freq = np.random.uniform(80.0, 220.0)
        voice_phase = np.random.uniform(-np.pi,np.pi)
        voice = 0.1 * np.sin(2 * np.pi * voice_freq * x + voice_phase)
    else:
        voice = np.zeros(int(signal_length * float(sample_rate)),dtype=np.float32)

    # Create accompaniment
    if voice_active and voice_freq > 150.0: # High frequency: Overlapping acc and voice F0
        acc_freq = voice_freq
    else:
        acc_freq = np.random.uniform(40.0, 300.0) # Low frequency: Random acc F0 (this introduces acc-voice correlations)
    acc_phase = np.random.uniform(-np.pi,np.pi)
    acc = np.zeros(int(signal_length * float(sample_rate)),dtype=np.float32)
    for harmonic in range(1,6):
        acc += 0.04 * np.sin(2 * np.pi * acc_freq * harmonic * x + acc_phase)

    mix = voice + acc
    mix_mag, _ = audioFileToSpectrogram(mix)
    acc_mag, _ = audioFileToSpectrogram(acc)
    voice_mag, _ = audioFileToSpectrogram(voice)
    return mix_mag[:-1,:], acc_mag[:-1,:], voice_mag[:-1,:]

def get_multitrack_placeholders(shape, num_sources, input_shape=None, name=""):
    '''
    Creates Tensorflow placeholders for mixture, accompaniment, and voice.
    :param shape: Shape of each individual sample
    :return: List of multitrack placeholders for mixture, accompaniment, and voice
    '''
    if input_shape is None:
        input_shape = shape
    mix = (tf.placeholder(dtype=tf.float32, shape=input_shape, name="mix_input" + name))

    sources = list()
    for i in range(num_sources):
        sources.append(tf.placeholder(dtype=tf.float32, shape=shape, name="source_" + str(i) + "_input" + name))
    return mix, sources

def get_multitrack_input(shape, batch_size, name="", input_shape=None):
    '''
    Creates multitrack placeholders and a random shuffle queue based on it
    :param input_shape: Shape of accompaniment and voice magnitudes
    :param batch_size: Number of samples in each batch
    :param name: How to name the placeholders
    :return: [List of mixture,acc,voice placeholders, random shuffle queue, symbolic batch sample from queue]
    '''
    m,a,v = get_multitrack_placeholders(shape, input_shape=input_shape)

    min_after_dequeue = 0
    buffer = 1000
    capacity = min_after_dequeue + buffer

    if input_shape is None:
        input_shape = shape
    queue = tf.RandomShuffleQueue(capacity, min_after_dequeue, [tf.float32, tf.float32, tf.float32], [input_shape, shape, shape])
    input_batch = queue.dequeue_many(batch_size, name="input_batch" + name)

    return [m,a,v], queue, input_batch


def random_amplify(magnitude):
    '''
    Randomly amplifies or attenuates the input magnitudes
    :param magnitude: SINGLE Magnitude spectrogram excerpt, or list of spectrogram excerpts that each have their own amplification factor
    :return: Amplified magnitude spectrogram
    '''
    if isinstance(magnitude, np.ndarray):
        return np.random.uniform(0.7, 1.0) * magnitude
    else:
        assert(isinstance(magnitude, list))
        factor = np.random.uniform(0.7, 1.0)
        for i in range(len(magnitude)):
            magnitude[i] = factor * magnitude[i]
        return magnitude

def randomPositionInAudio(audio_path, duration):
    length = librosa.get_duration(filename=audio_path)
    if duration >= length:
        return 0.0, None
    else:
        offset = np.random.uniform() * (length - duration)
        return offset, duration

def readWave(audio_path, start_frame, end_frame, mono=True, sample_rate=None, clip=True):
    snd_file = SoundFile(audio_path, mode='r')
    inf = snd_file._info
    audio_sr = inf.samplerate

    start_read = max(start_frame, 0)
    pad_front = -min(start_frame, 0)
    end_read = min(end_frame, inf.frames)
    pad_back = max(end_frame - inf.frames, 0)

    snd_file.seek(start_read)
    audio = snd_file.read(end_read - start_read, dtype='float32', always_2d=True) # (num_frames, channels)
    snd_file.close()

    # Pad if necessary (start_frame or end_frame out of bounds)
    audio = np.pad(audio, [(pad_front, pad_back), (0, 0)], mode="constant", constant_values=0.0)

    # Convert to mono if desired
    if mono:
        audio = np.mean(audio, axis=1, keepdims=True)

    # Resample if needed
    if sample_rate is not None and sample_rate != audio_sr:
        res_length = int(np.ceil(float(audio.shape[0]) * float(sample_rate) / float(audio_sr)))
        audio = np.pad(audio, [(1, 1), (0,0)], mode="reflect")  # Pad audio first
        audio = librosa.resample(audio.T, audio_sr, sample_rate, res_type="kaiser_fast").T
        skip = (audio.shape[0] - res_length) // 2
        audio = audio[skip:skip+res_length,:]

    # Clip to [-1,1] if desired
    if clip:
        audio = np.minimum(np.maximum(audio, -1.0), 1.0)

    return audio, audio_sr

def readAudio(audio_path, offset=0.0, duration=None, mono=True, sample_rate=None, clip=True, pad_frames=0, metadata=None):
    '''
    Reads an audio file wholly or partly, and optionally converts it to mono and changes sampling rate.
    By default, it loads the whole audio file. If the offset is set to None, the duration HAS to be not None,
    and the offset is then randomly determined so that a random section of the audio is selected with the desired duration.
    Optionally, the file can be zero-padded by a certain amount of seconds at the start and end before selecting this random section.

    :param audio_path: Path to audio file
    :param offset: Position in audio file (s) where to start reading. If None, duration has to be not None, and position will be randomly determined.
    :param duration: How many seconds of audio to read
    :param mono: Convert to mono after reading
    :param sample_rate: Convert to given sampling rate if given
    :param pad_frames: number of frames with wich to pad the audio at most if the samples at the borders are not available
    :param metadata: metadata about audio file, accelerates reading audio since duration does not need to be determined from file 
    :return: Audio signal, Audio sample rate
    '''

    if os.path.splitext(audio_path)[1][1:].lower() == "mp3":  # If its an MP3, call ffmpeg with offset and duration parameters
        # Get mp3 metadata information and duration
        if metadata is None:
            audio_sr, audio_channels, audio_duration = Metadata.get_mp3_metadata(audio_path)
        else:
            audio_sr = metadata[0]
            audio_channels = metadata[1]
            audio_duration = metadata[2]
        print(audio_duration)

        pad_front_duration = 0.0
        pad_back_duration = 0.0

        ref_sr = sample_rate if sample_rate is not None else audio_sr
        padding_duration = float(pad_frames) / float(ref_sr)

        if offset is None:  # In this case, select random section of audio file
            assert (duration is not None)
            max_start_pos = audio_duration+2*padding_duration-duration
            if (max_start_pos <= 0.0):  # If audio file is longer than duration of desired section, take all of it, will be padded later
                print("WARNING: Audio file " + audio_path + " has length " + str(audio_duration) + " but is expected to be at least " + str(duration))
                return librosa.load(audio_path, sample_rate, mono, res_type='kaiser_fast')  # Return whole audio file
            start_pos = np.random.uniform(0.0,max_start_pos) # Otherwise randomly determine audio section, taking padding on both sides into account
            offset = max(start_pos - padding_duration, 0.0) # Read from this position in audio file
            pad_front_duration = max(padding_duration - start_pos, 0.0)
        assert (offset is not None)

        if duration is not None: # Adjust duration if it overlaps with end of track
            pad_back_duration = max(offset + duration - audio_duration, 0.0)
            duration = duration - pad_front_duration - pad_back_duration # Subtract padding from the amount we have to read from file
        else: # None duration: Read from offset to end of file
            duration = audio_duration - offset

        pad_front_frames = int(pad_front_duration * float(audio_sr))
        pad_back_frames = int(pad_back_duration * float(audio_sr))


        args = ['ffmpeg', '-noaccurate_seek',
                '-ss', str(offset),
                '-t', str(duration),
                '-i', audio_path,
                '-f', 's16le', '-']

        audio = []
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=open(os.devnull, 'wb'))
        num_reads = 0
        while True:
            output = process.stdout.read(4096)
            if output == '' and process.poll() is not None:
                break
            if output:
                audio.append(librosa.util.buf_to_float(output, dtype=np.float32))
                num_reads += 1

        audio = np.concatenate(audio)
        if audio_channels > 1:
            audio = audio.reshape((-1, audio_channels)).T

    else: #Not an MP3: Handle with PySoundFile
        # open audio file
        snd_file = SoundFile(audio_path, mode='r')
        inf = snd_file._info
        audio_sr = inf.samplerate

        pad_orig_frames = pad_frames if sample_rate is None else int(np.ceil(float(pad_frames) * (float(audio_sr) / float(sample_rate))))

        pad_front_frames = 0
        pad_back_frames = 0

        if offset is not None and duration is not None:
            start_frame = int(offset * float(audio_sr))
            read_frames = int(duration * float(audio_sr))
        elif offset is not None and duration is None:
            start_frame = int(offset * float(audio_sr))
            read_frames = inf.frames - start_frame
        else:  # In this case, select random section of audio file
            assert (offset is None)
            assert (duration is not None)
            num_frames = int(duration * float(audio_sr))
            max_start_pos = inf.frames - num_frames # Maximum start position when ignoring padding on both ends of the file
            if (max_start_pos <= 0):  # If audio file is longer than duration of desired section, take all of it, will be padded later
                print("WARNING: Audio file " + audio_path + " has frames  " + str(inf.frames) + " but is expected to be at least " + str(num_frames))
                raise Exception("Could not read minimum required amount of audio data")
                #return librosa.load(audio_path, sample_rate, mono, res_type='kaiser_fast')  # Return whole audio file
            start_pos = np.random.randint(0, max_start_pos)  # Otherwise randomly determine audio section, taking padding on both sides into account

            # Translate source position into mixture input positions (take into account padding)
            start_mix_pos = start_pos - pad_orig_frames
            num_mix_frames = num_frames + 2*pad_orig_frames
            end_mix_pos = start_mix_pos + num_mix_frames

            # Now see how much of the mixture is available, pad the rest with zeros

            start_frame = max(start_mix_pos, 0)
            end_frame = min(end_mix_pos, inf.frames)
            read_frames = end_frame - start_frame
            pad_front_frames = -min(start_mix_pos, 0)
            pad_back_frames = max(end_mix_pos - inf.frames, 0)

        assert(num_frames > 0)
        snd_file.seek(start_frame)
        audio = snd_file.read(read_frames, dtype='float32', always_2d=True)
        snd_file.close()

        centre_start_frame = start_pos
        centre_end_frame = start_pos + num_frames

    # Pad as indicated at beginning and end
    audio = np.pad(audio, [(pad_front_frames, pad_back_frames), (0,0)],mode="constant",constant_values=0.0)

    # Convert to mono if desired
    if mono:
        audio = np.mean(audio, axis=1, keepdims=True)

    # Resample if needed
    if sample_rate is not None and sample_rate != audio_sr:
        audio = librosa.resample(audio.T, audio_sr, sample_rate, res_type="kaiser_fast").T

    # Clip to [-1,1] if desired
    if clip:
        audio = np.minimum(np.maximum(audio, -1.0), 1.0)

    if float(audio.shape[0])/float(sample_rate) < 1.0:
        raise IOError("Error while reading " + audio_path + " - ended up with audio shorter than one second!")

    if os.path.splitext(audio_path)[1][1:].lower() == "mp3":
        return audio, audio_sr, offset, offset+duration
    else:
        return audio, audio_sr, centre_start_frame, centre_end_frame, start_mix_pos, end_mix_pos

# Return a 2d numpy array of the spectrogram
def audioFileToSpectrogram(audioIn, fftWindowSize=1024, hopSize=512, offset=0.0, duration=None, expected_sr=None, buffer=False, padding_duration=0.0, metadata=None):
    '''
    Audio to FFT magnitude and phase conversion. Input can be a filepath to an audio file or a numpy array directly.
    By default, the whole audio is used for conversion. By setting duration to the desired number of seconds to be read from the audio file,
    reading can be sped up.
    For accelerating reading, the buffer option can be activated so that a numpy filedump of the magnitudes
    and phases is created after processing and loaded the next time it is requested.
    :param audioIn: 
    :param fftWindowSize: 
    :param hopSize: 
    :param offset: 
    :param duration: 
    :param expected_sr: 
    :param buffer: 
    :return: 
    '''

    writeNumpy = False
    if isinstance(audioIn, str): # Read from file
        if buffer and os.path.exists(audioIn + ".npy"): # Do we need to load a previous numpy buffer file?
            assert(offset == 0.0 and duration is None) # We can only load the whole buffer file
            with open(audioIn + ".npy", 'r') as file: # Try loading
                try:
                    [magnitude, phase] = np.load(file)
                    return magnitude, phase
                except Exception as e: # In case loading did not work, remember and overwrite file later
                    print("Could not load " + audioIn + ".npy. Loading audio again and recreating npy file!")
                    writeNumpy = True
        audio, sample_rate, _ , _= readAudio(audioIn, duration=duration, offset=offset, sample_rate=expected_sr, padding_duration=padding_duration, metadata=metadata) # If no buffering, read audio file
    else: # Input is already a numpy array
        assert(expected_sr is None and duration is None and offset == 0.0) # Make sure no other options are active
        audio = audioIn

    # Compute magnitude and phase
    spectrogram = librosa.stft(audio, fftWindowSize, hopSize)
    magnitude, phase = librosa.core.magphase(spectrogram)
    phase = np.angle(phase) # from e^(1j * phi) to phi
    assert(np.max(magnitude) < fftWindowSize and np.min(magnitude) >= 0.0)

    # Buffer results if desired
    if (buffer and ((not os.path.exists(audioIn + ".npy")) or  writeNumpy)):
        np.save(audioIn + ".npy", [magnitude, phase])

    return magnitude, phase

def add_audio(audio_list, path_postfix):
    '''
    Reads in a list of audio files, sums their signals, and saves them in new audio file which is named after the first audio file plus a given postfix string
    :param audio_list: List of audio file paths
    :param path_postfix: Name to append to the first given audio file path in audio_list which is then used as save destination
    :return: Audio file path where the sum signal was saved
    '''
    save_path = audio_list[0] + "_" + path_postfix + ".wav"
    if not os.path.exists(save_path):
        for idx, instrument in enumerate(audio_list):
            instrument_audio, sr = librosa.load(instrument, sr=None)
            if idx == 0:
                audio = instrument_audio
            else:
                audio += instrument_audio
        if np.min(audio) < -1.0 or np.max(audio) > 1.0:
            print("WARNING: Mixing tracks together caused the result to have sample values outside of [-1,1]. Clipping those values")
            audio = np.minimum(np.maximum(audio, -1.0), 1.0)

        librosa.output.write_wav(save_path, audio, sr)
    return save_path

def getRemainingSpectrum(mix_audio, instrument_audio_list, expected_sr, fftWindowSize=1024, hopSize=512, buffer=True):
    '''
    Takes a mixture audio file path and a list of instrument audio file paths, and computes the spectrogram belonging
    to the residual signal consisting of all remaining instruments in the mixture that are not in the given list.
    :param mix_audio: 
    :param instrument_audio_list: 
    :param fftWindowSize: 
    :param hopSize: 
    :param buffer: 
    :return: 
    '''
    assert(isinstance(instrument_audio_list, list))
    if buffer and os.path.exists(mix_audio + "minus.npy"):  # Check if numpy spectrogram file exists
        try:
            [magnitude, phase] = np.load(mix_audio + ".npy")
            assert(np.max(magnitude) < fftWindowSize)
            return magnitude, phase
        except Exception as e:
            print("Could not load " + mix_audio)

    audio, sampleRate = librosa.load(mix_audio, sr=expected_sr)
    for instrument in instrument_audio_list:
        instrument_audio, _ = librosa.load(instrument, sr=expected_sr)
        audio -= instrument_audio
    audio = np.maximum(np.minimum(audio, 1.0), -1.0)
    mag, ph = audioFileToSpectrogram(audio, fftWindowSize=fftWindowSize, hopSize=hopSize, buffer=False)
    if buffer:
        np.save(mix_audio + "minus.npy", [mag, ph])

    assert(np.max(mag) < fftWindowSize)
    return mag, ph

def apply_noise(magnitude):
    return magnitude + tf.random_normal(shape=magnitude.get_shape(), mean=1.0, stddev=0.1, dtype=tf.float32)

def norm(magnitude):
    '''
    Log(1 + magnitude)
    :param magnitude: 
    :return: 
    '''
    return tf.log1p(magnitude)

def norm_with_noise(magnitude):
    '''
    Log(1 + magnitude) + Noise
    :param magnitude: 
    :return: 
    '''
    return tf.log1p(magnitude) + tf.random_uniform(magnitude.shape, minval=1e-7, maxval=1e-5)

def inference_noise(magnitude_batch, gaussian_var):
    # Apply isometric Gaussian noise with variance gaussian_var to every sample in batch
    mu = np.zeros((magnitude_batch.get_shape().as_list()[0], np.prod(magnitude_batch.get_shape().as_list()[1:])))
    stddev = np.full((magnitude_batch.get_shape().as_list()[0], np.prod(magnitude_batch.get_shape().as_list()[1:])), fill_value=gaussian_var)
    pdf = tf.contrib.distributions.MultivariateNormalDiag(mu, stddev)
    noise = tf.reshape(tf.cast(pdf.sample(), tf.float32), magnitude_batch.get_shape())

    # Add to prediction, cut off negative values
    res = magnitude_batch + noise
    return tf.maximum(0.0, res)

def boxcox(magnitude):
    # Box-cox with lambda_1 = 0.133, lambda_2 = 1e-7
    lambda1 = 0.133
    lambda2 = 1e-7
    norm_magnitude = (tf.pow(magnitude + lambda2, lambda1) - 1.0) / lambda1
    return norm_magnitude + tf.random_uniform(magnitude.shape, minval=1e-7, maxval=1e-4)

def norm_range(magnitude, fftWindowSize=1024.):
    '''
    Normalise magnitudes to [0, 1] range
    :param magnitude: 
    :param fftWindowSize: 
    :return: 
    '''
    norm_magnitude = tf.log(magnitude + 1e-7)
    norm_magnitude = (norm_magnitude - tf.log(1e-7)) / (tf.log(fftWindowSize/2.0 + 1e-7) - tf.log(1e-7)) # Norm to [0,1] from [log(1e-7), log(fftWindowsize/2 + 1e-7)
    return norm_magnitude


def denorm_range(norm_magnitude, fftWindowSize=1024.):
    '''
    Denormalise magnitudes from [0, 1] range
    :param magnitude: 
    :param fftWindowSize: 
    :return: 
    '''
    magnitude = norm_magnitude * (tf.log(fftWindowSize/2.0 + 1e-7) - tf.log(1e-7))
    magnitude += tf.log(1e-7)
    magnitude = tf.exp(magnitude)
    return magnitude

def denorm(logmagnitude):
    return tf.expm1(logmagnitude)

def batchToAudiofiles(model_config, input_batch):
    for examples in range(0, input_batch.shape[0]):
        curr = input_batch[examples, :-1, :, 0]
        # curr[64:, :] = 0.0
        curr_audio = spectrogramToAudioFile(curr, fftWindowSize=model_config["num_fft"],hopSize=model_config["num_hop"], phaseIterations=10)
        librosa.output.write_wav(
            "out/fake_" + "_" + str(examples) + ".wav", curr_audio, model_config["expected_sr"])


def spectrogramToAudioFile(magnitude, fftWindowSize, hopSize, phaseIterations=10, phase=None, length=None):
    '''
    Computes an audio signal from the given magnitude spectrogram, and optionally an initial phase.
    Griffin-Lim is executed to recover/refine the given the phase from the magnitude spectrogram.
    :param magnitude: Magnitudes to be converted to audio
    :param fftWindowSize: Size of FFT window used to create magnitudes
    :param hopSize: Hop size in frames used to create magnitudes
    :param phaseIterations: Number of Griffin-Lim iterations to recover phase
    :param phase: If given, starts ISTFT with this particular phase matrix
    :param length: If given, audio signal is clipped/padded to this number of frames
    :return: 
    '''
    if phase is not None:
        if phaseIterations > 0:
            # Refine audio given initial phase with a number of iterations
            return reconPhase(magnitude, fftWindowSize, hopSize, phaseIterations, phase, length)
        # reconstructing the new complex matrix
        stftMatrix = magnitude * np.exp(phase * 1j) # magnitude * e^(j*phase)
        audio = librosa.istft(stftMatrix, hop_length=hopSize, length=length)
    else:
        audio = reconPhase(magnitude, fftWindowSize, hopSize, phaseIterations)
    return audio

def reconPhase(magnitude, fftWindowSize, hopSize, phaseIterations=10, initPhase=None, length=None):
    '''
    Griffin-Lim algorithm for reconstructing the phase for a given magnitude spectrogram, optionally with a given
    intial phase.
    :param magnitude: Magnitudes to be converted to audio
    :param fftWindowSize: Size of FFT window used to create magnitudes
    :param hopSize: Hop size in frames used to create magnitudes
    :param phaseIterations: Number of Griffin-Lim iterations to recover phase
    :param initPhase: If given, starts reconstruction with this particular phase matrix
    :param length: If given, audio signal is clipped/padded to this number of frames
    :return: 
    '''
    for i in range(phaseIterations):
        if i == 0:
            if initPhase is None:
                reconstruction = np.random.random_sample(magnitude.shape) + 1j * (2 * np.pi * np.random.random_sample(magnitude.shape) - np.pi)
            else:
                reconstruction = np.exp(initPhase * 1j) # e^(j*phase), so that angle => phase
        else:
            reconstruction = librosa.stft(audio, fftWindowSize, hopSize)
        spectrum = magnitude * np.exp(1j * np.angle(reconstruction))
        if i == phaseIterations - 1:
            audio = librosa.istft(spectrum, hopSize, length=length)
        else:
            audio = librosa.istft(spectrum, hopSize)
    return audio

def saveSpectrogramToImage(spectrogram, filePath):
    image = np.clip((spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram)), 0, 1)
    # Ignore Low-contrast image warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        io.imsave(filePath, image)