import tensorflow as tf
import numpy as np
import librosa

def getTrainableVariables(tag=""):
    return [v for v in tf.trainable_variables() if tag in v.name]

def getNumParams(tensors):
    return np.sum([np.prod(t.get_shape().as_list()) for t in tensors])

def crop_and_concat(x1,x2, match_feature_dim=True):
    '''
    Copy-and-crop operation for two feature maps of different size.
    Crops the first input x1 equally along its borders so that its shape is equal to 
    the shape of the second input x2, then concatenates them along the feature channel axis.
    :param x1: First input that is cropped and combined with the second input
    :param x2: Second input
    :return: Combined feature map
    '''
    if x2 is None:
        return x1

    x1 = crop(x1,x2.get_shape().as_list(), match_feature_dim)
    return tf.concat([x1, x2], axis=2)

def random_amplify(sample):
    '''
    Randomly amplifies or attenuates the input signal
    :return: Amplified signal
    '''
    for key, val in list(sample.items()):
        if key != "mix":
            sample[key] = tf.random_uniform([], 0.7, 1.0) * val

    sample["mix"] = tf.add_n([val for key, val in list(sample.items()) if key != "mix"])
    return sample

def crop_sample(sample, crop_frames):
    for key, val in list(sample.items()):
        if key != "mix" and crop_frames > 0:
            sample[key] = val[crop_frames:-crop_frames,:]
    return sample

def pad_freqs(tensor, target_shape):
    '''
    Pads the frequency axis of a 4D tensor of shape [batch_size, freqs, timeframes, channels] or 2D tensor [freqs, timeframes] with zeros
    so that it reaches the target shape. If the number of frequencies to pad is uneven, the rows are appended at the end. 
    :param tensor: Input tensor to pad with zeros along the frequency axis
    :param target_shape: Shape of tensor after zero-padding
    :return: Padded tensor
    '''
    target_freqs = (target_shape[1] if len(target_shape) == 4 else target_shape[0]) #TODO
    if isinstance(tensor, tf.Tensor):
        input_shape = tensor.get_shape().as_list()
    else:
        input_shape = tensor.shape

    if len(input_shape) == 2:
        input_freqs = input_shape[0]
    else:
        input_freqs = input_shape[1]

    diff = target_freqs - input_freqs
    if diff % 2 == 0:
        pad = [(diff/2, diff/2)]
    else:
        pad = [(diff//2, diff//2 + 1)] # Add extra frequency bin at the end

    if len(target_shape) == 2:
        pad = pad + [(0,0)]
    else:
        pad = [(0,0)] + pad + [(0,0), (0,0)]

    if isinstance(tensor, tf.Tensor):
        return tf.pad(tensor, pad, mode='constant', constant_values=0.0)
    else:
        return np.pad(tensor, pad, mode='constant', constant_values=0.0)

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def AudioClip(x, training):
    '''
    Simply returns the input if training is set to True, otherwise clips the input to [-1,1]
    :param x: Input tensor (coming from last layer of neural network)
    :param training: Whether model is in training (True) or testing mode (False)
    :return: Output tensor (potentially clipped)
    '''
    if training:
        return x
    else:
        return tf.maximum(tf.minimum(x, 1.0), -1.0)

def resample(audio, orig_sr, new_sr):
    return librosa.resample(audio.T, orig_sr, new_sr).T

def load(path, sr=22050, mono=True, offset=0.0, duration=None, dtype=np.float32):
    # ALWAYS output (n_frames, n_channels) audio
    y, orig_sr = librosa.load(path, sr, mono, offset, duration, dtype)
    if len(y.shape) == 1:
        y = np.expand_dims(y, axis=0)
    return y.T, orig_sr

def crop(tensor, target_shape, match_feature_dim=True):
    '''
    Crops a 3D tensor [batch_size, width, channels] along the width axes to a target shape.
    Performs a centre crop. If the dimension difference is uneven, crop last dimensions first.
    :param tensor: 4D tensor [batch_size, width, height, channels] that should be cropped. 
    :param target_shape: Target shape (4D tensor) that the tensor should be cropped to
    :return: Cropped tensor
    '''
    shape = np.array(tensor.get_shape().as_list())
    diff = shape - np.array(target_shape)
    assert(diff[0] == 0 and (diff[2] == 0 or not match_feature_dim))# Only width axis can differ
    if (diff[1] % 2 != 0):
        print("WARNING: Cropping with uneven number of extra entries on one side")
    assert diff[1] >= 0 # Only positive difference allowed
    if diff[1] == 0:
        return tensor
    crop_start = diff // 2
    crop_end = diff - crop_start

    return tensor[:,crop_start[1]:-crop_end[1],:]

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