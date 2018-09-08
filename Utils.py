import tensorflow as tf
import numpy as np
import librosa
from scipy.signal import resample_poly
from fractions import gcd

def resample(audio, orig_sr, new_sr):
    orig_dtype = audio.dtype
    factor = gcd(orig_sr, new_sr)
    down = orig_sr / factor
    up = new_sr / factor
    audio = resample_poly(audio, up, down).astype(orig_dtype)
    return audio

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
    x1 = crop(x1,x2.get_shape().as_list(), match_feature_dim)
    return tf.concat([x1, x2], axis=2)

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

def learned_interpolation_layer(input, padding, level):
    '''
    Implements a trainable upsampling layer by interpolation by a factor of two, from N samples to N*2 - 1.
    Interpolation of intermediate feature vectors v_1 and v_2 (of dimensionality F) is performed by
     w \cdot v_1 + (1-w) \cdot v_2, where \cdot is point-wise multiplication, and w an F-dimensional weight vector constrained to [0,1]
    :param input: Input features of shape [batch_size, 1, width, F]
    :param padding: 
    :param level: 
    :return: 
    '''
    assert(padding == "valid" or padding == "same")
    features = input.get_shape().as_list()[3]

    # Construct 2FxF weight matrix, where F is the number of feature channels in the feature map.
    # Matrix is constrained, made up out of two diagonal FxF matrices with diagonal weights w and 1-w. w is constrained to be in [0,1] # mioid
    weights = tf.get_variable("interp_" + str(level), shape=[features], dtype=tf.float32)
    weights_scaled = tf.nn.sigmoid(weights) # Constrain weights to [0,1]
    counter_weights = 1.0 - weights_scaled # Mirrored weights for the features from the other time step
    conv_weights = tf.expand_dims(tf.concat([tf.expand_dims(tf.diag(weights_scaled), axis=0), tf.expand_dims(tf.diag(counter_weights), axis=0)], axis=0), axis=0)
    intermediate_vals = tf.nn.conv2d(input, conv_weights, strides=[1,1,1,1], padding=padding.upper())
    #intermediate_vals = tf.layers.conv2d(input, features, [1,2], padding=padding)

    intermediate_vals = tf.transpose(intermediate_vals, [2, 0, 1, 3])
    out = tf.transpose(input, [2, 0, 1, 3])
    num_entries = out.get_shape().as_list()[0]
    out = tf.concat([out, intermediate_vals], axis=0)
    indices = list()

    # Interleave interpolated features with original ones, starting with the first original one
    num_outputs = (2*num_entries - 1) if padding == "valid" else 2*num_entries
    for idx in range(num_outputs):
        if idx % 2 == 0:
            indices.append(idx // 2)
        else:
            indices.append(num_entries + idx//2)
    out = tf.gather(out, indices)
    current_layer = tf.transpose(out, [1, 2, 0, 3])
    return current_layer

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def load(path, sr=22050, mono=True, offset=0.0, duration=None, dtype=np.float32):
    # ALWAYS output (n_frames, n_channels) audio
    y, orig_sr = librosa.load(path, None, mono, offset, duration, dtype)
    if len(y.shape) == 1:
        y = np.expand_dims(y, axis=0)
    y = y.T
    if sr != None:
        return resample(y, orig_sr, sr), sr
    else:
        return y, orig_sr

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