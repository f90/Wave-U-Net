import tensorflow as tf

import Utils

def independent_outputs(featuremap, num_sources, num_channels, filter_width, padding, activation):
    outputs = list()
    for i in range(num_sources):
        outputs.append(tf.layers.conv1d(featuremap, num_channels, filter_width, activation=activation, padding=padding))
    return outputs

def difference_output(input_mix, featuremap, num_sources, num_channels, filter_width, padding, activation, training):
    outputs = list()
    sum_source = 0
    for _ in range(num_sources-1):
        out = tf.layers.conv1d(featuremap, num_channels, filter_width, activation=activation, padding=padding)
        outputs.append(out)
        sum_source = sum_source + out

    # Compute last source based on the others
    last_source = Utils.crop(input_mix, sum_source.get_shape().as_list()) - sum_source
    last_source = Utils.AudioClip(last_source, training)
    outputs.append(last_source)
    return outputs