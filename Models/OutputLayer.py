import tensorflow as tf

def independent_outputs(featuremap, num_sources, num_channels):
    outputs = list()
    for _ in range(num_sources):
        outputs.append(tf.layers.conv1d(featuremap, num_channels, 1, activation=tf.tanh, padding='valid'))
    return outputs

def difference_output(input_mix, featuremap, num_sources, num_channels):
    outputs = list()
    last_source = input_mix
    for _ in range(num_sources-1):
        out = tf.layers.conv1d(featuremap, num_channels, 1, activation=tf.tanh, padding='valid')
        outputs.append(out)
        last_source = last_source - out
    outputs.append(last_source)
    return outputs