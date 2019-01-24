import tensorflow as tf


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