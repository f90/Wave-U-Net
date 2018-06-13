import tensorflow as tf

import Utils
from Utils import LeakyReLU
import numpy as np
import OutputLayer

class UnetAudioSeparator:
    '''
    U-Net separator network for singing voice separation.
    Uses valid convolutions, so it predicts for the centre part of the input - only certain input and output shapes are therefore possible (see getpadding function)
    '''

    def __init__(self, num_layers, num_initial_filters, upsampling, output_type, context, num_sources, mono, filter_size, merge_filter_size):
        '''
        Initialize U-net
        :param num_layers: Number of down- and upscaling layers in the network 
        '''
        self.num_layers = num_layers
        self.num_initial_filters = num_initial_filters
        self.filter_size = filter_size
        self.merge_filter_size = merge_filter_size
        self.upsampling = upsampling
        self.output_type = output_type
        self.context = context
        self.padding = "valid" if context else "same"
        self.num_sources = num_sources
        self.num_channels = 1 if mono else 2

    def get_padding(self, shape):
        '''
        Calculates the required amounts of padding along each axis of the input and output, so that the Unet works and has the given shape as output shape
        :param shape: Desired output shape 
        :return: Input_shape, output_shape, where each is a list [batch_size, time_steps, channels]
        '''

        if self.context:
            # Check if desired shape is possible as output shape - go from output shape towards lowest-res feature map
            rem = float(shape[1]) # Cut off batch size number and channel
            #rem = rem +  self.filter_size - 1
            for i in range(self.num_layers):
                rem = rem + self.merge_filter_size - 1
                rem = (rem + 1.) / 2.# out = in + in - 1 <=> in = (out+1)/

            # Round resulting feature map dimensions up to nearest integer
            x = np.asarray(np.ceil(rem),dtype=np.int64)
            assert(x >= 2)

            # Compute input and output shapes based on lowest-res feature map
            output_shape = x
            input_shape = x

            # Extra conv
            input_shape = input_shape + self.filter_size - 1

            # Go from centre feature map through up- and downsampling blocks
            for i in range(self.num_layers):
                output_shape = 2*output_shape - 1 #Upsampling
                output_shape = output_shape - self.merge_filter_size + 1 # Conv

                input_shape = 2*input_shape - 1 # Decimation
                input_shape = input_shape + self.filter_size - 1 # Conv


            input_shape = np.concatenate([[shape[0]], [input_shape], [self.num_channels]])
            output_shape = np.concatenate([[shape[0]], [output_shape], [self.num_channels]])

            return input_shape, output_shape
        else:
            return [shape[0], shape[1], self.num_channels], [shape[0], shape[1], self.num_channels]

    def get_output(self, input, training=None, return_spectrogram=False, reuse=True):
        '''
        Creates symbolic computation graph of the U-Net for a given input batch
        :param input: Input batch of mixtures, 3D tensor [batch_size, num_samples, num_channels]
        :param reuse: Whether to create new parameter variables or reuse existing ones
        :return: U-Net output: List of source estimates. Each item is a 3D tensor [batch_size, num_out_samples, num_channels]
        '''
        with tf.variable_scope("separator", reuse=reuse):
            enc_outputs = list()
            current_layer = input

            # Down-convolution: Repeat strided conv
            for i in range(self.num_layers):
                current_layer = tf.layers.conv1d(current_layer, self.num_initial_filters + (self.num_initial_filters * i), self.filter_size, strides=1, activation=LeakyReLU, padding=self.padding) # out = in - filter + 1
                enc_outputs.append(current_layer)
                current_layer = current_layer[:,::2,:] # Decimate by factor of 2 # out = (in-1)/2 + 1

            current_layer = tf.layers.conv1d(current_layer, self.num_initial_filters + (self.num_initial_filters * self.num_layers),self.filter_size,activation=LeakyReLU,padding=self.padding) # One more conv here since we need to compute features after last decimation

            # Feature map here shall be X along one dimension

            # Upconvolution
            for i in range(self.num_layers):
                #UPSAMPLING
                current_layer = tf.expand_dims(current_layer, axis=1)

                if self.upsampling == 'learned':
                    # Learned interpolation between two neighbouring time positions by using a convolution filter of width 2, and inserting the responses in the middle of the two respective inputs
                    current_layer = Utils.learned_interpolation_layer(current_layer, self.padding, i)
                else:
                    if self.context:
                        current_layer = tf.image.resize_bilinear(current_layer, [1, current_layer.get_shape().as_list()[2] * 2 - 1], align_corners=True)
                    else:
                        current_layer = tf.image.resize_bilinear(current_layer, [1, current_layer.get_shape().as_list()[2]*2]) # out = in + in - 1
                #current_layer = tf.layers.conv2d_transpose(current_layer, self.num_initial_filters + (16 * (self.num_layers-i-1)), [1, 15], strides=[1, 2], activation=LeakyReLU, padding='same') # output = input * stride + filter - stride
                current_layer = tf.squeeze(current_layer, axis=1)

                assert(enc_outputs[-i-1].get_shape().as_list()[1] == current_layer.get_shape().as_list()[1] or self.context) #No cropping should be necessary unless we are using context
                current_layer = Utils.crop_and_concat(enc_outputs[-i-1], current_layer, match_feature_dim=False)
                current_layer = tf.layers.conv1d(current_layer, self.num_initial_filters + (self.num_initial_filters * (self.num_layers - i - 1)), self.merge_filter_size,
                                                 activation=LeakyReLU,
                                                 padding=self.padding)  # out = in - filter + 1

            current_layer = Utils.crop_and_concat(input, current_layer, match_feature_dim=False)

            # Output layer
            if self.output_type == "direct":
                return OutputLayer.independent_outputs(current_layer, self.num_sources, self.num_channels)
            elif self.output_type == "difference":
                cropped_input = Utils.crop(input,current_layer.get_shape().as_list(), match_feature_dim=False)
                return OutputLayer.difference_output(cropped_input, current_layer, self.num_sources, self.num_channels)
            else:
                raise NotImplementedError
