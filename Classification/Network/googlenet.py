import tensorflow as tf
import numpy as np
import sys

# input_dim : [224, 224, 3]
class GoogLeNet(object):
    def __init__(self, x, keep_prob, num_classes, skip_layer,
                 weights_path, name):

        # Parse input arguments into class variables
        self.X = x
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.SKIP_LAYER = skip_layer
        self.WEIGHTS_PATH = weights_path
        self.name = name

        # Call the create function to build the computational graph
        self.create()

    def create(self):

        with tf.variable_scope(self.name, 'GoogLeNet', [self.X, self.NUM_CLASSES]):
            conv1_7x7_s2 = conv(self.X, 7, 7, 64, 2, 2, name='conv1_7x7_s2')
            pool1_3x3_s2 = max_pool(conv1_7x7_s2, 3, 3, 2, 2, name='pool1_3x3_s2')
            pool1_norm1 = lrn(pool1_3x3_s2, 2, 1.99999994948e-05, 0.75, name='pool1_norm1')
            conv2_3x3_reduce = conv(pool1_norm1, 1, 1, 64, 1, 1, name='conv2_3x3_reduce')
            conv2_3x3 = conv(conv2_3x3_reduce, 3, 3, 192, 1, 1, name='conv2_3x3')
            conv2_norm2 = lrn(conv2_3x3, 2, 1.99999994948e-05, 0.75, name='conv2_norm2')
            pool2_3x3_s2 = max_pool(conv2_norm2, 3, 3, 2, 2, name='pool2_3x3_s2')
            inception_3a_1x1 = conv(pool2_3x3_s2, 1, 1, 64, 1, 1, name='inception_3a_1x1')

            inception_3a_3x3_reduce = conv(pool2_3x3_s2, 1, 1, 96, 1, 1, name='inception_3a_3x3_reduce')
            inception_3a_3x3 = conv(inception_3a_3x3_reduce, 3, 3, 128, 1, 1, name='inception_3a_3x3')

            inception_3a_5x5_reduce = conv(pool2_3x3_s2, 1, 1, 16, 1, 1, name='inception_3a_5x5_reduce')
            inception_3a_5x5 = conv(inception_3a_5x5_reduce, 5, 5, 32, 1, 1, name='inception_3a_5x5')

            inception_3a_pool = max_pool(pool2_3x3_s2, 3, 3, 1, 1, name='inception_3a_pool')
            inception_3a_pool_proj = conv(inception_3a_pool, 1, 1, 32, 1, 1, name='inception_3a_pool_proj')

            inception_3a_output = concat([inception_3a_1x1, inception_3a_3x3, inception_3a_5x5, inception_3a_pool_proj], 3,
                                         name='inception_3a_output')
            inception_3b_1x1 = conv(inception_3a_output, 1, 1, 128, 1, 1, name='inception_3b_1x1')

            inception_3b_3x3_reduce = conv(inception_3a_output, 1, 1, 128, 1, 1, name='inception_3b_3x3_reduce')
            inception_3b_3x3 = conv(inception_3b_3x3_reduce, 3, 3, 192, 1, 1, name='inception_3b_3x3')

            inception_3b_5x5_reduce = conv(inception_3a_output, 1, 1, 32, 1, 1, name='inception_3b_5x5_reduce')
            inception_3b_5x5 = conv(inception_3b_5x5_reduce, 5, 5, 96, 1, 1, name='inception_3b_5x5')

            inception_3b_pool = max_pool(inception_3a_output, 3, 3, 1, 1, name='inception_3b_pool')
            inception_3b_pool_proj = conv(inception_3b_pool, 1, 1, 64, 1, 1, name='inception_3b_pool_proj')

            inception_3b_output = concat([inception_3b_1x1, inception_3b_3x3, inception_3b_5x5, inception_3b_pool_proj], 3,
                                         name='inception_3b_output')
            pool3_3x3_s2 = max_pool(inception_3b_output, 3, 3, 2, 2, name='pool3_3x3_s2')
            inception_4a_1x1 = conv(pool3_3x3_s2, 1, 1, 192, 1, 1, name='inception_4a_1x1')

            inception_4a_3x3_reduce = conv(pool3_3x3_s2, 1, 1, 96, 1, 1, name='inception_4a_3x3_reduce')
            inception_4a_3x3 = conv(inception_4a_3x3_reduce, 3, 3, 208, 1, 1, name='inception_4a_3x3')

            inception_4a_5x5_reduce = conv(pool3_3x3_s2, 1, 1, 16, 1, 1, name='inception_4a_5x5_reduce')
            inception_4a_5x5 = conv(inception_4a_5x5_reduce, 5, 5, 48, 1, 1, name='inception_4a_5x5')

            inception_4a_pool = max_pool(pool3_3x3_s2, 3, 3, 1, 1, name='inception_4a_pool')
            inception_4a_pool_proj = conv(inception_4a_pool, 1, 1, 64, 1, 1, name='inception_4a_pool_proj')

            inception_4a_output = concat([inception_4a_1x1, inception_4a_3x3, inception_4a_5x5, inception_4a_pool_proj], 3,
                                         name='inception_4a_output')
            inception_4b_1x1 = conv(inception_4a_output, 1, 1, 160, 1, 1, name='inception_4b_1x1')

            inception_4b_3x3_reduce = conv(inception_4a_output, 1, 1, 112, 1, 1, name='inception_4b_3x3_reduce')
            inception_4b_3x3 = conv(inception_4b_3x3_reduce, 3, 3, 224, 1, 1, name='inception_4b_3x3')

            inception_4b_5x5_reduce = conv(inception_4a_output, 1, 1, 24, 1, 1, name='inception_4b_5x5_reduce')
            inception_4b_5x5 = conv(inception_4b_5x5_reduce, 5, 5, 64, 1, 1, name='inception_4b_5x5')

            inception_4b_pool = max_pool(inception_4a_output, 3, 3, 1, 1, name='inception_4b_pool')
            inception_4b_pool_proj = conv(inception_4b_pool, 1, 1, 64, 1, 1, name='inception_4b_pool_proj')

            inception_4b_output = concat([inception_4b_1x1, inception_4b_3x3, inception_4b_5x5, inception_4b_pool_proj], 3,
                                         name='inception_4b_output')
            inception_4c_1x1 = conv(inception_4b_output, 1, 1, 128, 1, 1, name='inception_4c_1x1')

            inception_4c_3x3_reduce = conv(inception_4b_output, 1, 1, 128, 1, 1, name='inception_4c_3x3_reduce')
            inception_4c_3x3 = conv(inception_4c_3x3_reduce, 3, 3, 256, 1, 1, name='inception_4c_3x3')

            inception_4c_5x5_reduce = conv(inception_4b_output, 1, 1, 24, 1, 1, name='inception_4c_5x5_reduce')
            inception_4c_5x5 = conv(inception_4c_5x5_reduce, 5, 5, 64, 1, 1, name='inception_4c_5x5')

            inception_4c_pool = max_pool(inception_4b_output, 3, 3, 1, 1, name='inception_4c_pool')
            inception_4c_pool_proj = conv(inception_4c_pool, 1, 1, 64, 1, 1, name='inception_4c_pool_proj')

            inception_4c_output = concat([inception_4c_1x1, inception_4c_3x3, inception_4c_5x5, inception_4c_pool_proj], 3,
                                         name='inception_4c_output')
            inception_4d_1x1 = conv(inception_4c_output, 1, 1, 112, 1, 1, name='inception_4d_1x1')

            inception_4d_3x3_reduce = conv(inception_4c_output, 1, 1, 144, 1, 1, name='inception_4d_3x3_reduce')
            inception_4d_3x3 = conv(inception_4d_3x3_reduce, 3, 3, 288, 1, 1, name='inception_4d_3x3')

            inception_4d_5x5_reduce = conv(inception_4c_output, 1, 1, 32, 1, 1, name='inception_4d_5x5_reduce')
            inception_4d_5x5 = conv(inception_4d_5x5_reduce, 5, 5, 64, 1, 1, name='inception_4d_5x5')

            inception_4d_pool = max_pool(inception_4c_output, 3, 3, 1, 1, name='inception_4d_pool')
            inception_4d_pool_proj = conv(inception_4d_pool, 1, 1, 64, 1, 1, name='inception_4d_pool_proj')

            inception_4d_output = concat([inception_4d_1x1, inception_4d_3x3, inception_4d_5x5, inception_4d_pool_proj], 3,
                                         name='inception_4d_output')
            inception_4e_1x1 = conv(inception_4d_output, 1, 1, 256, 1, 1, name='inception_4e_1x1')

            inception_4e_3x3_reduce = conv(inception_4d_output, 1, 1, 160, 1, 1, name='inception_4e_3x3_reduce')
            inception_4e_3x3 = conv(inception_4e_3x3_reduce, 3, 3, 320, 1, 1, name='inception_4e_3x3')

            inception_4e_5x5_reduce = conv(inception_4d_output, 1, 1, 32, 1, 1, name='inception_4e_5x5_reduce')
            inception_4e_5x5 = conv(inception_4e_5x5_reduce, 5, 5, 128, 1, 1, name='inception_4e_5x5')

            inception_4e_pool = max_pool(inception_4d_output, 3, 3, 1, 1, name='inception_4e_pool')
            inception_4e_pool_proj = conv(inception_4e_pool, 1, 1, 128, 1, 1, name='inception_4e_pool_proj')

            inception_4e_output = concat([inception_4e_1x1, inception_4e_3x3, inception_4e_5x5, inception_4e_pool_proj], 3,
                                         name='inception_4e_output')
            pool4_3x3_s2 = max_pool(inception_4e_output, 3, 3, 2, 2, name='pool4_3x3_s2')
            inception_5a_1x1 = conv(pool4_3x3_s2, 1, 1, 256, 1, 1, name='inception_5a_1x1')

            inception_5a_3x3_reduce = conv(pool4_3x3_s2, 1, 1, 160, 1, 1, name='inception_5a_3x3_reduce')
            inception_5a_3x3 = conv(inception_5a_3x3_reduce, 3, 3, 320, 1, 1, name='inception_5a_3x3')

            inception_5a_5x5_reduce = conv(pool4_3x3_s2, 1, 1, 32, 1, 1, name='inception_5a_5x5_reduce')
            inception_5a_5x5 = conv(inception_5a_5x5_reduce, 5, 5, 128, 1, 1, name='inception_5a_5x5')

            inception_5a_pool = max_pool(pool4_3x3_s2, 3, 3, 1, 1, name='inception_5a_pool')
            inception_5a_pool_proj = conv(inception_5a_pool, 1, 1, 128, 1, 1, name='inception_5a_pool_proj')

            inception_5a_output = concat([inception_5a_1x1, inception_5a_3x3, inception_5a_5x5, inception_5a_pool_proj], 3,
                                         name='inception_5a_output')
            inception_5b_1x1 = conv(inception_5a_output, 1, 1, 384, 1, 1, name='inception_5b_1x1')

            inception_5b_3x3_reduce = conv(inception_5a_output, 1, 1, 192, 1, 1, name='inception_5b_3x3_reduce')
            inception_5b_3x3 = conv(inception_5b_3x3_reduce, 3, 3, 384, 1, 1, name='inception_5b_3x3')

            inception_5b_5x5_reduce = conv(inception_5a_output, 1, 1, 48, 1, 1, name='inception_5b_5x5_reduce')
            inception_5b_5x5 = conv(inception_5b_5x5_reduce, 5, 5, 128, 1, 1, name='inception_5b_5x5')

            inception_5b_pool = max_pool(inception_5a_output, 3, 3, 1, 1, name='inception_5b_pool')
            inception_5b_pool_proj = conv(inception_5b_pool, 1, 1, 128, 1, 1, name='inception_5b_pool_proj')

            inception_5b_output = concat([inception_5b_1x1, inception_5b_3x3, inception_5b_5x5, inception_5b_pool_proj], 3,
                                         name='inception_5b_output')
            self.pool5_7x7_s1 = avg_pool(inception_5b_output, 7, 7, 1, 1, padding='VALID', name='pool5_7x7_s1')
            pool5_drop_7x7_s1 = dropout(self.pool5_7x7_s1, self.KEEP_PROB, name='pool5_drop_7x7_s1')
            self.fc_final = fc(pool5_drop_7x7_s1, self.NUM_CLASSES, relu=False, name='loss3_classifier')

    def load_initial_weights(self, session):
        print('Start to load pre-trained model : ')
        # Load the weights into memory
        weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()

        # Setting Progress bar
        progress_t = len(weights_dict)
        progress_i = 0

        # Loop over all layer names stored in the weights dict
        for layer_name in weights_dict:
            # Start Progress bar
            printProgressBar(progress_i, progress_t, prefix='Load Model:', suffix='Complete', length=50)

            # Check if the layer is one of the layers that should be reinitialized
            if layer_name not in self.SKIP_LAYER:

                with tf.variable_scope(self.name + '/' + layer_name, reuse=True):
                    #print('layer name:   ', layer_name)

                    # Loop over list of weights/biases and assign them to their corresponding tf variable
                    for op in weights_dict[layer_name]:
                        data = weights_dict[layer_name][op]
                        # Biases
                        if op == b'biases':
                            #print('bias dim:     ', data.shape)
                            var = tf.get_variable('biases', trainable=False)
                            session.run(var.assign(data))

                        # Weights
                        elif op == b'weights':
                            #print('weights dim:  ', data.shape)
                            var = tf.get_variable('weights', trainable=False)
                            session.run(var.assign(data))

                        # Mean
                        elif op == b'mean':
                            #print('mean dim:     ', data.shape)
                            var = tf.get_variable('mean', trainable=False)
                            session.run(var.assign(data))

                        # Variance
                        elif op == b'variance':
                            #print('variance dim: ', data.shape)
                            var = tf.get_variable('variance', trainable=False)
                            session.run(var.assign(data))

                        # Scale
                        elif op == b'scale':
                            #print('scale dim:    ', data.shape)
                            var = tf.get_variable('scale', trainable=False)
                            session.run(var.assign(data))

                        # Offset
                        else:
                            #print('offset dim:   ', data.shape)
                            var = tf.get_variable('offset', trainable=False)
                            session.run(var.assign(data))
            progress_i += 1

        printProgressBar(progress_i, progress_t, prefix='Load Model:', suffix='Complete', length=50)

"""
Adapted from: https://github.com/ethereon/caffe-tensorflow
"""
def conv(input, filter_height, filter_width, num_filters, stride_y, stride_x, name, relu=True,
         padding='SAME', group=1, biased=True):

    # Get number of input channels
    input_channels = int(input.get_shape()[-1])

    # Verify that the grouping parameter is valid
    assert input_channels % group == 0
    assert num_filters % group == 0

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels / group, num_filters])

        if group == 1:
            # This is the common-case. Convolve the input without any further complications.
            output = convolve(input, weights)

        # In the cases of multiple groups, split inputs & weights and
        else:
            # Split the input into groups and then convolve each of them independently
            input_groups = tf.split(axis=3, num_or_size_splits=group, value=input)
            weight_groups = tf.split(axis=3, num_or_size_splits=group, value=weights)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

            # Concat the convolved output together again
            output = tf.concat(axis=3, values=output_groups)

        # Add biases
        if biased:
            #biases = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
            biases = tf.get_variable('biases', shape=[num_filters])
            output = tf.nn.bias_add(output, biases)

        if relu:
            # Apply relu function
            output = tf.nn.relu(output, name=scope.name)

        return output

def relu(input, name):
    return tf.nn.relu(input, name=name)

def fc(input, num_out, name, relu=True):
    with tf.variable_scope(name) as scope:
        input_shape = input.get_shape()
        if input_shape.ndims == 4:
            # The input is spatial. Vectorize it first.
            dim = 1
            for d in input_shape[1:].as_list():
                dim *= d
            feed_in = tf.reshape(input, [-1, dim])
        else:
            feed_in, dim = (input, input_shape[-1].value)

        weights = tf.get_variable('weights', shape=[dim, num_out], trainable=True)
        biases = tf.get_variable('biases', [num_out], trainable=True)
        op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
        fc = op(feed_in, weights, biases, name=scope.name)
        return fc

def max_pool(input, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
    return tf.nn.max_pool(input,
                          ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding,
                          name=name)

def avg_pool(input, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
    return tf.nn.avg_pool(input,
                          ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding,
                          name=name)

def lrn(input, radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(input,
                                              depth_radius=radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias,
                                              name=name)


def concat(inputs, axis, name):
    return tf.concat(values=inputs, axis=axis, name=name)

def add(inputs, name):
    return tf.add_n(inputs, name=name)

def batch_normalization(input, name, scale_offset=True, relu=False):
    # NOTE: Currently, only inference is supported
    with tf.variable_scope(name) as scope:
        shape = [input.get_shape()[-1]]
        if scale_offset:
            scale = tf.get_variable('scale', shape=shape, trainable=True)
            offset = tf.get_variable('offset', shape=shape, trainable=True)
        else:
            scale, offset = (None, None)
        output = tf.nn.batch_normalization(
            input,
            mean=tf.get_variable('mean', shape=shape, trainable=True),
            variance=tf.get_variable('variance', shape=shape, trainable=True),
            offset=offset,
            scale=scale,
            # TODO: This is the default Caffe batch norm eps
            # Get the actual eps from parameters
            variance_epsilon=1e-5,
            name=name)
        if relu:
            output = tf.nn.relu(output)
        return output

def dropout(input, keep_prob, name):
    return tf.nn.dropout(input, keep_prob, name=name)

# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    # print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    sys.stdout.write("\r%s |%s| %s%% %s" % (prefix, bar, percent, suffix))
    sys.stdout.flush()

    # Print New Line on Complete
    if iteration == total:
        print()