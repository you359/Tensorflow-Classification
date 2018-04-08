import tensorflow as tf
import numpy as np
import sys

# input_dim : [224, 224, 3]
class ResNet50(object):
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

        with tf.variable_scope(self.name, 'ResNet-50', [self.X, self.NUM_CLASSES]):
            conv1 = conv(self.X, 7, 7, 64, 2, 2, relu=False, name='conv1')
            bn_conv1 = batch_normalization(conv1, relu=True, name='bn_conv1')
            pool1 = max_pool(bn_conv1, 3, 3, 2, 2, name='pool1')
            res2a_branch1 = conv(pool1, 1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch1')
            bn2a_branch1 = batch_normalization(res2a_branch1, name='bn2a_branch1')

            res2a_branch2a = conv(pool1, 1, 1, 64, 1, 1, biased=False, relu=False, name='res2a_branch2a')
            bn2a_branch2a = batch_normalization(res2a_branch2a, relu=True, name='bn2a_branch2a')
            res2a_branch2b = conv(bn2a_branch2a, 3, 3, 64, 1, 1, biased=False, relu=False, name='res2a_branch2b')
            bn2a_branch2b = batch_normalization(res2a_branch2b, relu=True, name='bn2a_branch2b')
            res2a_branch2c = conv(bn2a_branch2b, 1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch2c')
            bn2a_branch2c = batch_normalization(res2a_branch2c, name='bn2a_branch2c')

            res2a = add([bn2a_branch1, bn2a_branch2c], name='res2a')
            res2a_relu = relu(res2a, name='res2a_relu')
            res2b_branch2a = conv(res2a_relu, 1, 1, 64, 1, 1, biased=False, relu=False, name='res2b_branch2a')
            bn2b_branch2a = batch_normalization(res2b_branch2a, relu=True, name='bn2b_branch2a')
            res2b_branch2b = conv(bn2b_branch2a, 3, 3, 64, 1, 1, biased=False, relu=False, name='res2b_branch2b')
            bn2b_branch2b = batch_normalization(res2b_branch2b, relu=True, name='bn2b_branch2b')
            res2b_branch2c = conv(bn2b_branch2b, 1, 1, 256, 1, 1, biased=False, relu=False, name='res2b_branch2c')
            bn2b_branch2c = batch_normalization(res2b_branch2c, name='bn2b_branch2c')

            res2b = add([res2a_relu, bn2b_branch2c], name='res2b')
            res2b_relu = relu(res2b, name='res2b_relu')
            res2c_branch2a = conv(res2b_relu, 1, 1, 64, 1, 1, biased=False, relu=False, name='res2c_branch2a')
            bn2c_branch2a = batch_normalization(res2c_branch2a, relu=True, name='bn2c_branch2a')
            res2c_branch2b = conv(bn2c_branch2a, 3, 3, 64, 1, 1, biased=False, relu=False, name='res2c_branch2b')
            bn2c_branch2b = batch_normalization(res2c_branch2b, relu=True, name='bn2c_branch2b')
            res2c_branch2c = conv(bn2c_branch2b, 1, 1, 256, 1, 1, biased=False, relu=False, name='res2c_branch2c')
            bn2c_branch2c = batch_normalization(res2c_branch2c, name='bn2c_branch2c')

            res2c = add([res2b_relu, bn2c_branch2c], name='res2c')
            res2c_relu = relu(res2c, name='res2c_relu')
            res3a_branch1 = conv(res2c_relu, 1, 1, 512, 2, 2, biased=False, relu=False, name='res3a_branch1')
            bn3a_branch1 = batch_normalization(res3a_branch1, name='bn3a_branch1')

            res3a_branch2a = conv(res2c_relu, 1, 1, 128, 2, 2, biased=False, relu=False, name='res3a_branch2a')
            bn3a_branch2a = batch_normalization(res3a_branch2a, relu=True, name='bn3a_branch2a')
            res3a_branch2b = conv(bn3a_branch2a, 3, 3, 128, 1, 1, biased=False, relu=False, name='res3a_branch2b')
            bn3a_branch2b = batch_normalization(res3a_branch2b, relu=True, name='bn3a_branch2b')
            res3a_branch2c = conv(bn3a_branch2b, 1, 1, 512, 1, 1, biased=False, relu=False, name='res3a_branch2c')
            bn3a_branch2c = batch_normalization(res3a_branch2c, name='bn3a_branch2c')

            res3a = add([bn3a_branch1, bn3a_branch2c], name='res3a')
            res3a_relu = relu(res3a, name='res3a_relu')
            res3b_branch2a = conv(res3a_relu, 1, 1, 128, 1, 1, biased=False, relu=False, name='res3b_branch2a')
            bn3b_branch2a = batch_normalization(res3b_branch2a, relu=True, name='bn3b_branch2a')
            res3b_branch2b = conv(bn3b_branch2a, 3, 3, 128, 1, 1, biased=False, relu=False, name='res3b_branch2b')
            bn3b_branch2b = batch_normalization(res3b_branch2b, relu=True, name='bn3b_branch2b')
            res3b_branch2c = conv(bn3b_branch2b, 1, 1, 512, 1, 1, biased=False, relu=False, name='res3b_branch2c')
            bn3b_branch2c = batch_normalization(res3b_branch2c, name='bn3b_branch2c')

            res3b = add([res3a_relu, bn3b_branch2c], name='res3b')
            res3b_relu = relu(res3b, name='res3b_relu')
            res3c_branch2a = conv(res3b_relu, 1, 1, 128, 1, 1, biased=False, relu=False, name='res3c_branch2a')
            bn3c_branch2a = batch_normalization(res3c_branch2a, relu=True, name='bn3c_branch2a')
            res3c_branch2b = conv(bn3c_branch2a, 3, 3, 128, 1, 1, biased=False, relu=False, name='res3c_branch2b')
            bn3c_branch2b = batch_normalization(res3c_branch2b, relu=True, name='bn3c_branch2b')
            res3c_branch2c = conv(bn3c_branch2b, 1, 1, 512, 1, 1, biased=False, relu=False, name='res3c_branch2c')
            bn3c_branch2c = batch_normalization(res3c_branch2c, name='bn3c_branch2c')

            res3c = add([res3b_relu, bn3c_branch2c], name='res3c')
            res3c_relu = relu(res3c, name='res3c_relu')
            res3d_branch2a = conv(res3c_relu, 1, 1, 128, 1, 1, biased=False, relu=False, name='res3d_branch2a')
            bn3d_branch2a = batch_normalization(res3d_branch2a, relu=True, name='bn3d_branch2a')
            res3d_branch2b = conv(bn3d_branch2a, 3, 3, 128, 1, 1, biased=False, relu=False, name='res3d_branch2b')
            bn3d_branch2b = batch_normalization(res3d_branch2b, relu=True, name='bn3d_branch2b')
            res3d_branch2c = conv(bn3d_branch2b, 1, 1, 512, 1, 1, biased=False, relu=False, name='res3d_branch2c')
            bn3d_branch2c = batch_normalization(res3d_branch2c, name='bn3d_branch2c')

            res3d = add([res3c_relu, bn3d_branch2c], name='res3d')
            res3d_relu = relu(res3d, name='res3d_relu')
            res4a_branch1 = conv(res3d_relu, 1, 1, 1024, 2, 2, biased=False, relu=False, name='res4a_branch1')
            bn4a_branch1 = batch_normalization(res4a_branch1, name='bn4a_branch1')

            res4a_branch2a = conv(res3d_relu, 1, 1, 256, 2, 2, biased=False, relu=False, name='res4a_branch2a')
            bn4a_branch2a = batch_normalization(res4a_branch2a, relu=True, name='bn4a_branch2a')
            res4a_branch2b = conv(bn4a_branch2a, 3, 3, 256, 1, 1, biased=False, relu=False, name='res4a_branch2b')
            bn4a_branch2b = batch_normalization(res4a_branch2b, relu=True, name='bn4a_branch2b')
            res4a_branch2c = conv(bn4a_branch2b, 1, 1, 1024, 1, 1, biased=False, relu=False, name='res4a_branch2c')
            bn4a_branch2c = batch_normalization(res4a_branch2c, name='bn4a_branch2c')

            res4a = add([bn4a_branch1, bn4a_branch2c], name='res4a')
            res4a_relu = relu(res4a, name='res4a_relu')
            res4b_branch2a = conv(res4a_relu, 1, 1, 256, 1, 1, biased=False, relu=False, name='res4b_branch2a')
            bn4b_branch2a = batch_normalization(res4b_branch2a, relu=True, name='bn4b_branch2a')
            res4b_branch2b = conv(bn4b_branch2a, 3, 3, 256, 1, 1, biased=False, relu=False, name='res4b_branch2b')
            bn4b_branch2b = batch_normalization(res4b_branch2b, relu=True, name='bn4b_branch2b')
            res4b_branch2c = conv(bn4b_branch2b, 1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b_branch2c')
            bn4b_branch2c = batch_normalization(res4b_branch2c, name='bn4b_branch2c')

            res4b = add([res4a_relu, bn4b_branch2c], name='res4b')
            res4b_relu = relu(res4b, name='res4b_relu')
            res4c_branch2a = conv(res4b_relu, 1, 1, 256, 1, 1, biased=False, relu=False, name='res4c_branch2a')
            bn4c_branch2a = batch_normalization(res4c_branch2a, relu=True, name='bn4c_branch2a')
            res4c_branch2b = conv(bn4c_branch2a, 3, 3, 256, 1, 1, biased=False, relu=False, name='res4c_branch2b')
            bn4c_branch2b = batch_normalization(res4c_branch2b, relu=True, name='bn4c_branch2b')
            res4c_branch2c = conv(bn4c_branch2b, 1, 1, 1024, 1, 1, biased=False, relu=False, name='res4c_branch2c')
            bn4c_branch2c = batch_normalization(res4c_branch2c, name='bn4c_branch2c')

            res4c = add([res4b_relu, bn4c_branch2c], name='res4c')
            res4c_relu = relu(res4c, name='res4c_relu')
            res4d_branch2a = conv(res4c_relu, 1, 1, 256, 1, 1, biased=False, relu=False, name='res4d_branch2a')
            bn4d_branch2a = batch_normalization(res4d_branch2a, relu=True, name='bn4d_branch2a')
            res4d_branch2b = conv(bn4d_branch2a, 3, 3, 256, 1, 1, biased=False, relu=False, name='res4d_branch2b')
            bn4d_branch2b = batch_normalization(res4d_branch2b, relu=True, name='bn4d_branch2b')
            res4d_branch2c = conv(bn4d_branch2b, 1, 1, 1024, 1, 1, biased=False, relu=False, name='res4d_branch2c')
            bn4d_branch2c = batch_normalization(res4d_branch2c, name='bn4d_branch2c')

            res4d = add([res4c_relu, bn4d_branch2c], name='res4d')
            res4d_relu = relu(res4d, name='res4d_relu')
            res4e_branch2a = conv(res4d_relu, 1, 1, 256, 1, 1, biased=False, relu=False, name='res4e_branch2a')
            bn4e_branch2a = batch_normalization(res4e_branch2a, relu=True, name='bn4e_branch2a')
            res4e_branch2b = conv(bn4e_branch2a, 3, 3, 256, 1, 1, biased=False, relu=False, name='res4e_branch2b')
            bn4e_branch2b = batch_normalization(res4e_branch2b, relu=True, name='bn4e_branch2b')
            res4e_branch2c = conv(bn4e_branch2b, 1, 1, 1024, 1, 1, biased=False, relu=False, name='res4e_branch2c')
            bn4e_branch2c = batch_normalization(res4e_branch2c, name='bn4e_branch2c')

            res4e = add([res4d_relu, bn4e_branch2c], name='res4e')
            res4e_relu = relu(res4e, name='res4e_relu')
            res4f_branch2a = conv(res4e_relu, 1, 1, 256, 1, 1, biased=False, relu=False, name='res4f_branch2a')
            bn4f_branch2a = batch_normalization(res4f_branch2a, relu=True, name='bn4f_branch2a')
            res4f_branch2b = conv(bn4f_branch2a, 3, 3, 256, 1, 1, biased=False, relu=False, name='res4f_branch2b')
            bn4f_branch2b = batch_normalization(res4f_branch2b, relu=True, name='bn4f_branch2b')
            res4f_branch2c = conv(bn4f_branch2b, 1, 1, 1024, 1, 1, biased=False, relu=False, name='res4f_branch2c')
            bn4f_branch2c = batch_normalization(res4f_branch2c, name='bn4f_branch2c')

            res4f = add([res4e_relu, bn4f_branch2c], name='res4f')
            res4f_relu = relu(res4f, name='res4f_relu')
            res5a_branch1 = conv(res4f_relu, 1, 1, 2048, 2, 2, biased=False, relu=False, name='res5a_branch1')
            bn5a_branch1 = batch_normalization(res5a_branch1, name='bn5a_branch1')

            res5a_branch2a = conv(res4f_relu, 1, 1, 512, 2, 2, biased=False, relu=False, name='res5a_branch2a')
            bn5a_branch2a = batch_normalization(res5a_branch2a, relu=True, name='bn5a_branch2a')
            res5a_branch2b = conv(bn5a_branch2a, 3, 3, 512, 1, 1, biased=False, relu=False, name='res5a_branch2b')
            bn5a_branch2b = batch_normalization(res5a_branch2b, relu=True, name='bn5a_branch2b')
            res5a_branch2c = conv(bn5a_branch2b, 1, 1, 2048, 1, 1, biased=False, relu=False, name='res5a_branch2c')
            bn5a_branch2c = batch_normalization(res5a_branch2c, name='bn5a_branch2c')

            res5a = add([bn5a_branch1, bn5a_branch2c], name='res5a')
            res5a_relu = relu(res5a, name='res5a_relu')
            res5b_branch2a = conv(res5a_relu, 1, 1, 512, 1, 1, biased=False, relu=False, name='res5b_branch2a')
            bn5b_branch2a = batch_normalization(res5b_branch2a, relu=True, name='bn5b_branch2a')
            res5b_branch2b = conv(bn5b_branch2a, 3, 3, 512, 1, 1, biased=False, relu=False, name='res5b_branch2b')
            bn5b_branch2b = batch_normalization(res5b_branch2b, relu=True, name='bn5b_branch2b')
            res5b_branch2c = conv(bn5b_branch2b, 1, 1, 2048, 1, 1, biased=False, relu=False, name='res5b_branch2c')
            bn5b_branch2c = batch_normalization(res5b_branch2c, name='bn5b_branch2c')

            res5b = add([res5a_relu, bn5b_branch2c], name='res5b')
            res5b_relu = relu(res5b, name='res5b_relu')
            res5c_branch2a = conv(res5b_relu, 1, 1, 512, 1, 1, biased=False, relu=False, name='res5c_branch2a')
            bn5c_branch2a = batch_normalization(res5c_branch2a, relu=True, name='bn5c_branch2a')
            res5c_branch2b = conv(bn5c_branch2a, 3, 3, 512, 1, 1, biased=False, relu=False, name='res5c_branch2b')
            bn5c_branch2b = batch_normalization(res5c_branch2b, relu=True, name='bn5c_branch2b')
            res5c_branch2c = conv(bn5c_branch2b, 1, 1, 2048, 1, 1, biased=False, relu=False, name='res5c_branch2c')
            bn5c_branch2c = batch_normalization(res5c_branch2c, name='bn5c_branch2c')

            res5c = add([res5b_relu, bn5c_branch2c], name='res5c')
            res5c_relu = relu(res5c, name='res5c_relu')
            pool5 = avg_pool(res5c_relu, 7, 7, 1, 1, padding='VALID', name='pool5')
            self.fc_final = fc(pool5, self.NUM_CLASSES, relu=False, name='fc1000')

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