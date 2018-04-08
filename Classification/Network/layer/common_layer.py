import tensorflow as tf
import sys

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