"""Builds the C3D network.

Implements the inference pattern for model building.
model(): Builds the model as far as is required for running the network
forward to make predictions.
"""
import re

import numpy as np
import tensorflow as tf


def accuracy(logit, labels):
    correct_pred = tf.equal(tf.argmax(logit, 1), labels)
    return tf.metrics.mean(tf.cast(correct_pred, tf.float32))


def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % 'tower', '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable

    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, wd=None):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with "Xavier" initialization.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.

    Returns:
      Variable Tensor
    """
    var = _variable_on_cpu(
        name,
        shape,
        tf.contrib.layers.xavier_initializer())
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def conv_3d(kernel_name, biases_name, input, kernel_shape,
            biases_shape, kernel_wd, biases_wd=None):
    kernel = _variable_with_weight_decay(kernel_name, kernel_shape, kernel_wd)
    conv = tf.nn.conv3d(input, kernel, [1, 1, 1, 1, 1], padding='SAME')
    biases = _variable_with_weight_decay(biases_name, biases_shape, biases_wd)
    pre_activation = tf.nn.bias_add(conv, biases)
    return pre_activation


def max_pool(name, l_input, k):
    return tf.nn.max_pool3d(l_input, ksize=[1, k, 2, 2, 1],
                            strides=[1, k, 2, 2, 1], padding='SAME',
                            name=name)


def model(videos, width, height, channel, num_class, dropout=1, complexity_factor=1):
    """Generate the 3d convolution classification output according to the input
      videos

    Args:
      videos: Data Input, the shape of the Data Input is
        [batch_size, sequence_size, height, weight, channel]
      channel: Image channel (1 for grey image or 3 for color image)
      num_class: number of class
      dropout: [0, 1], dropout value for fully connected layer
      complexity_factor: Integer, the complexity of the model, default is 1 (Should be integer).
    Return:
      out: classification result, the shape is [batch_size, num_classes]
    """
    # Add shape information of the video to video tensor
    videos.set_shape([None, None, height, width, channel])

    # Conv1 Layer
    with tf.variable_scope('conv1') as scope:
        conv1 = conv_3d('weight', 'biases', videos,
                        [3, 3, 3, channel, 16 * complexity_factor], [16 * complexity_factor], 0.0005)
        conv1 = tf.nn.relu(conv1, name=scope.name)
        _activation_summary(conv1)

    # pool1
    pool1 = max_pool('pool1', conv1, k=1)

    # Conv2 Layer
    with tf.variable_scope('conv2') as scope:
        conv2 = conv_3d('weight', 'biases', pool1,
                        [3, 3, 3, 16 * complexity_factor, 32 * complexity_factor], [32 * complexity_factor], 0.0005)
        conv2 = tf.nn.relu(conv2, name=scope.name)
        _activation_summary(conv2)

    # pool2
    pool2 = max_pool('pool2', conv2, k=2)

    # Conv3 Layer
    with tf.variable_scope('conv3') as scope:
        conv3 = conv_3d('weight_a', 'biases_a', pool2,
                        [3, 3, 3, 32 * complexity_factor, 64 * complexity_factor], [64 * complexity_factor], 0.0005)
        conv3 = tf.nn.relu(conv3, name=scope.name + 'a')
        conv3 = conv_3d('weight_b', 'biases_b', conv3,
                        [3, 3, 3, 64 * complexity_factor, 64 * complexity_factor], [64 * complexity_factor], 0.0005)
        conv3 = tf.nn.relu(conv3, name=scope.name + 'b')
        _activation_summary(conv3)

    # pool3
    pool3 = max_pool('pool3', conv3, k=2)

    # Conv4 Layer
    with tf.variable_scope('conv4') as scope:
        conv4 = conv_3d('weight_a', 'biases_a', pool3,
                        [3, 3, 3, 64 * complexity_factor, 128 * complexity_factor], [128 * complexity_factor],
                        0.0005)
        conv4 = tf.nn.relu(conv4, name=scope.name + 'a')
        conv4 = conv_3d('weight_b', 'biases_b', conv4,
                        [3, 3, 3, 128 * complexity_factor, 128 * complexity_factor], [128 * complexity_factor],
                        0.0005)
        conv4 = tf.nn.relu(conv4, name=scope.name + 'b')
        _activation_summary(conv4)

    # pool4
    pool4 = max_pool('pool4', conv4, k=2)

    # Conv5 Layer
    with tf.variable_scope('conv5') as scope:
        conv5 = conv_3d('weight_a', 'biases_a', pool4,
                        [3, 3, 3, 128 * complexity_factor, 128 * complexity_factor], [128 * complexity_factor],
                        0.0005)
        conv5 = tf.nn.relu(conv5, name=scope.name + 'a')
        conv5 = conv_3d('weight_b', 'biases_b', conv5,
                        [3, 3, 3, 128 * complexity_factor, 128 * complexity_factor], [128 * complexity_factor],
                        0.0005)
        conv5 = tf.nn.relu(conv5, name=scope.name + 'b')
        _activation_summary(conv5)

    # pool5
    pool5 = max_pool('pool5', conv5, k=2)

    # calculate the shape for the fully connected layer
    pool5_shape = pool5.get_shape().as_list()
    first_dimension = np.prod(pool5_shape[1:])
    second_dimension = first_dimension / 2

    # local6
    with tf.variable_scope('local6') as scope:
        weights = _variable_with_weight_decay('weights', [first_dimension, second_dimension], 0.0005)
        biases = _variable_with_weight_decay('biases', [second_dimension])
        pool5 = tf.transpose(pool5, perm=[0, 1, 4, 2, 3])
        local6 = tf.reshape(pool5, [-1, weights.get_shape().as_list()[0]])
        local6 = tf.nn.relu(tf.matmul(local6, weights) + biases, name=scope.name)
        local6 = tf.nn.dropout(local6, dropout)
        _activation_summary(local6)

    # local7
    with tf.variable_scope('local7') as scope:
        weights = _variable_with_weight_decay('weights', [second_dimension, second_dimension], 0.0005)
        biases = _variable_with_weight_decay('biases', [second_dimension])
        local7 = tf.nn.relu(tf.matmul(local6, weights) + biases, name=scope.name)
        local7 = tf.nn.dropout(local7, dropout)
        _activation_summary(local7)

    # linear layer(Wx + b)
    with tf.variable_scope('softmax_lineaer') as scope:
        weights = _variable_with_weight_decay('weights', [second_dimension, num_class], 0.0005)
        biases = _variable_with_weight_decay('biases', [num_class])
        softmax_linear = tf.add(tf.matmul(local7, weights), biases, name=scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear


def loss(logits, labels):
    """Add L2Loss to all the trainable variable

    Add summary for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference()
      labels: Labels from dataset. 1-D tensor of shape [batch_size]

    Returns:
      Loss tensor of type float
    """
    # Calculate the average cross entropy loss across the batch
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the across entropy loss plus all the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')
