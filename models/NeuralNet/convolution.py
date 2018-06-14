import tensorflow as tf
import numpy as np

ACT_LIST = ['lrelu', 'relu', 'linear', 'sigmoid', 'softmax', 'tanh']


def concat(tensors, axis, *args, **kwargs):
    return tf.concat(tensors, axis, *args, **kwargs)


def batch_norm(tensors, train=True, name='batch_norm'):
    return tf.contrib.layers.batch_norm(
        tensors, decay=0.9, updates_collections=None, epsilon=1e-5, is_training=train, scope=name, scale=True
    )


def conv_cond_concat(x, y):
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return concat([x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


def act_func(input_, activation):
    if activation == 'lrelu':
        return tf.maximum(input_, 0.2 * input_, name='lrelu')
    elif activation == 'relu':
        return tf.nn.relu(input_, name='relu')
    elif activation == 'sigmoid':
        return tf.nn.sigmoid(input_, name='sigmoid')
    elif activation == 'tanh':
        return tf.nn.tanh(input_, name='tanh')
    elif activation == 'softmax':
        return tf.nn.softmax(input_, name='softmax')
    else:
        with tf.variable_scope('linear'):
            return input_


def conv2dwithrandom(input_, k=3, s=1, k2=0, name='con2drandom', activation='relu', withbatch=True):
    assert activation in ACT_LIST, 'Unknown activation function'
    if k2 is 0:
        k2 = k
    with tf.variable_scope(name):
        w = tf.get_variable(
            'w', [k, k2, input_.get_shape()[-1], 1], initializer=tf.truncated_normal_initializer(stddev=0.02)
        )
        b = tf.get_variable('b', [1], initializer=tf.constant_initializer(0.0))
        w2 = tf.get_variable(
            'w2', [k, k2, input_.get_shape()[-1], 1], initializer=tf.truncated_normal_initializer(stddev=0.02)
        )
        b2 = tf.get_variable('b2', [1], initializer=tf.constant_initializer(0.0))

        conv1 = tf.nn.conv2d(input_, w, strides=[1, s, s, 1], padding='SAME') + b
        conv2 = tf.nn.conv2d(input_, w2, strides=[1, s, s, 1], padding='SAME') + b2
        if withbatch:
            conv1 = act_func(batch_norm(conv1), activation)
            conv2 = act_func(batch_norm(conv2, name='batch_norm2'), activation)
        else:
            conv1 = act_func(conv1, activation)
            conv2 = act_func(conv2, activation)

        stddev = tf.sqrt(tf.exp(conv1))
        salt = tf.random_normal([tf.shape(input_)[0], tf.shape(input_)[1], tf.shape(input_)[2], 1], conv2,
                                stddev=stddev + 0.002)

        return salt


def conv2d(input_, output_dim, k=3, s=1, k2=0, name='con2d', activation='relu', withbatch=True, withweight=False):
    assert activation in ACT_LIST, 'Unkwon activation function'
    if k2 is 0:
        k2 = k
    with tf.variable_scope(name):
        w = tf.get_variable(
            'w', [k, k2, input_.get_shape()[-1], output_dim], initializer=tf.truncated_normal_initializer(stddev=0.02)
        )
        b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input_, w, strides=[1, s, s, 1], padding='SAME') + b

        if withbatch:
            if withweight:
                return act_func(batch_norm(conv), activation), w
            else:
                return act_func(batch_norm(conv), activation)
        else:
            if withweight:
                return act_func(conv, activation), w
            else:
                return act_func(conv, activation)


def decon2d_with_upsampling(
        input_, output_shape, k=5, s=1, name='deconv2d+upsample', withbatch=False, activation='relu', issampling=False):
    with tf.variable_scope(name):
        w = tf.get_variable(
            'w', [k, k, input_.get_shape()[-1], output_shape[-1]], initializer=tf.random_normal_initializer(stddev=0.02)
        )
        b = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))

        upsample = upsampling2d(input_, output_shape[-2])
        deconv = tf.nn.conv2d(upsample, w, strides=[1, s, s, 1], padding='SAME') + b

        return (lambda conv, tag: act_func(
            batch_norm(conv, train=issampling), activation) if tag else act_func(conv, activation))(deconv, withbatch)


def deconv2d(
        input_, output_shape, k=5, s=2, name='deconv2d', with_w=False, withbatch=False, activation='relu',
        issampling=False):
    with tf.variable_scope(name):
        w = tf.get_variable(
            'w', [k, k, output_shape[-1], input_.get_shape()[-1]], initializer=tf.random_normal_initializer(stddev=0.02)
        )

        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, s, s, 1])
        b = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = deconv + b

        if with_w:
            return (lambda conv, tag: act_func(
                batch_norm(conv, train=not issampling),
                activation) if tag else act_func(conv, activation))(deconv, withbatch), w, b
        else:
            return (lambda conv, tag: act_func(
                batch_norm(conv, train=not issampling),
                activation) if tag else act_func(conv, activation))(deconv, withbatch)


def maxpool(input_, k=3, s=2, name='maxpool'):
    with tf.variable_scope(name):
        return tf.nn.max_pool(input_, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='SAME')


def avgpool(input_, k, s, name='avgpool'):
    with tf.variable_scope(name):
        return tf.nn.avg_pool(input_, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='SAME')


def fc(
        input_, output_dim, name='fc', keepprob=0.7, withdropout=False, activation='relu', withbatch=False,
        issampling=False):
    if len(input_.get_shape()) != 2:
        input_ = makeflat(input_)
    with tf.variable_scope(name):
        w = tf.get_variable(
            'w', [input_.get_shape()[1], output_dim], initializer=tf.random_normal_initializer(stddev=0.02))
        b = tf.get_variable(
            'biases', [output_dim], initializer=tf.constant_initializer(0.0)
        )
        if withbatch:
            h = act_func(batch_norm(tf.matmul(input_, w) + b, train=not issampling), activation=activation)
        else:
            h = act_func(tf.matmul(input_, w) + b, activation=activation)
        if withdropout:
            return tf.nn.dropout(h, keep_prob=keepprob)
        else:
            return h


def makeflat(input_):
    input_shape = input_.get_shape()
    assert len(input_shape) > 1, 'Tensor size is strange....!'
    return tf.reshape(input_, [-1, int(np.prod(input_shape[1:]))])


def upsampling2d(input_, new_size, method='bilinear'):
    assert method in ['bilinear', 'bicubic'], 'Unacceptable upscaling method'
    if method == 'bicubic':
        return tf.image.resize_bicubic(input_, [new_size, new_size])
    else:
        return tf.image.resize_bilinear(input_, [new_size, new_size])
