# this file if for constructing the tensorflow network quickly
# consisting a lot of functions

import numpy as np
import tensorflow as tf

# ==========Variables=============

def weight_variable_norm(shape):
    #   generate a variable with the name of 'bias'.
    #  reuturn a Variable whose shape is 'shape'
    initial = tf.truncated_normal(shape, stddev=0.1, name='bias')
    return tf.Variable(initial)


def bias_variable_zero(shape):
    #   generate a variable with
    initial = tf.constant(0.1, shape=shape, name='bias')
    return tf.Variable(initial)


def in_shape(input):
    # return the list format of the input tensor
    return input.get_shape().as_list()


# ===========Convolution=============

def conv2d(name, input, numout, kernel_size, strides=(1, 1, 1, 1), padding='SAME'):
    # the standard conv2d generator
    in_channel = in_shape(input)[3]
    with tf.name_scope(name):
        W = weight_variable_norm([kernel_size, kernel_size, in_channel, numout])
        b = bias_variable_zero([numout])
        conv = tf.nn.conv2d(input, W, strides=strides, padding=padding)
        conv = tf.nn.bias_add(conv, b)
    return conv


def conv3x3(name, input, numout):
    # the standard 3x3 conv2d used in many situation
    return conv2d(name,input,numout,3)

# ==========Fully Connected=========

def fc(name,input,numout):
    input = tf.reshape(input,[-1,np.prod(in_shape(input)[1:])])
    in_channel = in_shape(input)[1]
    with tf.name_scope(name):
        W = weight_variable_norm([in_channel, numout])
        b = bias_variable_zero([numout])
        rst= tf.matmul(input,W)+b
    return rst


# ===========Pooling=============

def max_pool2d(input,ksize=(1,2,2,1),strides=(1,2,2,1)):
    # the standard pooling
    return
    tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def pool2x2(name, input):
    # poolinh with
    with tf.name_scope(name):
        return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
