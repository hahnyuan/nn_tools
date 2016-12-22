# this file if for constructing the tensorflow network quickly
# consisting a lot of functions

import numpy as np
import tensorflow as tf

# ==========Variables=============

def weight_variable_norm(shape):
    #   generate a variable with the name of 'bias'.
    #  reuturn a Variable whose shape is 'shape'
    initial = tf.truncated_normal(shape, stddev=0.1, name='weight')
    weight=tf.Variable(initial)
    add_weight_l2_regularation(weight)
    return weight


def bias_variable_zero(shape):
    #   generate a variable with
    initial = tf.constant(0.1, shape=shape, name='bias')
    return tf.Variable(initial)


def in_shape(input):
    # return the list format of the input tensor
    return input.get_shape().as_list()


def to_categorical(y, nb_classes=None):
    '''Convert class vector (integers from 0 to nb_classes) to binary class matrix, for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
        nb_classes: total number of classes

    # Returns
        A binary matrix representation of the input.
    '''
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y


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

def conv2d_with_weight(name,input,weights,biases,strides=(1,1,1,1),padding='SAME'):
    # the standard conv2d generator with weights and biases given
    in_channel = in_shape(input)[3]
    with tf.name_scope(name):
        W = tf.Variable(weights)
        b = tf.Variable(biases)
        conv = tf.nn.conv2d(input, W, strides=strides, padding=padding)
        conv = tf.nn.bias_add(conv, b)
    return conv

def conv3x3(name, input, numout):
    # the standard 3x3 conv2d used in many situation
    return conv2d(name,input,numout,3)


# ==========Fully Connected=========

def fc(name,input,numout,with_relu=1):
    input = tf.reshape(input,[-1,np.prod(in_shape(input)[1:])])
    in_channel = in_shape(input)[1]
    with tf.name_scope(name):
        W = weight_variable_norm([in_channel, numout])
        b = bias_variable_zero([numout])
        rst= tf.matmul(input,W)+b
        if with_relu:
            rst=tf.nn.relu(rst)
    return rst


# ===========Pooling=============

def max_pool2d(name,input,ksize=(1,2,2,1),strides=(1,2,2,1)):
    # the standard pooling
    return
    tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def pool2x2(name, input):
    # poolinh with
    with tf.name_scope(name):
        return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# ===========Activation=====

def prule(name,input):
    with tf.name_scope(name):
        pos=tf.nn.relu(input,'pos')
        shape=in_shape(input)[-1]
        w=weight_variable_norm([shape])
        neg=tf.nn.relu(-input,'neg')*w
        rst=pos+neg
    return rst

# ===========Loss============

def classific_loss(y,y_):
    # the shape of y and y_ is [None,class_num]
    # return cross_entropy loss
    diff = tf.nn.softmax_cross_entropy_with_logits(y, y_)
    cross_entropy = tf.reduce_mean(diff)
    return cross_entropy

l2_weight_loss=[]
def add_weight_l2_regularation(weights):
    l2_weight_loss.append(tf.nn.l2_loss(weights))

def add_l2_loss(loss,factor=0.001):
    for i in l2_weight_loss:
        loss+=i*factor
    return loss

# ===========Evaluate===========

def classific_accurancy(y,y_):
    # the shape of y and y_ is [None,class_num]
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy