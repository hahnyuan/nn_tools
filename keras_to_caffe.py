import keras
import Caffe.net as caffe
import numpy as np
from funcs import *

def convert_filter(numpy_filter_weight):
    return np.transpose(numpy_filter_weight,(3,2,1,0))

def convert_fc(numpy_fc_weight):
    return np.transpose(numpy_fc_weight,(1,0))

def keras_weights_to_caffemodel(keras_model):
    """
    Only Implement the conv layer and fc layer
    :param keras_model:
    :return:
    """
    net=caffe.Net()
    layers=keras_model.layers

    for layer in layers:
        if type(layer)==keras.layers.Convolution2D:
            w,b=layer.get_weights()
            w=convert_filter(w)
            param=caffe.Layer_param(layer.name,'Convolution')
            net.add_layer_with_data(param,[w,b])
        if type(layer)==keras.layers.Dense:
            w, b = layer.get_weights()
            w = convert_fc(w)
            param = caffe.Layer_param(layer.name, 'InnerProduct')
            net.add_layer_with_data(param, [w, b])
    return net

if __name__=='__main__':
    pass