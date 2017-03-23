from  nn_tools.profilling import *
from collections import OrderedDict

def profilling(net,input=None):
    blob_dict = OrderedDict({'data': input})
    for i, layer in enumerate(net.net.layer):
        out = None
        if len(layer.top) == 1 and len(layer.bottom) == 1:
            if layer.type == 'Convolution':
                param = layer.convolution_param
                out = conv(blob_dict[layer.bottom[0]], param.kernel_size, param.num_output, param.stride,
                             param.pad, None, layer.name)
            if layer.type == 'ReLU':
                out = Activation(blob_dict[layer.bottom[0]], 'relu', layer.name)
            if layer.type == 'Pooling':
                param = layer.pooling_param
                out = Pool(blob_dict[layer.bottom[0]], param.kernel_size, param.stride,
                             param.pad, layer.name,param.pool)
            if out:
                blob_dict[layer.top[0]] = out()
            else:
                assert 'layer type: %s cannot be P' % (layer.type)
    return out()