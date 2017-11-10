import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from collections import OrderedDict
from layers import *

tracked_layers=[]
blob_dict=[]

def analyse(module,raw_input):
    input=[]
    for i in raw_input:
        s = i.size()
        if len(s)==4:
            input.append(Blob([s[0],s[2],s[3],s[1]]))
        else:
            input.append(Blob(s))
    out=None
    if isinstance(module,nn.Conv2d):
        out=conv(input[0],module.kernel_size,module.out_channels,
                 module.stride,module.padding)
    elif isinstance(module,nn.BatchNorm2d):
        out=Norm(input[0],'batch_norm')
    elif isinstance(module,nn.Linear):
        out=fc(input[0],module.out_features)
    elif isinstance(module,nn.MaxPool2d):
        out = pool(input[0], module.kernel_size,module.stride,module.padding,
                   name='max_pool',pool_type='max')
    elif isinstance(module,nn.AvgPool2d):
        out = pool(input[0], module.kernel_size,module.stride,module.padding,
                   name='avg_pool',pool_type='avg')
    elif isinstance(module,nn.ReLU):
        out = Activation(input[0],'relu')
    if out:
        tracked_layers.append(out)
    else:
        print 'WARNING: skip layer %s' % str(module)

def module_hook(module, input, output):
    # print('module hook')
    # print module
    # for i in input:
    #     print ('input',i.size())
    # for i in output:
    #     print('out', i.size())
    analyse(module,input)

def register(module):
    module.register_forward_hook(module_hook)

def profiling(net,input):
    if isinstance(input,dict):
        assert NotImplementedError
    else:
        assert isinstance(input,Variable),\
            "profiling input must be Variable or list of Variable"
    net.apply(register)
    net.forward(input)
    return blob_dict,tracked_layers