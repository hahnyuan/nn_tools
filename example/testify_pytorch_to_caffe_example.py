import caffe
import torch
import numpy as np
import argparse
from collections import OrderedDict
from torch.autograd import Variable
import torch.nn as nn

def arg_parse():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model','-m',default='alexnet')
    parser.add_argument('--decimal','-d',default=2)
    parser.add_argument('--gpu','-gpu',action='store_true')
    args=parser.parse_args()
    return args

def generate_random(shape,gpu=False):
    data_np=np.random.rand(np.prod(shape)).reshape(shape)
    data_torch=Variable(torch.Tensor(data_np))
    if gpu:
        data_torch=data_torch.cuda()
    return [data_np],[data_torch]

def get_input_size(caffe_net):
    input_name = caffe_net.inputs[0]
    return caffe_net.blobs[input_name].data.shape

def forward_torch(net,data):
    blobs=OrderedDict()
    module2name={}


    for layer_name,m in net.named_modules():
        layer_name=layer_name.replace('.','_')
        module2name[m]=layer_name
        # turn off all the inplace operation
        if hasattr(m,'inplace'):
            m.inplace=False
    def forward_hook(m,i,o):
        o_np = o.data.cpu().numpy()
        blobs[module2name[m]]=o_np
    for m in net.modules():
        m.register_forward_hook(forward_hook)
    output=net.forward(*data)
    output=output.data.cpu().numpy()
    return blobs,output

def forward_caffe(net,data):
    for input_name,d in zip(net.inputs,data):
        net.blobs[input_name].data[...] = d
    rst=net.forward()
    blobs=OrderedDict()
    blob2layer={}
    for layer_name,tops in net.top_names.items():
        for top in tops:
            blob2layer[top]=layer_name
    for name,value in net.blobs.items():
        layer_name=blob2layer[name]
        value=value.data
        if layer_name in blobs:
            blobs[layer_name].append(value)
        else:
            blobs[layer_name]=[value]
    output_name=net.outputs[0]
    return blobs,rst[output_name]

def test(net_caffe,net_torch,data_np,data_torch):
    blobs_caffe, rst_caffe = forward_caffe(net_caffe, data_np)
    blobs_torch, rst_torch = forward_torch(net_torch, data_torch)
    # test the output of every layer
    fail=0
    for layer, value in blobs_caffe.items():
        if layer in blobs_torch:
            value_torch = blobs_torch[layer]
            value = value[0]
            if value.size!=value_torch.size:continue
            if 'relu' in layer: continue
            try:
                np.testing.assert_almost_equal(value, value_torch, decimal=args.decimal)
                print("TEST layer {}: PASS".format(layer))
            except:
                fail=1
                print("TEST layer {}: FAIL".format(layer))
                # np.testing.assert_almost_equal(np.clip(value, min=0), np.clip(value_torch, min=0))
    # test the output
    np.testing.assert_almost_equal(rst_caffe, rst_torch, decimal=args.decimal)
    if fail:
        print("TEST FAIL")
    else:
        print("TEST PASS")


if __name__=='__main__':
    args=arg_parse()

    if args.model=='alexnet':
        # Alexnet example
        from torchvision.models.alexnet import alexnet
        net_torch = alexnet(True).eval()
        if args.gpu:
            net_torch.cuda()
        try:
            net_caffe = caffe.Net('alexnet.prototxt', 'alexnet.caffemodel', caffe.TEST)
        except:
            raise ("Please run alexnet_pytorch_to_caffe.py first")
        shape=get_input_size(net_caffe)
        data_np,data_torch=generate_random(shape,args.gpu)
        test(net_caffe,net_torch,data_np,data_torch)

    elif args.model=='resnet18':
        # ResNet example
        from torchvision.models.resnet import resnet18
        net_torch = resnet18(True).eval()
        if args.gpu:
            net_torch.cuda()
        net_caffe = caffe.Net('resnet18.prototxt', 'resnet18.caffemodel', caffe.TEST)
        shape = get_input_size(net_caffe)
        data_np, data_torch = generate_random(shape, args.gpu)
        test(net_caffe,net_torch,data_np,data_torch)
    else:
        raise NotImplementedError()

    2==2
