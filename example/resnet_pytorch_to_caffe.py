import sys
sys.path.insert(0,'.')
import torch
from torch.autograd import Variable
from torchvision.models.resnet import *
import pytorch_to_caffe


if __name__=='__main__':
    name='resnet18'
    net=resnet18(True)
    net.eval()
    input=Variable(torch.ones([1,3,224,224]))
    pytorch_to_caffe.trans_net(net,input,name)
    pytorch_to_caffe.save_prototxt('{}.prototxt'.format(name))
    pytorch_to_caffe.save_caffemodel('{}.caffemodel'.format(name))