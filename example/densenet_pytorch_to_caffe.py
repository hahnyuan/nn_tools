import sys
sys.path.insert(0,'.')
sys.path.append('../')
import torch
from torch.autograd import Variable
from torchvision.models.densenet import *
import pytorch_to_caffe

if __name__=='__main__':
    name='densenet121'
    net=densenet121(True)
    input=Variable(torch.ones([1,3,224,224]))
    net.eval()
    pytorch_to_caffe.trans_net(net,input,name)
    pytorch_to_caffe.save_prototxt('{}.prototxt'.format(name))
    pytorch_to_caffe.save_caffemodel('{}.caffemodel'.format(name))