# coding=utf-8
from __future__ import absolute_import
import torch
import argparse
import sys,os
from analysis.PytorchA import profiling
from analysis.utils import save_csv
from torch.autograd import Variable

"""
Supporting analyse the inheritors of torch.nn.Moudule class.

Command：`pytorch_analyser.py [-h] [--out OUT] [--class_args ARGS] path class_name shape`
- The path is the python file path which contaning your class.
- The class_name is the class name in your python file.
- The shape is the input shape of the network(split by comma `,`), in pytorch image shape should be: batch_size, channel, image_height, image_width.
- The out (optinal) is path to save the csv file, default is '/tmp/pytorch_analyse.csv'.
- The class_args (optional) is the args to init the class in python file, default is empty.

For example `python pytorch_analyser.py tmp/pytorch_analysis_test.py ResNet218 1,3,224,224`
"""


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('path',help='python file location',type=str)
    parser.add_argument('class_name',help='class name in python file',type=str)
    parser.add_argument('shape',help='input shape of the network(split by comma `,`), image shape should be: batch,h,w,c',type=str)
    parser.add_argument('--out',help='path to save the csv file',default='/tmp/pytorch_analyse.csv',type=str)
    parser.add_argument('--class_args',help='args to init the class in python file',default='',type=str)

    args=parser.parse_args()
    path,filename=os.path.split(args.path)
    filename=os.path.splitext(filename)[0]
    sys.path.insert(0,path)
    exec('from %s import %s as Net'%(filename,args.class_name))
    net=Net(*args.class_args.split())
    shape = [int(i) for i in args.shape.split(',')]
    x = Variable(torch.rand(shape))
    blob_dict, layers = profiling(net, x)
    save_csv(layers, args.out)


