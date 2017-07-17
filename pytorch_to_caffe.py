import argparse
from Caffe import caffe_net

parser=argparse.ArgumentParser()
parser.add_argument('src_net',help='pytorch model net file path',type=str)
parser.add_argument('src_pth',help='pytorch model weight pth file path',type=str)
parser.add_argument('src_pth',help='pytorch model weight pth file',type=str)
parser.add_argument('dst',help='output model dir')
