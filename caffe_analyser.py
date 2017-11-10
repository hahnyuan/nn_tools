import argparse
from analysis.CaffeA import *
from Caffe import caffe_net
from analysis.utils import save_csv

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('prototxt',help='path of the prototxt file',type=str)
    parser.add_argument('outdir',help='path to save the csv file',type=str)
    parser.add_argument('shape',help='input shape of the network(split by comma `,`), image shape should be: batch,h,w,c',type=str)
    args=parser.parse_args()
    shape=[int(i) for i in args.shape.split(',')]
    net=caffe_net.Prototxt(args.prototxt)
    blob_dict, layers=profiling(net, Blob(shape))
    save_csv(layers,args.outdir)