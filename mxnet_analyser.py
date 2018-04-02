# coding=utf-8
from __future__ import absolute_import
import argparse
from analysis.MxnetA import *
from analysis.utils import save_csv
import os
import sys
import mxnet

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='python file location', type=str)
    parser.add_argument('name', help='the symbol object name or function that generate the symbol in your python file', type=str)
    parser.add_argument('shape',
                        help='input shape of the network(split by comma `,`), image shape should be: batch,c,h,w',
                        type=str)
    parser.add_argument('--out', help='path to save the csv file', default='/tmp/mxnet_analyse.csv', type=str)
    parser.add_argument('--func_args', help='args parse to the function in your python file', default='', type=str)
    parser.add_argument('--func_kwargs', help='kwargs parse to the function, eg. a=1,b=2', default='', type=str)

    args = parser.parse_args()
    path, filename = os.path.split(args.path)
    path=os.path.abspath(path)
    print(path)
    filename = os.path.splitext(filename)[0]
    sys.path.insert(0, path)
    exec ('from %s import %s as sym' % (filename, args.name))
    if isinstance(sym, mxnet.sym.Symbol):
        sym = sym
    elif hasattr(sym,'__call__'):
        _kwargs=args.func_kwargs.split(',')
        kwargs=[]
        for a in _kwargs:
            if a=='':
                continue
            kwargs.append(a.split('='))
        kwargs=dict(kwargs)
        sym = sym(*args.func_args.split(),**kwargs)
    else:
        assert ("Error, The sym is not a instance of mxnet.sym.Symbol or function")
    shape = [int(i) for i in args.shape.split(',')]
    profiling_symbol(sym,shape)
    save_csv(tracked_layers, '/tmp/a.csv')