from __future__ import absolute_import
import mxnet as mx
import mxnet.symbol as sym
import json
from analysis.layers import *
import re
import ctypes

from mxnet.ndarray import NDArray
import mxnet.ndarray as nd
from mxnet.base import NDArrayHandle, py_str



blob_dict=[]
tracked_layers = []

def tmpnet():
    x=sym.Variable('data')
    y=sym.Convolution(x,kernel=(3,3),num_filter=32)
    y=sym.Activation(y,'relu')
    y = sym.Convolution(y, kernel=(3, 3), num_filter=64,stride=(2,2),num_group=2)
    y=sym.softmax(y)
    return y

def analyse(data_infos,module_json,data_name='data'):

    datas={}
    for info in data_infos:
        datas[info[1]]=info[2]
    nodes=json.loads(module_json)['nodes']
    input=[]
    out=None
    for node in nodes:
        name=node['name']
        bottoms=[str(nodes[i[0]]['name']) for i in node['inputs']]
        for i,bottom in enumerate(bottoms):
            if bottom+'_output' in datas:
                bottoms[i]=datas[bottom+'_output']
            elif bottom+'_0' in datas:
                bottoms[i]=datas[bottom+'_0']
            elif bottom in datas:
                bottoms[i]=datas[bottom]
            else:
                cur_node=node
                while True:
                    bottom = [str(nodes[inp[0]]['name']) for inp in cur_node['inputs']][0]
                    if bottom + '_output' in datas:
                        bottoms[i] = datas[bottom + '_output']
                        break
                    elif bottom + '_0' in datas:
                        bottoms[i] = datas[bottom + '_0']
                        break
                    elif bottom in datas:
                        bottoms[i] = datas[bottom]
                        break
                    try:
                        bottom_node = nodes[cur_node['inputs'][0][0]]
                    except:
                        pass
                    cur_node=bottom_node
        if data_name==name:
            input.append(Blob(datas[data_name]))
        elif node['op']=='Convolution':
            kernel=eval(node['attrs']['kernel'])
            num_out=eval(node['attrs']['num_filter'])
            group_size=eval(node['attrs'].get('num_group','1'))
            pad=eval(node['attrs'].get('pad','(0,0)'))
            stride=eval(node['attrs'].get('stride','(1,1)'))
            x=Blob(bottoms[0])
            out=Conv(x,kernel_size=kernel,stride=stride,pad=pad,
                     num_out=num_out,group_size=group_size,name=name)
            tracked_layers.append(out)
        elif node['op']=='BatchNorm':
            x=Blob(bottoms[0])
            out = Norm(x, 'batch_norm',name=name)
            tracked_layers.append(out)
        elif node['op']=='FullyConnected':
            x=Blob(bottoms[0])
            num_hidden=eval(node['attrs']['num_hidden'])
            out=Fc(x,num_hidden,name=name)
            tracked_layers.append(out)
        elif node['op']=='Activation':
            pass
        elif 'elemwise' in node['op']:
            pass


class Monitor(object):
    def __init__(self, interval=1, pattern='.*', sort=False):
        def stat(x):
            return x.shape
        self.stat_func = stat
        self.interval = interval
        self.activated = False
        self.queue = []
        self.step = 0
        self.exes = []
        self.re_prog = re.compile(pattern)
        self.sort = sort
        def stat_helper(name, array):
            array = ctypes.cast(array, NDArrayHandle)
            array = NDArray(array, writable=False)
            if not self.activated or not self.re_prog.match(py_str(name)):
                return
            self.queue.append((self.step, py_str(name), stat(array)))
        self.stat_helper = stat_helper

    def install(self, exe):
        exe.set_monitor_callback(self.stat_helper)
        self.exes.append(exe)

    def tic(self):
        if self.step % self.interval == 0:
            for exe in self.exes:
                for array in exe.arg_arrays:
                    array.wait_to_read()
                for array in exe.aux_arrays:
                    array.wait_to_read()
            self.queue = []
            self.activated = True
        self.step += 1

    def toc(self):
        if not self.activated:
            return []
        for exe in self.exes:
            for array in exe.arg_arrays:
                array.wait_to_read()
            for array in exe.aux_arrays:
                array.wait_to_read()
        for exe in self.exes:
            for name, array in zip(exe._symbol.list_arguments(), exe.arg_arrays):
                self.queue.append((self.step, name, self.stat_func(array)))
            for name, array in zip(exe._symbol.list_auxiliary_states(), exe.aux_arrays):
                # if self.re_prog.match(name):
                self.queue.append((self.step, name, self.stat_func(array)))
        self.activated = False
        res = []
        if self.sort:
            self.queue.sort(key=lambda x: x[1])
        for n, k, v_list in self.queue:
            res.append((n, k, v_list))
        self.queue = []
        return res
    def toc_print(self):
        pass

def profiling_symbol(symbol,data_shape,data_name='data'):
    monitor = Monitor()
    model=mx.mod.Module(symbol)
    model.bind(data_shapes=[(data_name,tuple(data_shape))])
    model.install_monitor(monitor)
    model.init_params()
    monitor.tic()
    model.forward(mx.io.DataBatch(data=(nd.ones(data_shape),)))
    data_infos=monitor.toc()
    module_json=symbol.tojson()
    analyse(data_infos,module_json,data_name)
