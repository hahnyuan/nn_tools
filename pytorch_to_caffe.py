import torch
import torch.nn as nn
import traceback
from Caffe import caffe_net
import torch.nn.functional as F
from torch.autograd import Variable
from Caffe import layer_param
from torch.nn.modules.utils import _pair
import numpy as np
import inspect

"""
How to support a new layer type:
 layer_name=log.add_layer(layer_type_name)
 top_blobs=log.add_blobs(<output of that layer>)
 layer=caffe_net.Layer_param(xxx)
 <set layer parameters>
 [<layer.add_data(*datas)>]
 log.cnet.add_layer(layer)
 
Please MUTE the inplace operations to avoid not find in graph
"""


NET_INITTED=False
WARNING_STRINGS=''
RP_TRANSFERRING_FLAG=False  # this flag prevents transferring Rp function in Rp function.

class Blob_LOG():
    def __init__(self):
        self.data={}
    def __setitem__(self, key, value):
        self.data[key]=value
    def __getitem__(self, key):
        return self.data[key]
    def __len__(self):
        return len(self.data)

class TransLog(object):

    def __init__(self):
        """
        doing init() with inputs Variable before using it
        """
        self.layers={}
        self._blobs=Blob_LOG()
        self._blobs_data=[]
        self.cnet=caffe_net.Caffemodel('')
        self.debug=False
        self.pytorch_layer_name=None

    def init(self,inputs):
        """
        :param inputs: is a list of input variables
        """
        self.add_blobs(inputs)
    def add_layer(self,name='layer'):
        name='noname_'+name
        if name in self.layers:
            return self.layers[name]
        if self.pytorch_layer_name:
            pytorch_name=self.pytorch_layer_name.replace('.','_')
            name=pytorch_name
            cnt=1
            while name in self.layers:
                name='{}_sub{}'.format(pytorch_name,cnt)
                cnt+=1
            self.pytorch_layer_name=None
        else:
            name='{}{}'.format(name,len(self.layers))
        self.layers[name]=name
        if self.debug:
            print("{} was added to layers".format(self.layers[name]))
        return self.layers[name]

    def add_blobs(self, blobs,name='blob',with_num=True):
        rst=[]
        for blob in blobs:
            self._blobs_data.append(blob) # to block the memory address be rewrited
            blob_id=int(id(blob))
            if with_num:
                rst.append('{}{}'.format(name,len(self._blobs)))
            else:
                rst.append('{}'.format(name))
            if self.debug:
                print("{}:{} was added to blobs".format(blob_id,rst[-1]))
            print('Add blob {} : {}'.format(rst[-1].center(21),blob.size()))
            self._blobs[blob_id]=rst[-1]
        return rst

    def get_blobs(self, var):
        var=id(var)
        if self.debug:
            print("{}:{} getting".format(var, self._blobs[var]))
        try:
            return self._blobs[var]
        except:
            print("===\nWARNING: CANNOT FOUND blob at layer {}, this may cause a NoneType Error. "
                  "This may caused by the previous operation which produce the blob(tensor) is not implemented in nn_tools. "
                  "You can issue this at https://github.com/hahnyuan/nn_tools/issues. \n===".format(self.pytorch_layer_name))
            return None

log=TransLog()

layer_names={}

class Rp(object):
    def __init__(self,raw,replace,**kwargs):
        # replace the raw function to replace function
        self.obj=replace
        self.raw=raw

    def __call__(self,*args,**kwargs):
        global RP_TRANSFERRING_FLAG
        if RP_TRANSFERRING_FLAG:
            return self.raw(*args,**kwargs)
        RP_TRANSFERRING_FLAG=True
        if not NET_INITTED:
            return self.raw(*args,**kwargs)
        for stack in traceback.walk_stack(None):
            if 'self' in stack[0].f_locals:
                layer=stack[0].f_locals['self']
                if layer in layer_names:
                    log.pytorch_layer_name=layer_names[layer]
                    print("Processing Layer: "+layer_names[layer])
                    break
        out=self.obj(self.raw,*args,**kwargs)
        RP_TRANSFERRING_FLAG=False
        # if isinstance(out,Variable):
        #     out=[out]
        return out

# ----- for torch.nn.functional operations -----
def _conv2d(raw,input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    x=raw(input,weight,bias,stride,padding,dilation,groups)
    name=log.add_layer(name='conv')
    log.add_blobs([x],name='conv_blob')
    layer=caffe_net.Layer_param(name=name, type='Convolution',
                                bottom=[log.get_blobs(input)], top=[log.get_blobs(x)])
    layer.conv_param(x.size()[1],weight.size()[2:],stride=_pair(stride),
                     pad=_pair(padding),dilation=_pair(dilation),bias_term=bias is not None,groups=groups)
    if bias is not None:
        layer.add_data(weight.cpu().data.numpy(),bias.cpu().data.numpy())
    else:
        layer.param.convolution_param.bias_term=False
        layer.add_data(weight.cpu().data.numpy())
    log.cnet.add_layer(layer)
    return x

def _conv_transpose2d(raw,input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    x=raw(input, weight, bias, stride, padding, output_padding, groups, dilation)
    name=log.add_layer(name='conv_transpose')
    log.add_blobs([x],name='conv_transpose_blob')
    layer=caffe_net.Layer_param(name=name, type='Deconvolution',
                                bottom=[log.get_blobs(input)], top=[log.get_blobs(x)])
    layer.conv_param(x.size()[1],weight.size()[2:],stride=_pair(stride),
                     pad=_pair(padding),dilation=_pair(dilation),bias_term=bias is not None)
    if bias is not None:
        layer.add_data(weight.cpu().data.numpy(),bias.cpu().data.numpy())
    else:
        layer.param.convolution_param.bias_term=False
        layer.add_data(weight.cpu().data.numpy())
    log.cnet.add_layer(layer)
    return x

def _interpolate(raw,input, size=None, scale_factor=None, mode='nearest', align_corners=None):
    raise NotImplementedError("The interpolate upsampling in pytorch cannot be implimented in caffe by This function, I'll try later. ")

    if mode=='bilinear':
        x=raw(input, size, scale_factor, mode, align_corners)
    else:
        raise NotImplementedError("The interpolate upsampling only support bilinear in Caffe")
    name=log.add_layer(name='interpolate')
    log.add_blobs([x],name='interpolate_blob')
    layer=caffe_net.Layer_param(name=name, type='Deconvolution',
                                bottom=[log.get_blobs(input)], top=[log.get_blobs(x)])

    def bilinear_weight(shape):
        weight = np.zeros(np.prod(shape), dtype='float32')
        f = np.ceil(shape[3] / 2.)
        c = (2 * f - 1 - f % 2) / (2. * f)
        for i in range(np.prod(shape)):
            x = i % shape[3]
            y = (i / shape[3]) % shape[2]
            weight[i] = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
        return weight.reshape(shape)
    kernel_size=2*scale_factor-scale_factor%2
    stride=scale_factor
    pad=int(np.ceil((scale_factor-1)/2))
    channels=x.size(1)
    weight=bilinear_weight([channels,1,kernel_size,kernel_size])
    layer.conv_param(channels,kernel_size,stride=stride,pad=pad,bias_term=False,groups=channels)
    layer.add_data(weight)
    log.cnet.add_layer(layer)
    return x

def _linear(raw,input, weight, bias=None):
    x=raw(input,weight,bias)
    layer_name=log.add_layer(name='fc')
    top_blobs=log.add_blobs([x],name='fc_blob')
    layer=caffe_net.Layer_param(name=layer_name, type='InnerProduct',
                                bottom=[log.get_blobs(input)], top=top_blobs)
    layer.fc_param(x.size()[1],has_bias=bias is not None)
    if bias is not None:
        layer.add_data(weight.cpu().data.numpy(),bias.cpu().data.numpy())
    else:
        layer.add_data(weight.cpu().data.numpy())
    log.cnet.add_layer(layer)
    return x

def _pool(type,raw,input,x,kernel_size,stride,padding,ceil_mode):
    # TODO dilation,ceil_mode,return indices
    layer_name = log.add_layer(name='{}_pool'.format(type))
    top_blobs = log.add_blobs([x], name='{}_pool_blob'.format(type))
    layer = caffe_net.Layer_param(name=layer_name, type='Pooling',
                                  bottom=[log.get_blobs(input)], top=top_blobs)
    # TODO w,h different kernel, stride and padding
    # processing ceil mode
    layer.pool_param(kernel_size=kernel_size, stride=kernel_size if stride is None else stride,
                     pad=padding, type=type.upper())
    log.cnet.add_layer(layer)
    if ceil_mode==False and stride is not None:
        oheight = (input.size()[2] - _pair(kernel_size)[0] + 2 * _pair(padding)[0]) % (_pair(stride)[0])
        owidth = (input.size()[3] - _pair(kernel_size)[1] + 2 * _pair(padding)[1]) % (_pair(stride)[1])
        if oheight!=0 or owidth!=0:
            caffe_out=raw(input, kernel_size, stride, padding, ceil_mode=True)
            warn="WARN: the output shape miss match at {}: " \
                  "input {} output---Pytorch:{}---Caffe:{}\n" \
                  "This is caused by the different implementation that ceil mode in caffe and the floor mode in pytorch" \
                 ".\n".format(layer_name,input.size(),x.size(),caffe_out.size())+ \
                "WARN: Adding the clip layer `{}` `{}` in caffe prototxt to solve the shape mismatch error in caffe. " \
                "You can remove them manually if you don't need them.\n".format(layer_name + '_slice1',layer_name + '_slice2')
            print(warn)
            global WARNING_STRINGS
            WARNING_STRINGS+=warn
            top_name=top_blobs[0]
            tmp1_name=top_name+'_tmp1'
            drop1_name=top_name+'_drop1'
            tmp2_name=top_name+'_tmp2'
            drop2_name=top_name+'_drop2'
            log.cnet.net.layer[-1].top[0]=tmp1_name

            slice1_layer=caffe_net.Layer_param(name=layer_name+'_slice1',type='Slice',bottom=[tmp1_name],top=[tmp2_name,drop1_name])
            slice1_layer.slice_param(-1,[x.size()[-1]])
            log.cnet.add_layer(slice1_layer)
            slice2_layer = caffe_net.Layer_param(name=layer_name + '_slice2', type='Slice', bottom=[tmp2_name], top=top_blobs+[drop2_name])
            slice2_layer.slice_param(-2, [x.size()[-2]])
            log.cnet.add_layer(slice2_layer)

def _max_pool2d(raw,input, kernel_size, stride=None, padding=0, dilation=1,
               ceil_mode=False, return_indices=False):
    x = raw(input, kernel_size, stride, padding, dilation,ceil_mode, return_indices)
    _pool('max',raw,input, x, kernel_size, stride, padding,ceil_mode)
    return x

def _avg_pool2d(raw,input, kernel_size, stride = None, padding = 0, ceil_mode = False, count_include_pad = True):
    x = raw(input, kernel_size, stride, padding, ceil_mode, count_include_pad)
    _pool('ave',raw,input, x, kernel_size, stride, padding,ceil_mode)
    return x

def _dropout(raw,input,p=0.5, training=False, inplace=False):
    x=raw(input,p, training, False)
    bottom_blobs=[log.get_blobs(input)]
    layer_name=log.add_layer(name='dropout')
    top_blobs=log.add_blobs([x],name=bottom_blobs[0],with_num=False)
    layer=caffe_net.Layer_param(name=layer_name,type='Dropout',
                                bottom=bottom_blobs,top=top_blobs)
    layer.param.dropout_param.dropout_ratio = p
    layer.param.include.extend([caffe_net.pb.NetStateRule(phase=0)]) # 1 for test, 0 for train
    log.cnet.add_layer(layer)
    return x

def _threshold(raw,input, threshold, value, inplace=False):
    # for threshold or relu
    if threshold==0 and value==0:
        x = raw(input,threshold, value, False)
        name = log.add_layer(name='relu')
        log.add_blobs([x], name='relu_blob')
        layer = caffe_net.Layer_param(name=name, type='ReLU',
                                      bottom=[log.get_blobs(input)], top=[log.get_blobs(x)])
        log.cnet.add_layer(layer)
        return x
    if value!=0:
        raise NotImplemented("value !=0 not implemented in caffe")
    x=raw(input,input, threshold, value, False)
    bottom_blobs=[log.get_blobs(input)]
    layer_name=log.add_layer(name='threshold')
    top_blobs=log.add_blobs([x],name='threshold_blob')
    layer=caffe_net.Layer_param(name=layer_name,type='Threshold',
                                bottom=bottom_blobs,top=top_blobs)
    layer.param.threshold_param.threshold = threshold
    log.cnet.add_layer(layer)
    return x

def _relu(raw, input, inplace=False):
    # for threshold or prelu
    x = raw(input, False)
    name = log.add_layer(name='relu')
    log.add_blobs([x], name='relu_blob')
    layer = caffe_net.Layer_param(name=name, type='ReLU',
                                  bottom=[log.get_blobs(input)], top=[log.get_blobs(x)])
    log.cnet.add_layer(layer)
    return x

def _prelu(raw, input, weight):
    # for threshold or prelu
    x = raw(input, weight)
    name = log.add_layer(name='prelu')
    log.add_blobs([x], name='prelu_blob')
    layer = caffe_net.Layer_param(name=name, type='PReLU',
                                  bottom=[log.get_blobs(input)], top=[log.get_blobs(x)])
    if weight.size()[0]==1:
        layer.param.prelu_param.channel_shared=True
        layer.add_data(weight.cpu().data.numpy()[0])
    else:
        layer.add_data(weight.cpu().data.numpy())
    log.cnet.add_layer(layer)
    return x

def _leaky_relu(raw, input, negative_slope=0.01, inplace=False):
    x = raw(input, negative_slope)
    name = log.add_layer(name='leaky_relu')
    log.add_blobs([x], name='leaky_relu_blob')
    layer = caffe_net.Layer_param(name=name, type='ReLU',
                                  bottom=[log.get_blobs(input)], top=[log.get_blobs(x)])
    layer.param.relu_param.negative_slope=negative_slope
    log.cnet.add_layer(layer)
    return x

def _tanh(raw, input):
    # for tanh activation
    x = raw(input)
    name = log.add_layer(name='tanh')
    log.add_blobs([x], name='tanh_blob')
    layer = caffe_net.Layer_param(name=name, type='TanH',
                                  bottom=[log.get_blobs(input)], top=[log.get_blobs(x)])
    log.cnet.add_layer(layer)
    return x

def _softmax(raw, input, dim=None, _stacklevel=3):
    # for F.softmax
    x=raw(input, dim=dim)
    if dim is None:
        dim=F._get_softmax_dim('softmax', input.dim(), _stacklevel)
    name = log.add_layer(name='softmax')
    log.add_blobs([x], name='softmax_blob')
    layer = caffe_net.Layer_param(name=name, type='Softmax',
                                  bottom=[log.get_blobs(input)], top=[log.get_blobs(x)])
    layer.param.softmax_param.axis=dim
    log.cnet.add_layer(layer)
    return x

def _batch_norm(raw,input, running_mean, running_var, weight=None, bias=None,
               training=False, momentum=0.1, eps=1e-5):
    # because the runing_mean and runing_var will be changed after the _batch_norm operation, we first save the parameters

    x = raw(input, running_mean, running_var, weight, bias,
               training, momentum, eps)
    bottom_blobs = [log.get_blobs(input)]
    layer_name1 = log.add_layer(name='batch_norm')
    top_blobs = log.add_blobs([x], name='batch_norm_blob')
    layer1 = caffe_net.Layer_param(name=layer_name1, type='BatchNorm',
                                   bottom=bottom_blobs, top=top_blobs)
    if running_mean is None or running_var is None:
        # not use global_stats, normalization is performed over the current mini-batch
        layer1.batch_norm_param(use_global_stats=0,eps=eps)
    else:
        layer1.batch_norm_param(use_global_stats=1, eps=eps)
        running_mean_clone = running_mean.clone()
        running_var_clone = running_var.clone()
        layer1.add_data(running_mean_clone.cpu().numpy(), running_var_clone.cpu().numpy(), np.array([1.0]))
    log.cnet.add_layer(layer1)
    if weight is not None and bias is not None:
        layer_name2 = log.add_layer(name='bn_scale')
        layer2 = caffe_net.Layer_param(name=layer_name2, type='Scale',
                                       bottom=top_blobs, top=top_blobs)
        layer2.param.scale_param.bias_term = True
        layer2.add_data(weight.cpu().data.numpy(), bias.cpu().data.numpy())
        log.cnet.add_layer(layer2)
    return x

def _instance_norm(raw, input, running_mean=None, running_var=None, weight=None,
                  bias=None, use_input_stats=True, momentum=0.1, eps=1e-5):
    # TODO: the batch size!=1 view operations
    print("WARNING: The Instance Normalization transfers to Caffe using BatchNorm, so the batch size should be 1")
    if running_var is not None or weight is not None:
        # TODO: the affine=True or track_running_stats=True case
        raise NotImplementedError("not implement the affine=True or track_running_stats=True case InstanceNorm")
    x= torch.batch_norm(
        input, weight, bias, running_mean, running_var,
        use_input_stats, momentum, eps,torch.backends.cudnn.enabled)
    bottom_blobs = [log.get_blobs(input)]
    layer_name1 = log.add_layer(name='instance_norm')
    top_blobs = log.add_blobs([x], name='instance_norm_blob')
    layer1 = caffe_net.Layer_param(name=layer_name1, type='BatchNorm',
                                   bottom=bottom_blobs, top=top_blobs)
    if running_mean is None or running_var is None:
        # not use global_stats, normalization is performed over the current mini-batch
        layer1.batch_norm_param(use_global_stats=0,eps=eps)
        running_mean=torch.zeros(input.size()[1])
        running_var=torch.ones(input.size()[1])
    else:
        layer1.batch_norm_param(use_global_stats=1, eps=eps)
    running_mean_clone = running_mean.clone()
    running_var_clone = running_var.clone()
    layer1.add_data(running_mean_clone.cpu().numpy(), running_var_clone.cpu().numpy(), np.array([1.0]))
    log.cnet.add_layer(layer1)
    if weight is not None and bias is not None:
        layer_name2 = log.add_layer(name='bn_scale')
        layer2 = caffe_net.Layer_param(name=layer_name2, type='Scale',
                                       bottom=top_blobs, top=top_blobs)
        layer2.param.scale_param.bias_term = True
        layer2.add_data(weight.cpu().data.numpy(), bias.cpu().data.numpy())
        log.cnet.add_layer(layer2)
    return x


def op_placeholder(raw, *args, **kwargs):
    output = raw(*args, **kwargs)
    bottom_blobs=[]
    warning_string="======\nCRITICAL WARN: layer {} cannot be transfer, " \
          "because it cannot be implemented with original version of Caffe or it just is not implemented in nn_tools! \n" \
          "Nn_tools place a placeholder with Python type layer in Caffe. \n======".format(log.pytorch_layer_name)
    # print(warning_string)
    global WARNING_STRINGS
    WARNING_STRINGS+=warning_string
    for arg in args:
        if isinstance(arg,torch.Tensor):
            try:
                bottom_blobs.append(log.get_blobs(arg))
            except:
                print("WARN: at op_placehoder, tensor {} is not in the graph".format(arg))
    output_blobs=[]
    if isinstance(output,tuple):
        for out in output:
            output_blobs.append(out)
    else:
        output_blobs.append(output)
    top_blobs = log.add_blobs(output_blobs, name='op_placehoder_blob')
    layer_name = log.add_layer(name='op_placehoder')
    layer = caffe_net.Layer_param(name=layer_name, type='Python',
                                   bottom=bottom_blobs, top=top_blobs)
    log.cnet.add_layer(layer)
    return output

F_supported=[
    'conv2d',
    'linear',
    'relu',
    'leaky_relu',
    'max_pool2d',
    'avg_pool2d',
    'dropout',
    'threshold',
    'prelu',
    'batch_norm',
    'instance_norm',
    'softmax',
    'conv_transpose2d',
    #'interpolate',  # TODO, interpolate function cannot transfer correctly now

]

for op_name in F.__dict__:
    if op_name in F_supported:
        raw_func=getattr(F, op_name)
        transfer_func=globals()['_'+op_name]
        op_wrapper=Rp(raw_func,transfer_func)
        setattr(F, op_name, op_wrapper)
    else:
        if op_name[0]=='_' or op_name in ['division','warnings','math','torch','utils','vision','Col2Im','Im2Col','grad','weak_script','List']:
            continue
        setattr(F,op_name,Rp(getattr(F,op_name),op_placeholder))

# ----- for torch operations -----
def torch_max(raw,*args):
    x=raw(*args)
    if len(args)==1:
        # TODO max in one tensor
        assert NotImplementedError
    else:
        bottom_blobs=[]
        for arg in args:
            bottom_blobs.append(log.get_blobs(arg))
        layer_name=log.add_layer(name='max')
        top_blobs=log.add_blobs([x],name='max_blob')
        layer=caffe_net.Layer_param(name=layer_name,type='Eltwise',
                                    bottom=bottom_blobs,top=top_blobs)
        layer.param.eltwise_param.operation =2
        log.cnet.add_layer(layer)
    return x

def torch_cat(raw,inputs, dimension=0):
    x=raw(inputs, dimension)
    bottom_blobs=[]
    for input in inputs:
        bottom_blobs.append(log.get_blobs(input))
    layer_name=log.add_layer(name='cat')
    top_blobs=log.add_blobs([x],name='cat_blob')
    layer=caffe_net.Layer_param(name=layer_name,type='Concat',
                                bottom=bottom_blobs,top=top_blobs)
    layer.param.concat_param.axis =dimension
    log.cnet.add_layer(layer)
    return x

def torch_split(raw,tensor, split_size, dim=0):
    # split in pytorch is slice in caffe
    x=raw(tensor, split_size, dim)
    layer_name=log.add_layer('split')
    top_blobs=log.add_blobs(x,name='split_blob')
    layer=caffe_net.Layer_param(name=layer_name, type='Slice',
                                bottom=[log.get_blobs(tensor)], top=top_blobs)
    slice_num=int(np.floor(tensor.size()[dim]/split_size))
    slice_param=caffe_net.pb.SliceParameter(axis=dim,slice_point=[split_size*i for i in range(1,slice_num)])
    layer.param.slice_param.CopyFrom(slice_param)
    log.cnet.add_layer(layer)
    return x

def torch_add(raw,*args):
    x = raw(*args)
    if not NET_INITTED:
        return x
    layer_name = log.add_layer(name='add')
    top_blobs = log.add_blobs([x], name='add_blob')
    layer = caffe_net.Layer_param(name=layer_name, type='Eltwise',
                                  bottom=[log.get_blobs(input), log.get_blobs(args[0])], top=top_blobs)
    layer.param.eltwise_param.operation = 1  # sum is 1
    log.cnet.add_layer(layer)
    return x

def torch_sub(raw,*args):
    return ___sub__(*args)

def torch_mul(raw,*args):
    return ___mul__(*args)

def torch_div(raw,*args):
    return ___div__(*args)

def torch_pow(raw,*args):
    x = raw(*args)
    if not NET_INITTED:
        return x
    if not isinstance(args[0], int):
        raise NotImplementedError('power only support int now in nn_tools')
    layer_name = log.add_layer(name='power')
    top_blobs = log.add_blobs([x], name='power_blob')
    layer = caffe_net.Layer_param(name=layer_name, type='Power',
                                  bottom=[log.get_blobs(input)], top=top_blobs)
    layer.param.power_param.power = args[0]  # product is 1
    log.cnet.add_layer(layer)
    return x

def torch_sqrt(raw,*args):
    x = raw(*args)
    if not NET_INITTED:
        return x
    if not isinstance(args[0], int):
        raise NotImplementedError('sqrt only support int now in nn_tools')
    layer_name = log.add_layer(name='sqrt')
    top_blobs = log.add_blobs([x], name='sqrt_blob')
    layer = caffe_net.Layer_param(name=layer_name, type='Power',
                                  bottom=[log.get_blobs(input)], top=top_blobs)
    layer.param.power_param.power = 0.5
    log.cnet.add_layer(layer)
    return x

torch_op_supported=[
    'split',
    'max',
    'cat',
    # 'add',
    # 'sub',
    # 'mul',
    # 'div',
    # 'pow',
    # 'sqrt',
]

for op_name in torch_op_supported:
    raw_op = getattr(torch, op_name)
    op_wrapper=Rp(raw_op,globals()['torch_'+op_name])
    setattr(torch, op_name, op_wrapper)

# ----- for Variable/torch.Tensor operations --------

def _view(input, *args):
    x=raw_tensor_magic_op['view'](input, *args)
    if not NET_INITTED:
        return x
    layer_name=log.add_layer(name='view')
    top_blobs=log.add_blobs([x],name='view_blob')
    layer=caffe_net.Layer_param(name=layer_name, type='Reshape',
                                bottom=[log.get_blobs(input)], top=top_blobs)
    # TODO: reshpae added to nn_tools layer
    dims=list(args)
    dims[0]=0 # the first dim should be batch_size
    layer.param.reshape_param.shape.CopyFrom(caffe_net.pb.BlobShape(dim=dims))
    log.cnet.add_layer(layer)
    return x

def _mean(input, *args,**kwargs):
    x=raw_tensor_magic_op['mean'](input, *args, **kwargs)
    if not NET_INITTED:
        return x
    layer_name=log.add_layer(name='mean')
    top_blobs=log.add_blobs([x],name='mean_blob')
    layer=caffe_net.Layer_param(name=layer_name, type='Reduction',
                                bottom=[log.get_blobs(input)], top=top_blobs)
    if len(args)==1:
        dim=args[0]
    elif 'dim' in kwargs:
        dim=kwargs['dim']
    else:
        raise NotImplementedError('mean operation must specify a dim')
    if dim!=len(input.size())-1:
        raise NotImplementedError('mean in Caffe Reduction Layer: only reduction along ALL "tail" axes is supported')
    if kwargs.get('keepdim'):
        raise NotImplementedError('mean operation must keep_dim=False')
    layer.param.reduction_param.operation=4
    layer.param.reduction_param.axis=dim
    log.cnet.add_layer(layer)
    return x

def _sum(input, *args,**kwargs):
    x=raw_tensor_magic_op['sum'](input, *args, **kwargs)
    if not NET_INITTED:
        return x
    layer_name=log.add_layer(name='sum')
    top_blobs=log.add_blobs([x],name='sum_blob')
    layer=caffe_net.Layer_param(name=layer_name, type='Reduction',
                                bottom=[log.get_blobs(input)], top=top_blobs)
    if len(args)==1:
        dim=args[0]
    elif 'dim' in kwargs:
        dim=kwargs['dim']
    else:
        raise NotImplementedError('sum operation must specify a dim')
    if dim!=len(input.size())-1:
        raise NotImplementedError('sum in Caffe Reduction Layer: only reduction along ALL "tail" axes is supported')
    if kwargs.get('keepdim'):
        raise NotImplementedError('sum operation must keep_dim=False')
    layer.param.reduction_param.operation=1 # operation 1 for sum
    layer.param.reduction_param.axis=dim
    log.cnet.add_layer(layer)
    return x

def _add(input,*args):
    return ___add__(input, *args)

def _sub(input,*args):
    return ___sub__(input, *args)

def _mul(input,*args):
    return ___mul__(input, *args)

def _div(input,*args):
    return ___div__(input, *args)

def _pow(input,*args):
    return ___pow__(input, *args)

def _sqrt(input, *args):
    x = raw_tensor_magic_op['sqrt'](input, *args)
    if not NET_INITTED:
        return x
    layer_name = log.add_layer(name='sqrt')
    top_blobs = log.add_blobs([x], name='sqrt_blob')
    layer = caffe_net.Layer_param(name=layer_name, type='Power',
                                  bottom=[log.get_blobs(input)], top=top_blobs)
    layer.param.power_param.power = 0.5
    log.cnet.add_layer(layer)
    return x

def ___add__(input, *args):
    x = raw_tensor_magic_op['__add__'](input, *args)
    if not NET_INITTED:
        return x
    layer_name = log.add_layer(name='add')
    top_blobs = log.add_blobs([x], name='add_blob')
    if not isinstance(args[0],torch.Tensor):
        layer = caffe_net.Layer_param(name=layer_name, type='Power',
                                      bottom=[log.get_blobs(input)], top=top_blobs)
        layer.param.power_param.shift = args[0]
    else:
        layer = caffe_net.Layer_param(name=layer_name, type='Eltwise',
                                      bottom=[log.get_blobs(input), log.get_blobs(args[0])], top=top_blobs)
        layer.param.eltwise_param.operation = 1 # sum is 1
    log.cnet.add_layer(layer)
    return x

def ___iadd__(input, *args):
    x = raw_tensor_magic_op['__iadd__'](input, *args)
    if not NET_INITTED:
        return x
    x=x.clone()
    layer_name = log.add_layer(name='add')
    top_blobs = log.add_blobs([x], name='add_blob')
    if not isinstance(args[0], torch.Tensor):
        layer = caffe_net.Layer_param(name=layer_name, type='Power',
                                      bottom=[log.get_blobs(input)], top=top_blobs)
        layer.param.power_param.shift = args[0]
    else:
        layer = caffe_net.Layer_param(name=layer_name, type='Eltwise',
                                      bottom=[log.get_blobs(input), log.get_blobs(args[0])], top=top_blobs)
        layer.param.eltwise_param.operation = 1  # sum is 1
    return x

def ___sub__(input, *args):
    x = raw_tensor_magic_op['__sub__'](input, *args)
    if not NET_INITTED:
        return x
    layer_name = log.add_layer(name='sub')
    top_blobs = log.add_blobs([x], name='sub_blob')
    layer = caffe_net.Layer_param(name=layer_name, type='Eltwise',
                                  bottom=[log.get_blobs(input), log.get_blobs(args[0])], top=top_blobs)
    layer.param.eltwise_param.operation = 1 # sum is 1
    layer.param.eltwise_param.coeff.extend([1.,-1.])
    log.cnet.add_layer(layer)
    return x

def ___isub__(input, *args):
    x = raw_tensor_magic_op['__isub__'](input, *args)
    if not NET_INITTED:
        return x
    x=x.clone()
    layer_name = log.add_layer(name='sub')
    top_blobs = log.add_blobs([x], name='sub_blob')
    layer = caffe_net.Layer_param(name=layer_name, type='Eltwise',
                                  bottom=[log.get_blobs(input), log.get_blobs(args[0])], top=top_blobs)
    layer.param.eltwise_param.operation = 1 # sum is 1
    log.cnet.add_layer(layer)
    return x

def ___mul__(input, *args):
    x = raw_tensor_magic_op['__mul__'](input, *args)
    if not NET_INITTED:
        return x
    layer_name = log.add_layer(name='mul')
    top_blobs = log.add_blobs([x], name='mul_blob')
    layer = caffe_net.Layer_param(name=layer_name, type='Eltwise',
                                  bottom=[log.get_blobs(input), log.get_blobs(args[0])], top=top_blobs)
    layer.param.eltwise_param.operation = 0  # product is 1
    log.cnet.add_layer(layer)
    return x

def ___imul__(input, *args):
    x = raw_tensor_magic_op['__imul__'](input, *args)
    if not NET_INITTED:
        return x
    x = x.clone()
    layer_name = log.add_layer(name='mul')
    top_blobs = log.add_blobs([x], name='mul_blob')
    layer = caffe_net.Layer_param(name=layer_name, type='Eltwise',
                                  bottom=[log.get_blobs(input), log.get_blobs(args[0])], top=top_blobs)
    layer.param.eltwise_param.operation = 0  # product is 1
    layer.param.eltwise_param.coeff.extend([1., -1.])
    log.cnet.add_layer(layer)
    return x

def ___div__(input, *args):
    x = raw_tensor_magic_op['__div__'](input, *args)
    if not NET_INITTED:
        return x
    if not isinstance(args[0],torch.Tensor):
        layer_name = log.add_layer(name='div')
        top_blobs = log.add_blobs([x], name='div_blob')
        layer = caffe_net.Layer_param(name=layer_name, type='Power',
                                      bottom=[log.get_blobs(input)], top=top_blobs)
        layer.param.power_param.scale = 1/args[0]
        log.cnet.add_layer(layer)
    else:
        pre_layer_name=log.add_layer(name='pre_div')
        pre_div_blobs = log.add_blobs([x], name='pre_div_blob')
        pre_layer = caffe_net.Layer_param(name=pre_layer_name, type='Power',
                                      bottom=[log.get_blobs(input)], top=pre_div_blobs)
        pre_layer.param.power_param.power=-1
        pre_layer.param.power_param.shift = 1e-6
        log.cnet.add_layer(pre_layer)
        layer_name = log.add_layer(name='div')
        top_blobs = log.add_blobs([x], name='div_blob')
        layer = caffe_net.Layer_param(name=layer_name, type='Eltwise',
                                      bottom=[pre_div_blobs[0], log.get_blobs(args[0])], top=top_blobs)
        layer.param.eltwise_param.operation = 0  # product is 1
        log.cnet.add_layer(layer)
    return x

def ___pow__(input, *args):
    x = raw_tensor_magic_op['__pow__'](input, *args)
    if not NET_INITTED:
        return x
    if not isinstance(args[0],int):
        raise NotImplementedError('power only support int now in nn_tools')
    layer_name = log.add_layer(name='power')
    top_blobs = log.add_blobs([x], name='power_blob')
    layer = caffe_net.Layer_param(name=layer_name, type='Power',
                                  bottom=[log.get_blobs(input)], top=top_blobs)
    layer.param.power_param.power = args[0]  # product is 1
    log.cnet.add_layer(layer)
    return x

# TODO: other types of the view function

tensor_op_supported=[]

tensor_magic_op_supported=[
    'view',
    'mean',
    'add',
    'sub',
    'mul',
    'div',
    'pow',
    'sqrt',
    'sum',
    '__add__',
    '__iadd__',
    '__sub__',
    '__isub__',
    '__mul__',
    '__imul__',
    '__div__',
    '__pow__',
]

raw_tensor_magic_op={}
if hasattr(Variable,'__add__'):
    tensor_target=Variable
else:
    # for new version >=0.4.0
    tensor_target=torch.Tensor

for op_name in tensor_magic_op_supported:
    raw_op=getattr(tensor_target,op_name)
    raw_tensor_magic_op[op_name]=raw_op
    setattr(tensor_target,op_name,globals()['_'+op_name])

for op_name in tensor_op_supported:
    raw_op = getattr(tensor_target, op_name)
    op_wrapper = Rp(raw_op, globals()['_' + op_name])
    setattr(tensor_target, op_name, op_wrapper)



def trans_net(net,input_var,name='TransferedPytorchModel'):
    print('Starting Transform, This will take a while')
    log.init([input_var])
    log.cnet.net.name=name
    log.cnet.net.input.extend([log.get_blobs(input_var)])
    log.cnet.net.input_dim.extend(input_var.size())
    global NET_INITTED
    NET_INITTED=True
    for name,layer in net.named_modules():
        layer_names[layer]=name
    out = net.forward(input_var)
    print('Transform Completed')
    print(WARNING_STRINGS)

def save_prototxt(save_name):
    log.cnet.save_prototxt(save_name)

def save_caffemodel(save_name):
    log.cnet.save(save_name)
