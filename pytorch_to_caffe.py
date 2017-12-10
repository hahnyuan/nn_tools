from __future__ import absolute_import
from collections import OrderedDict
import numpy as np
from .Caffe import caffe_net
from .Caffe.layer_param import set_enum
# import caffe

layer_dict = {'ConvNdBackward': 'Convolution',
              'ThresholdBackward': 'ReLU',
              'MaxPool2dBackward': 'Pooling',
              'AvgPool2dBackward': 'Pooling',
              'DropoutBackward': 'Dropout',
              'AddmmBackward': 'InnerProduct',
              'BatchNormBackward': 'BatchNorm',
              'AddBackward': 'Eltwise',
              'ViewBackward': 'Reshape',
              'ConcatBackward': 'Concat',
              'UpsamplingNearest2d': 'Deconvolution',
              'UpsamplingBilinear2d': 'Deconvolution',
              'SigmoidBackward': 'Sigmoid',
              'LeakyReLUBackward': 'ReLU',
              'NegateBackward': 'Power',
              'MulBackward': 'Eltwise',
              'SpatialCrossMapLRNFunc': 'LRN'}

layer_id=0

def trans_to_protobuf(net, input_var, output_var, name):
    net.net.name=name
    net.net.input.extend(['data'])
    for size in input_var.size():
        net.net.input_dim.extend([size])
    global layer_id
    layer_id=1
    seen = set()
    top_names = dict()
    def add_layer(func):
        global layer_id
        parent_type = str(type(func).__name__)
        parent_bottoms = []
        if hasattr(func, 'next_functions'):
            for u in func.next_functions:
                # Generating DAG
                if u[0] is not None:
                    child_type = str(type(u[0]).__name__)
                    if child_type != 'AccumulateGrad' and (
                            parent_type != 'AddmmBackward' or child_type != 'TransposeBackward'):
                        if u[0] not in seen:
                            top_name = add_layer(u[0])
                            parent_bottoms.append(top_name)
                            seen.add(u[0])
                        else:
                            top_name = top_names[u[0]]
                            parent_bottoms.append(top_name)
                        if child_type != 'ViewBackward':
                            # Ignore the View Backward
                            # TODO: view
                            layer_id = layer_id + 1

        parent_name = layer_dict[parent_type] + str(layer_id)

        if parent_type not in layer_dict.keys():
            raise(NotImplementedError,'The layer is not implemented yet: %s'%parent_type)

        parent_top = parent_name
        if layer_id > 1:
            bottom = parent_bottoms
        else:
            bottom = ['data']
        top_names[func]=parent_name
        # initial a layer
        layer_param=caffe_net.Layer_param(name=parent_name,type=layer_dict[parent_type],
                                          top=[parent_name],bottom=bottom)
        if parent_type == 'MulBackward':
            # Element Wise Prod Layer
            param=caffe_net.pb.EltwiseParameter()
            set_enum(param,'operation','PROD')
            layer_param.param.eltwise_param.CopyFrom(param)
        elif parent_type == 'AddBackward':
            # Element Wise Sum Layer
            param = caffe_net.pb.EltwiseParameter()
            param.operation=param.EltwiseOp.Value('SUM')
            layer_param.param.eltwise_param.CopyFrom(param)
        elif parent_type == 'NegateBackward':
            # Power Layer to Negative the data
            param=caffe_net.pb.PowerParameter()
            param.power=1
            param.scale=-1
            param.shift=0
            layer_param.param.power_param.CopyFrom(param)
        elif parent_type == 'LeakyReLUBackward':
            # Leaky Relu
            negative_slope = func.additional_args[0]
            param=caffe_net.pb.ReLUParameter()
            param.negative_slope=negative_slope
            layer_param.param.power_param.CopyFrom(param)
        elif parent_type == 'UpsamplingNearest2d':
            # Deconvolution Layer to Apply Nearest Upsampling
            pass
            # TODO: UpsamplingNearest2d
        elif parent_type == 'UpsamplingBilinear2d':
            # Deconvolution Layer to Apply Nearest Upsampling
            pass
            # TODO: UpsamplingBilinear2d
        elif parent_type == 'ConcatBackward':
            # Concat Layer
            param=caffe_net.pb.ConcatParameter()
            param.axis=func.dim
            layer_param.param.concat_param.CopyFrom(param)
        elif parent_type == 'ConvNdBackward':
            # Convolution Layer
            if func.transposed is True and func.next_functions[1][0] is None:
                # TODO: UpsamplingCaffe
                pass
            else:
                weights = func.next_functions[1][0].variable
                layer_param.conv_param(num_output=weights.size(0),
                                       kernel_size=(weights.size(2),weights.size(3)),
                                       stride=(func.stride[0],),
                                       pad=(func.padding[0],func.padding[1]),
                                       bias_term=False if func.next_functions[2][0] is None else True,
                                       dilation=(func.dilation[0],)
                                       )
                if func.next_functions[2][0]:
                    layer_param.add_data(func.next_functions[1][0].variable.data.numpy(),
                                          func.next_functions[2][0].variable.data.numpy())
                else:
                    layer_param.add_data(func.next_functions[1][0].variable.data.numpy())

        elif parent_type == 'BatchNormBackward':
            # BatchNorm Layer
            param=caffe_net.pb.BatchNormParameter()
            param.use_global_stats=True
            param.eps=func.eps
            layer_param.param.batch_norm_param.CopyFrom(param)
            layer_param.add_data(func.running_mean.numpy(),func.running_var.numpy(),np.array([1.0]))
            # Caffe Implement BatchNorm = BatchNorm + Affine
            if func.next_functions[1][0] is not None:
                net.add_layer(layer_params=layer_param)
                layer_param = caffe_net.Layer_param(name=parent_name+'_Scale', type='Scale',
                                                    top=[parent_name], bottom=[parent_name])
                param=caffe_net.pb.ScaleParameter()
                param.bias_term=True
                layer_param.param.scale_param.CopyFrom(param)
                layer_param.add_data(func.next_functions[1][0].variable.data.numpy(),
                                      func.next_functions[2][0].variable.data.numpy())

        elif parent_type == 'ThresholdBackward':
            # ReLU, no parameters
            pass
        elif parent_type == 'MaxPool2dBackward':
            # Max pooling
            layer_param.pool_param(type='MAX',
                                   kernel_size=func.kernel_size[0],
                                   stride=func.stride[0],
                                   pad=func.padding[0],
                                   )
        elif parent_type == 'AvgPool2dBackward':
            # Average pooling
            layer_param.pool_param(type='AVE',
                                   kernel_size=func.kernel_size[0],
                                   stride=func.stride[0],
                                   pad=func.padding[0],
                                   )
        elif parent_type == 'DropoutBackward':
            # Dropout Layer
            param=caffe_net.pb.DropoutParameter()
            param.dropout_ratio=func.p
            layer_param.param.dropout_param.CopyFrom(param)
        elif parent_type == 'AddmmBackward':
            # Inner product
            layer_param.fc_param(num_output=func.next_functions[0][0].variable.size(0))
            layer_param.add_data(func.next_functions[2][0].next_functions[0][0].variable.data.numpy(),
                                 func.next_functions[0][0].variable.data.numpy())
        elif parent_type == 'ViewBackward':
            # Ignore the View
            # TODO: View
            parent_top = parent_bottoms[0]
            return parent_top
        elif parent_type == 'SpatialCrossMapLRNFunc':
            # LRN Layer
            param=caffe_net.pb.LRNParameter()
            param.local_size=func.size
            param.alpha=func.alpha
            param.beta=func.beta
            layer_param.param.lrn_param.CopyFrom(param)

        net.add_layer(layer_params=layer_param)
        return parent_top

    add_layer(output_var.grad_fn)
    return net

def pytorch_to_caffe(input_var, output_var, prototxt, caffemodel, name='no_name'):
    """
    :param input_var: net input Variable
    :param output_var: net output Variable
    :param prototxt: file name to save the prototxt
    :param caffemodel: file name to save the caffemodel
    """
    print("starting to transfrom net %s"%name)
    net = caffe_net.Caffemodel('')
    trans_to_protobuf(net, input_var, output_var, name)
    print('save prototxt to %s' % prototxt)
    net.save_prototxt(prototxt)
    print('save caffemodel to %s' % caffemodel)
    net.save(caffemodel)
