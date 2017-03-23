import nn_tools.Caffe.caffe_pb2 as pb
import numpy as np

class Layer_param():
    def __init__(self,name='',type='',top=(),bottom=()):
        self.param=pb.LayerParameter()
        self.name=self.param.name=name
        self.type=self.param.type=type

        self.top=self.param.top
        self.top.extend(top)
        self.bottom=self.param.bottom
        self.bottom.extend(bottom)

    def conv_param(self, num_output, kernel_size, stride=(1), weight_filler_type='xavier', bias_filler_type='constant'):
        """
        add a conv_param layer if you spec the layer type "Convolution"
        Args:
            num_output: a int
            kernel_size: int list
            stride: a int list
            weight_filler_type: the weight filer type
            bias_filler_type: the bias filler type

        Returns:

        """
        if self.type!='Convolution':
            raise TypeError,'the layer type must be Convolution if you want set conv param'
        conv_param=pb.ConvolutionParameter()
        conv_param.num_output=num_output
        conv_param.kernel_size.extend(kernel_size)
        conv_param.stride.extend(stride)
        conv_param.weight_filler.type=weight_filler_type
        conv_param.bias_filler.type = bias_filler_type
        self.param.convolution_param.CopyFrom(conv_param)

    def fc_param(self,num_output,weight_filler='xavier',bias_filler='constant'):
        if self.type!='InnerProduct':
            raise TypeError,'the layer type must be InnerProduct if you want set fc param'
        fc_param=pb.InnerProductParameter()
        fc_param.num_output=num_output
        fc_param.weight_filler.type = weight_filler
        fc_param.bias_filler.type = bias_filler
        self.param.inner_product_param.CopyFrom(fc_param)


    def set_params_by_dict(self,dic):
        pass

    def copy_from(self,layer_param):
        pass