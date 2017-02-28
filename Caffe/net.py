import nn_tools.Caffe.caffe_pb2 as pb
import google.protobuf.text_format as text_format
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

class Net():
    def __init__(self, file_name='',type='caffemodel'):
        # caffe_model dir
        self.net = pb.NetParameter()
        if file_name!='':
            f = open(file_name,'a+')
            if type=='caffemodel':
                self.net.ParseFromString(f.read())
            elif type=='prototxt':
                text_format.Parse(f.read(), self.net)
            else:
                raise TypeError,'type must be caffemodel or prototxt'
            f.close()

    def layer(self, layer_name):
        """
        :param file_name:  prototxt file name
        :return: a Message object of the layer
        """
        for layer in self.net.layer:
            if layer.name == layer_name:
                return layer
        raise AttributeError, "cannot found layer %s" % str(layer_name)

    def save_prototxt(self,path):
        prototxt=pb.NetParameter()
        prototxt.CopyFrom(self.net)
        for layer in prototxt.layer:
            del layer.blobs[:]
        with open(path,'w') as f:
            f.write(text_format.MessageToString(prototxt))

    def save(self, path, type='caffemodel'):
        if type=='prototxt':
            self.save_prototxt(path)
        with open(path,'wb') as f:
            f.write(self.net.SerializeToString())

    def remove_layer(self,layer_name):
        for i,layer in enumerate(self.net.layer):
            if layer.name == layer_name:
                del self.net.layer[i]
                return
        raise AttributeError, "cannot found layer %s" % str(layer_name)

    def layer_index(self,layer_name):
        """
        find a layer's index by name
        Args:
            layer_name: string for layer
        Returns:
            if the layer was found, the layer position in the net was returned.
            if there is no layer found, return -1
        """
        for i, layer in enumerate(self.net.layer):
            if layer.name == layer_name:
                return i

    def add_layer(self,layer_params,before='',after=''):

        # find the before of after layer's position
        index = -1
        if after != '':
            index = self.layer_index(after) + 1
        if before != '':
            index = self.layer_index(before)
        new_layer = pb.LayerParameter()
        new_layer.CopyFrom(layer_params.param)
        #insert the layer into the layer protolist
        if index != -1:
            self.net.layer.add()
            for i in range(len(self.net.layer) - 1, index, -1):
                self.net.layer[i].CopyFrom(self.net.layer[i - 1])
            self.net.layer[index].CopyFrom(new_layer)
        else:
            self.net.layer.extend([new_layer])

    def add_layer_with_data(self,layer_params,datas, before='', after=''):
        """
        Args:
            layer_params:A Layer_Param object
            datas:a fixed dimension numpy object list
            after: put the layer after a specified layer
            before: put the layer before a specified layer
        """
        self.add_layer(layer_params,before,after)
        new_layer =self.layer(layer_params.name)

        #process blobs
        del new_layer.blobs[:]
        for data in datas:
            new_blob=new_layer.blobs.add()
            for dim in data.shape:
                new_blob.shape.dim.append(dim)
            new_blob.data.extend(data.flatten().astype(float))

    def get_layer_data(self,layer_name):
        layer=self.layer(layer_name)
        datas=[]
        for blob in layer.blobs:
            shape=list(blob.shape.dim)
            data=np.array(blob.data).reshape(shape)
            datas.append(data)
        return datas

    def set_layer_data(self,layer_name,datas):
        # datas is a list of [weights,bias]
        layer=self.layer(layer_name)
        for blob,data in zip(layer.blobs,datas):
            blob.data[:]=data.flatten()
            pass

class Model():
    def __init__(self,prototxt_dir,caffemode_dir):
        import caffe
        self.net=caffe.Net(prototxt_dir,caffemode_dir,caffe.TEST)

    def layer(self,layer_name):
        for i,layer in enumerate(self.net.layers):
            if layer_name == self.net._layer_names[i]:
                return layer

    def fill_blobs(self,pb_net):
        """
        fill the caffe model blobs by a protobuffer type Net
        Args:
            pb_net: Net
        """
        for i, layer in enumerate(self.net.layers):
            datas=pb_net.get_layer_data(self.net._layer_names[i])
            if len(datas)!=len(datas.blobs):
                raise IndexError, 'The number of blobs in protobuffer is different from that in caffemodel'
            for j,data in enumerate(datas):
                layer.blobs[j]=data