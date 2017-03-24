import numpy as np
from blob import Blob
import csv,pprint

box=[]
class Base(object):
    def __init__(self,input,name=''):
        if isinstance(input,Base):
            input=input()
            assert isinstance(input,Blob),'The input of layer %s is not Blob, please use nn_tools.P.blob.Blob as input'%name
        self.name=name
        self.input=input
        self.weight_size=0
        self.blob_size=None
        self.input_size=np.prod(self.input.data)
        self.dot=0
        self.add=0
        self.pow=0
        self.compare=0
        self.flops=0
        self.out=None
        self.layer_info=None
        box.append(self)

    def __call__(self, *args, **kwargs):
        return self.out
    def __setattr__(self, key, value):
        if key=='out' and value!=None:
            self.blob_size=np.prod(value.data.shape)
        return object.__setattr__(self, key,value)
    def __getattribute__(self, item):
        if item=='flops':
            self.flops=self.pow+self.add+self.dot+self.compare
        return object.__getattribute__(self,item)

class Norm(Base):
    valid_tuple=('norm')
    def __init__(self,input,type,name=None):
        if type not in Norm.valid_tuple:
            raise NameError('the norm type:' + type + ' is not supported. ' \
                             'the valid type is: ' + str(Activation.valid_tuple))
        if name == None: name = type
        Base.__init__(self, input, name=name)
        getattr(self, type)()
        self.out = self.input.new(self)

    def norm(self):
        self.dot = self.input_size
        self.add = self.input_size

class Activation(Base):
    #valid tuple lists the valid activation function type
    valid_tuple=('relu','tanh','prelu')
    def __init__(self,input,type,name=None):
        if type not in Activation.valid_tuple:
            raise NameError('the activation type:'+type+' is not supported. ' \
                            'the valid type is: '+str(Activation.valid_tuple))
        if name==None:name=type
        Base.__init__(self,input,name=name)
        getattr(self,type)()
        self.out=self.input.new(self)

    def relu(self):
        self.compare=self.input_size

    def sigmoid(self):
        self.add=self.dot=self.pow=self.input_size

    def tanh(self):
        self.dot=self.input_size
        self.add=self.pow=self.input_size*2


class Sliding(Base):
    def __init__(self,input,kernel_size,num_out,stride=1,pad=0,name='sliding',ceil=False):
        # input (w,h,c)
        super(Sliding,self).__init__(input,name=name)
        if type(kernel_size)==int:
            self.kernel_size=[kernel_size,kernel_size]
        else:
            self.kernel_size=[i for i in kernel_size]
            if len(self.kernel_size)==1:self.kernel_size*=2
        if type(stride)==int:
            self.stride=[stride,stride]
        else:
            self.stride=[i for i in stride]
            if len(self.stride)==1:self.stride*=2
            elif len(self.stride)==0:self.stride=[1,1]
            elif len(self.stride)>2:raise AttributeError
        if type(pad)==int:
            self.pad=[pad,pad]
        else:
            self.pad=[i for i in pad]
            if len(self.pad)==1:self.pad*=2
            elif len(self.pad)==0:self.pad=[0,0]
            elif len(self.pad)>2:raise AttributeError
        self.num_out=num_out
        self.layer_info='kernel=%dx%d,stride=%dx%d,pad=%dx%d'%(self.kernel_size[0],self.kernel_size[1],
                                                            self.stride[0],self.stride[1],self.pad[0],self.pad[1])
        #calc out
        if not ceil:
            out_w=np.floor(float(self.input[0]+self.pad[0]*2-self.kernel_size[0])/self.stride[0])+1
            out_h=np.floor(float(self.input[1]+self.pad[1]*2-self.kernel_size[1])/self.stride[1])+1
        else:
            out_w = np.ceil(float(self.input[0] + self.pad[0] * 2 - self.kernel_size[0]) / self.stride[0]) + 1
            out_h = np.ceil(float(self.input[1] + self.pad[1] * 2 - self.kernel_size[1]) / self.stride[1]) + 1
        self.out=Blob([out_w,out_h,num_out],self)

class Conv(Sliding):
    def __init__(self,input,kernel_size,num_out,stride=1,pad=0,activation='relu',name='conv',ceil=False):
        if isinstance(input,Base):
            input=input()
        Sliding.__init__(self,input,kernel_size,num_out,stride,pad,name=name,ceil=ceil)
        self.layer_info+=',num_out=%d'%(num_out)
        self.dot = self.out[0] * self.out[1] * self.input[2] * np.prod(self.kernel_size) * self.num_out
        self.add = self.dot
        self.weight_size=np.prod(self.kernel_size)*num_out*input[2]
        if activation:
            Activation(self.out,activation)
conv=Conv

class Pool(Sliding):
    def __init__(self,input,kernel_size,stride=1,pad=0,name='pool',pool_type='max',ceil=False):
        if isinstance(input,Base):
            input=input()
        Sliding.__init__(self,input,kernel_size,input[2],stride,pad,name=name,ceil=ceil)
        self.pool_type=pool_type
        self.layer_info+=',type=%s'%(pool_type)
        self.compare=self.out[0]*self.out[1]*(np.prod(self.kernel_size)-1)*self.num_out
pool=Pool

class InnerProduct(Base):
    def __init__(self,input,num_out,activation='relu',name='innerproduct'):
        if isinstance(input,Base):
            input=input()
        Base.__init__(self,input,name=name)
        self.num_out=num_out
        self.dot=self.num_out
        self.add=self.num_out*self.input_size
        self.out=Blob([self.num_out],self)
        self.weight_size = self.num_out * self.input_size
        if activation:
            Activation(self.out,activation)
Fc=InnerProduct
fc=InnerProduct

class Permute(Base):
    def __init__(self, input,dims, name='permute'):
        super(Permute,self).__init__(input,name)
        self.out = Blob(dims,self)

class Flatten(Base):
    def __init__(self,input, name='permute'):
        super(Flatten, self).__init__(input, name)
        dim=[np.prod(input.data.shape)]
        self.out = Blob(dim, self)

def save_csv(blobs,csv_save_path,save_items=('name', 'layer_info', 'input', 'out', 'dot', 'add', 'compare','flops', 'weight_size','blob_size')):
    layers = get_layer_blox_from_blobs(blobs)
    print_list = []
    for layer in layers:
        print_list.append([str(getattr(layer, param)) for param in save_items])
    if csv_save_path!=None:
        with open(csv_save_path,'w') as file:
            writer=csv.writer(file)
            writer.writerow(save_items)
            for layer in print_list:
                writer.writerow(layer)
    pprint.pprint(print_list,depth=3,width=200)
    print 'saved!'

def get_layer_blox_from_blobs(blobs):
    layers=[]
    def creator_search(blob):
        for father in blob.father:
            if isinstance(father,Base) and father not in layers:
                layers.append(father)
                creator_search(father.input)
    for blob in blobs:
        creator_search(blob)
    return layers

def print_by_blob(blobs,print_items=('name', 'layer_info', 'input', 'out', 'dot', 'add', 'compare','flops', 'weight_size','blob_size')):
    layers=get_layer_blox_from_blobs(blobs)
    print_list = []
    for layer in layers:
        print_list.append([str(getattr(layer, param)) for param in print_items])
    pprint.pprint(print_list, depth=3, width=200)
    return print_list
