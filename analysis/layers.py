import numpy as np
from .blob import Blob


box=[]

class Base(object):
    def __init__(self,input,name=''):
        def transfer_input(_input):
            if isinstance(_input,Base):
                _input=_input()
                assert isinstance(input,Blob),'The input of layer %s is not Blob, please use nn_tools.P.blob.Blob as input'%name
            return _input
        if type(input)==list:
            # if multi input
            self.input=[transfer_input(i) for i in input]
            self.input_size = np.sum([np.prod(i.shape) for i in self.input])
            self.muti_input=True
        else:
            self.input = transfer_input(input)
            self.input_size = np.prod(self.input.shape)
            self.muti_input = False
        self.name=name
        self.weight_size=0
        self.blob_size=None
        self.dot=0
        self.add=0
        self.pow=0
        self.compare=0
        self.ops=0
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
        if item=='ops':
            self.ops=self.pow+self.add+self.dot+self.compare
        return object.__getattribute__(self,item)

class Norm(Base):
    valid_tuple=('norm','batch_norm','lrn')
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

    def batch_norm(self):
        self.dot = self.input_size
        self.add = self.input_size

    def lrn(self):
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
        # input is the instance of blob.Blob with shape (h,w,c) or (batch,h,w,c)
        super(Sliding,self).__init__(input,name=name)
        if len(self.input.shape)==3:
            self.input_w=self.input[0]
            self.input_h=self.input[1]
            self.batch_size=1
            self.in_channel =self.input[2]
        elif len(self.input.shape)==4:
            self.input_w = self.input[1]
            self.input_h = self.input[2]
            self.batch_size = self.input[0]
            self.in_channel = self.input[3]
        else:
            raise ValueError('Sliding must have a input with (w,h,c) or (batch,w,h,c)')


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
            out_w=np.floor(float(self.input_w +self.pad[0]*2-self.kernel_size[0])/self.stride[0])+1
            out_h= np.floor(float(self.input_h + self.pad[1] * 2 - self.kernel_size[1]) / self.stride[1]) + 1
        else:
            out_w = np.ceil(float(self.input_w + self.pad[0] * 2 - self.kernel_size[0]) / self.stride[0]) + 1
            out_h = np.ceil(float(self.input_h + self.pad[1] * 2 - self.kernel_size[1]) / self.stride[1]) + 1
        self.out=Blob([self.batch_size,out_w,out_h,num_out],self)

class Conv(Sliding):
    def __init__(self,input,kernel_size,num_out,stride=1,pad=0,
                 activation='relu',name='conv',ceil=False,group_size=1):
        if isinstance(input,Base):
            input=input()
        Sliding.__init__(self,input,kernel_size,num_out,stride,pad,name=name,ceil=ceil)
        self.layer_info+=',num_out=%d'%(num_out)
        self.dot = np.prod(self.out.shape) * np.prod(self.kernel_size) * self.in_channel
        self.weight_size = np.prod(self.kernel_size) * num_out * self.in_channel
        if group_size!=1:
            self.layer_info += ',group_size=%d' % (group_size)
            self.dot /= group_size
            self.weight_size /= group_size
        self.add = self.dot
        if activation:
            Activation(self.out,activation)
conv=Conv

class Pool(Sliding):
    def __init__(self,input,kernel_size,stride=1,pad=0,name='pool',pool_type='max',ceil=False):
        if isinstance(input,Base):
            input=input()
        Sliding.__init__(self,input,kernel_size,input[3],stride,pad,name=name,ceil=ceil)
        self.pool_type=pool_type
        self.layer_info+=',type=%s'%(pool_type)
        self.compare= np.prod(self.out.shape) * (np.prod(self.kernel_size) - 1)
pool=Pool

class InnerProduct(Base):
    def __init__(self,input,num_out,activation='relu',name='innerproduct'):
        if isinstance(input,Base):
            input=input()
        Base.__init__(self,input,name=name)
        self.left_dim=np.prod(input.shape[1:])
        self.num_out=num_out
        self.dot=self.num_out*self.input_size
        self.add=self.num_out*self.input_size
        self.out=Blob([self.num_out],self)
        self.weight_size = self.num_out * self.left_dim
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

class Eltwise(Base):
    def __init__(self,inputs,type='sum',name='eltwise'):
        super(Eltwise,self).__init__(inputs,name,)
        self.out=inputs[0].new(self)
        if type in ['sum','SUM']:
            self.add=np.prod(self.out.shape)
        elif type in ['product','PROD']:
            self.dot=np.prod(self.out.shape)
        elif type in ['max','MAX']:
            self.compare=np.prod(self.out.shape)
        else:
            raise AttributeError('the Eltwise layer type must be sum, max or product')

class Concat(Base):
    def __init__(self,inputs,name='concat'):
        super(Concat,self).__init__(inputs,name,)
        outc=0
        for input in inputs:
            outc+=input[-1]
        dim=list(inputs[0].shape[:-1])+[outc]
        self.out=Blob(dim,self)

class Scale(Base):
    def __init__(self, input, factor=None, name='scale'):
        super(Scale, self).__init__(input, name, )
        self.out = input.new(self)

        self.dot=self.input_size
        # TODO scale analysis


