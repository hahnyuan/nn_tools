import numpy as np

box=[]
class Base(object):
    def __init__(self,input,name=''):
        if isinstance(input,Base):
            input=input()
        self.name=name
        self.input=input
        self.weight_size=0
        self.input_size=np.prod(self.input)
        self.dot=0
        self.add=0
        self.pow=0
        self.compare=0
        self.flops=0
        self.out=None
        box.append(self)

    def __call__(self, *args, **kwargs):
        return self.out

class Activation(Base):
    #valid tuple lists the valid activation function type
    valid_tuple=('relu','tanh','prelu')
    def __init__(self,input,type,name=None):
        if type not in Activation.valid_tuple:
            raise NameError,'the activation type:'+type+' is not supported. ' \
                            'the valid type is: '+str(Activation.valid_tuple)
        if name==None:name=type
        Base.__init__(self,input,name=name)
        getattr(self,type)()
        self.out=self.input

    def relu(self):
        self.compare=self.input_size

    def sigmoid(self):
        self.add=self.dot=self.pow=self.input_size

    def tanh(self):
        self.dot=self.input_size
        self.add=self.pow=self.input_size*2


class Sliding(Base):
    def __init__(self,input,kernel_size,num_out,stride=1,pad=0,name='sliding'):
        # input (w,h,c)
        super(Sliding,self).__init__(input,name=name)
        self.kernel_size=kernel_size
        self.num_out=num_out
        self.stride=stride
        self.pad=pad
        #calc out
        out_w=np.floor(float(self.input[0]+pad*2-kernel_size)/stride)+1
        out_h=np.floor(float(self.input[1]+pad*2-kernel_size)/stride)+1
        self.out=np.array([out_w,out_h,num_out])

class Conv(Sliding):
    def __init__(self,input,kernel_size,num_out,stride=1,pad=0,activation=None,name='conv'):
        Sliding.__init__(self,input,kernel_size,num_out,stride,pad,name=name)
        self.dot = self.out[0] * self.out[1] * self.input[2] * self.kernel_size ** 2 * self.num_out
        self.add = self.dot
        if activation:
            Activation(self.out,activation)

class Pool(Sliding):
    def __init__(self,input,kernel_size,stride=1,pad=0,name='pool'):
        Sliding.__init__(self,input,kernel_size,input[2],stride,pad,name=name)
        self.compare=self.out[0]*self.out[1]*(kernel_size**2-1)*self.num_out

class InnerProduct(Base):
    def __init__(self,input,num_out,activation=None,name='innerproduct'):
        Base.__init__(self,input,name=name)
        self.input=np.prod(self.input)
        self.num_out=num_out
        self.dot=self.num_out*self.input
        self.add=self.num_out*self.input
        self.out=np.array([self.num_out])
        if activation:
            Activation(self.out,activation)
Fc=InnerProduct
