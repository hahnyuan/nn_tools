import numpy as np

box=[]
class Base(object):
    def __init__(self,input,name=''):
        if isinstance(input,Base):
            input=input()
        self.name=name
        self.input=input
        self.weight_size=0
        self.blob_size=None
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
    def __setattr__(self, key, value):
        if key=='out':
            self.blob_size=np.prod(value)
        return object.__setattr__(self, key,value)


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
    def __init__(self,input,kernel_size,num_out,stride=1,pad=0,name='sliding',ceil=False):
        # input (w,h,c)
        super(Sliding,self).__init__(input,name=name)
        self.kernel_size=kernel_size
        self.num_out=num_out
        self.stride=stride
        self.pad=pad
        #calc out
        if not ceil:
            out_w=np.floor(float(self.input[0]+pad*2-kernel_size)/stride)+1
            out_h=np.floor(float(self.input[1]+pad*2-kernel_size)/stride)+1
        else:
            out_w = np.ceil(float(self.input[0] + pad * 2 - kernel_size) / stride) + 1
            out_h = np.ceil(float(self.input[1] + pad * 2 - kernel_size) / stride) + 1
        self.out=np.array([out_w,out_h,num_out])

class Conv(Sliding):
    def __init__(self,input,kernel_size,num_out,stride=1,pad=0,activation=None,name='conv',ceil=False):
        if isinstance(input,Base):
            input=input()
        Sliding.__init__(self,input,kernel_size,num_out,stride,pad,name=name,ceil=ceil)
        self.dot = self.out[0] * self.out[1] * self.input[2] * self.kernel_size ** 2 * self.num_out
        self.add = self.dot
        self.weight_size=self.kernel_size**2*num_out*input[2]
        if activation:
            Activation(self.out,activation)
conv=Conv

class Pool(Sliding):
    def __init__(self,input,kernel_size,stride=1,pad=0,name='pool',ceil=False):
        if isinstance(input,Base):
            input=input()
        Sliding.__init__(self,input,kernel_size,input[2],stride,pad,name=name,ceil=ceil)
        self.compare=self.out[0]*self.out[1]*(kernel_size**2-1)*self.num_out
pool=Pool

class InnerProduct(Base):
    def __init__(self,input,num_out,activation=None,name='innerproduct'):
        if isinstance(input,Base):
            input=input()
        Base.__init__(self,input,name=name)
        self.input=np.prod(self.input)
        self.num_out=num_out
        self.dot=self.num_out*self.input
        self.add=self.num_out*self.input
        self.out=np.array([self.num_out])
        self.weight_size = self.num_out * self.input
        if activation:
            Activation(self.out,activation)
Fc=InnerProduct
fc=InnerProduct

def save_csv(csv_save_path,save_items=('name', 'input', 'out', 'dot', 'add', 'compare', 'weight_size','blob_size')):
    import csv
    if csv_save_path!=None:
        with open(csv_save_path,'w') as file:
            writer=csv.writer(file)
            writer.writerow(save_items)
            for layer in box:
                writer.writerow([getattr(layer,param) for param in save_items])

    for layer in box:
        print(','.join(str(j) for j in [getattr(layer,param) for param in save_items]))
    print 'saved!'