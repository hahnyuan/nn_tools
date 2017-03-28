import numpy as np
from .blob import Blob
from .layers import Base


class Flatten(Base):
    def __init__(self,input, name='permute'):
        super(Flatten, self).__init__(input, name)
        dim=[np.prod(input.data.shape)]
        self.out = Blob(dim, self)

class PSROIPool(Base):
    def __init__(self,input,rois,output_dim,group_size,name='psroipool'):
        super(PSROIPool,self).__init__(input,name)
        self.rois=rois
        dim=[rois.shape[0],output_dim,group_size,group_size]
        self.out=Blob(dim,self)
        self.layer_info='output_dim:%d,group_size:%d'%(output_dim,group_size)

        # TODO PSROIPOOL ANALYSIS