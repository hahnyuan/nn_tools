import numpy as np

# the blob with shape of (h,w,c) or (batch,h,w,c) for image
class Blob():
    def __init__(self,shape,father=None):
        shape=[int(i) for i in shape]
        self.data=np.ones(shape)
        self.shape=self.data.shape
        self.father=type(father)==list and father or [father]

    @property
    def size(self):
        return np.prod(self.shape)

    def new(self,father):
        return Blob(self.data.shape,father)
    def __getitem__(self, key):
        return self.data.shape[key]
    def __str__(self):
        return str(self.data.shape)
    def flaten(self):
        return Blob([np.prod(self.data.shape)])