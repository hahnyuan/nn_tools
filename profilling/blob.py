import numpy as np

class Blob():
    def __init__(self,shape,father=None):
        self.data=np.ones(shape)
        self.father=type(father)==list and father or [father]
    def new(self,father):
        return Blob(self.data.shape,father)
    def __getitem__(self, key):
        return self.data.shape[key]
    def __str__(self):
        return str(self.data.shape)
    def flaten(self):
        return Blob([np.prod(self.data.shape)])