import numpy as np

class BaseDataset():
    # the father of all datasets class, there are some interface was defined
    def __init__(self):
        # the data saved in the class was a list with the format [data,label,...]
        self.data={}
        self.init_data()
        self.p={'train':0,'test':0,'val':0}

    def init_data(self):
        pass

    def get_test_data(self):
        pass

    def get_train_data(self):
        pass

    def get_val_data(self):
        pass

    def _shuffle(self, x):

        idx = np.arange(len(x[0]))
        np.random.shuffle(idx)
        for i, xi in enumerate(x):
            x[i] = xi[idx]
        return x

    def batch(self, name, batch_size):
        p = self.p[name]
        if p + batch_size >= len(self.data[name]):
            self.p[name] = 0
            self.data[name] = self._shuffle(self.data[name])
        self.p[name] += batch_size
        return [i[p:p+batch_size] for i in self.data[name]]