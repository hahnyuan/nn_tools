import numpy as np
import os
import struct
import base_datasets

class Mnist(base_datasets.BaseDataset):
    # read the raw mnist data
    def __init__(self, dir, only_test=0):
        # dir should include t10k-images-idx3-ubyte  t10k-labels-idx1-ubyte  train-images-idx3-ubyte  train-labels-idx1-ubyte
        # the data can be download at http://yann.lecun.com/exdb/mnist/
        self.dir=dir
        self.train_dir=[os.path.join(dir,'train-images-idx3-ubyte'),
                        os.path.join(dir,'train-labels-idx1-ubyte')]
        self.test_dir = [os.path.join(dir, 't10k-images-idx3-ubyte'),
                          os.path.join(dir, 't10k-labels-idx1-ubyte')]
        base_datasets.BaseDataset.__init__(self)
        self.test_data=self.get_test_data()
        if not only_test:
            self.train_data=self.get_train_data()
        self.ptest=0
        self.ptrain=0

    def norm(self,x):
        data = x.reshape([-1, 28, 28, 1]).astype(np.float32)
        data = (data - 167) / 167
        return data

    def get_train_data(self,norm=1):
        #return the train dataset as a list [images,labels]
        data=self.read_images(self.train_dir[0])
        if norm==1:
            data=self.norm(data)
        return [data,self.read_labels(self.train_dir[1])]

    def get_test_data(self,norm=1):
        #return the test dataset as a list [images,labels]
        data=self.read_images(self.test_dir[0])
        if norm==1:
            data=self.norm(data)
        return [data,self.read_labels(self.test_dir[1])]

    def shuffle(self,x):
        idx=np.arange(len(x[0]))
        np.random.shuffle(idx)
        for i,xi in enumerate(x):
            x[i]=xi[idx]
        return x

    def _batch(self,data,p,batch_size):
        return [i[p-batch_size:p] for i in data]

    def get_test_batch(self,batch_size):
        if self.ptest+batch_size>=len(self.test_data):
            self.ptest=0
            self.test_data=self.shuffle(self.test_data)
        self.ptest+=batch_size
        return self._batch(self.test_data,self.ptest,batch_size)

    def get_train_batch(self, batch_size):
        if self.ptrain + batch_size >= len(self.train_data):
            self.ptrain = 0
            self.train_data=self.shuffle(self.train_data)
        self.ptrain += batch_size
        return self._batch(self.train_data,self.ptrain,batch_size)

    def read_labels(self, file_name):
        """
           file_name:the byte file's direction
        """
        label_file=open(file_name,'rb')
        print(label_file)
        # get the basic information about the labels
        label_file.seek(0)
        magic_number = label_file.read(4)
        magic_number = struct.unpack('>i', magic_number)
        print('Magic Number: ' + str(magic_number[0]))

        data_type = label_file.read(4)
        data_type = struct.unpack('>i', data_type)
        print('Number of Lables: ' + str(data_type[0]))

        labels = []
        for idx in range(data_type[0]):
            label_file.seek(8 + idx)
            tmp_d = label_file.read(1)
            tmp_d = struct.unpack('>B', tmp_d)
            labels.append(tmp_d)
        return np.array(labels)


    def read_images(self, file_name):
        """
           file_name:the byte file's direction
        """
        img_file = open(file_name, 'rb')
        print(img_file)
        # get the basic information about the images
        img_file.seek(0)
        magic_number = img_file.read(4)
        magic_number = struct.unpack('>i', magic_number)
        print('Magic Number: ' + str(magic_number[0]))

        data_type = img_file.read(4)
        data_type = struct.unpack('>i', data_type)
        print('Number of Images: ' + str(data_type[0]))

        dim = img_file.read(8)
        dimr = struct.unpack('>i', dim[0:4])
        dimr = dimr[0]
        print('Number of Rows: ' + str(dimr))
        dimc = struct.unpack('>i', dim[4:])
        dimc = dimc[0]
        print('Number of Columns:' + str(dimc))


        images=[]
        for idx in range(data_type[0]):
            image = np.ndarray(shape=(dimr, dimc))
            img_file.seek(16 + dimc * dimr * idx)

            for row in range(dimr):
                for col in range(dimc):
                    tmp_d = img_file.read(1)
                    tmp_d = struct.unpack('>B', tmp_d)
                    image[row, col] = tmp_d[0]
            images.append(image)
        return np.array(images)
