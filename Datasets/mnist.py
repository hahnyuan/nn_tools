import numpy as np
import os
import struct

class Mnist():
    # read the raw mnist data
    def __init__(self,dir):
        # dir should include t10k-images-idx3-ubyte  t10k-labels-idx1-ubyte  train-images-idx3-ubyte  train-labels-idx1-ubyte
        # the data can be download at http://yann.lecun.com/exdb/mnist/
        self.dir=dir
        self.train_dir=[os.path.join(dir,'train-images-idx3-ubyte'),
                        os.path.join(dir,'train-labels-idx1-ubyte')]
        self.test_dir = [os.path.join(dir, 't10k-images-idx3-ubyte'),
                          os.path.join(dir, 't10k-labels-idx1-ubyte')]

    def train(self):
        return self.read_images(self.train_dir[0])

    def test(self):
        return self.read_images(self.test_dir[0]),self.read_labels(self.test_dir[1])

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
