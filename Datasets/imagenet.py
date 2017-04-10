from . import lmdb_datasets
from torchvision import datasets,transforms
import os
import os.path as osp
import torch.utils.data
import numpy as np
import cv2
from . import lmdb_data_pb2 as pb2
import Queue
import time
import multiprocessing

DATASET_SIZE=100

class Imagenet_LMDB(lmdb_datasets.LMDB):
    def __init__(self,imagenet_dir,train=False):
        self.train_name='imagenet_train_lmdb'
        self.val_name='imagenet_val_lmdb'
        self.train=train
        super(Imagenet_LMDB, self).__init__(osp.join(imagenet_dir,train and self.train_name or self.val_name))
        txn=self.env.begin()
        self.cur=txn.cursor()
        self.mean=np.array([[[0.485]], [[0.456]], [[0.406]]])
        self.std=np.array([[[0.229]], [[0.224]], [[0.225]]])
        self.data = Queue.Queue(maxsize=DATASET_SIZE*10)
        self.target = Queue.Queue(maxsize=DATASET_SIZE*10)

        self.read_from_lmdb()


    def data_transfrom(self,data,other):
        data=data.astype(np.float32)
        if self.train:
            shape=np.fromstring(other[0],np.uint16)
            data=data.reshape(shape)
            data=(data/255 - self.mean)/ self.std
            # Random crop
            _, w, h = data.shape
            x1 = np.random.randint(0, w - 224)
            y1 = np.random.randint(0, h - 224)
            data=data[:,x1:x1+224 ,y1:y1 + 224]
            # HorizontalFlip
            #TODO horizontal flip
            if data.shape!=(3,224,224):
                pass
            else:
                pass
        else:
            data = data.reshape([3, 224, 224])
            data = (data / 255 - self.mean) / self.std
        data = torch.FloatTensor(data)
        return data

    def target_transfrom(self,target):
        return target

    def read_from_lmdb(self):
        # r=time.time()
        self.cur.next()
        if not self.cur.key():
            self.cur.first()
        dataset=pb2.Dataset().FromString(self.cur.value())
        for datum in dataset.datums:
            data=np.fromstring(datum.data,np.uint8)
            try:
                data=self.data_transfrom(data,datum.other)
            except:
                print 'cannot trans ',data.shape
                continue
            target=int(datum.target)
            target=self.target_transfrom(target)
            self.data.put(data)
            self.target.put(target)
        # print 'read_from_lmdb', time.time()-r

    def __getitem__(self,index):
        if self.target.empty():
            self.read_from_lmdb()
        data,target=self.data.get(),self.target.get()
        return data,target

    def __len__(self):
        return self.env.stat()['entries']*DATASET_SIZE

def Imagenet_LMDB_generate(imagenet_dir, output_dir, make_val=False, make_train=False):
    # the imagenet_dir should have direction named 'train' or 'val',with 1000 folders of raw jpeg photos
    train_name = 'imagenet_train_lmdb'
    val_name = 'imagenet_val_lmdb'

    def target_trans(target):
        return target

    if make_val:
        val_lmdb=lmdb_datasets.LMDB_generator(osp.join(output_dir,val_name))
        def trans_val_data(dir):
            tensor = transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ])(dir)
            tensor=(tensor.numpy()*255).astype(np.uint8)
            return tensor

        val = datasets.ImageFolder(osp.join(imagenet_dir,'val'), trans_val_data,target_trans)
        val_lmdb.write_classification_lmdb(val, num_per_dataset=DATASET_SIZE)
    if make_train:
        train_lmdb = lmdb_datasets.LMDB_generator(osp.join(output_dir, train_name))
        def trans_train_data(dir):
            tensor = transforms.Compose([
                transforms.Scale(256),
                transforms.ToTensor()
            ])(dir)
            tensor=(tensor.numpy()*255).astype(np.uint8)
            return tensor

        train = datasets.ImageFolder(osp.join(imagenet_dir, 'train'), trans_train_data, target_trans)
        train.imgs=np.random.permutation(train.imgs)

        train_lmdb.write_classification_lmdb(train, num_per_dataset=DATASET_SIZE, write_shape=True)

